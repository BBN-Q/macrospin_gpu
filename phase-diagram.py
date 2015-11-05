#!/usr/bin/python

from __future__ import division # Prevent integer division errors
import pycuda.autoinit
import pycuda.driver   as cuda
import pycuda.gpuarray as gpua
from   pycuda.compiler import SourceModule
import pycuda.curandom
from   jinja2 import Template
import numpy as np
from   scipy import interpolate
from   scipy.ndimage.filters import gaussian_filter
from   demag import demagCylinder
from   tqdm import *
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

kernelFile = open("kernel.cu")           # Import the CUDA kernel
modTemp    = Template(kernelFile.read()) # Create Template

# Some helper functions
def doubleElements(array):
    """ returns a list with each element duplicated """
    return np.vstack((array, array)).ravel([-1])

def arrayAvg(array):
    """ Quickly average entire columns as arrays """
    total = np.zeros_like(array[0])
    for thing in array:
        total += thing
    return total/len(array)

# Constants in CGS, so watch out for AbAmps and such!
ech   = 1.6022e-20
hbar  = 6.6261e-27 / (2.0*np.pi)
muB   = 9.2740e-21
kB    = 1.3807e-16
g     = 2.0
gamma = g*muB / hbar

# Geometry in cm 
pillar_length =  45.0e-07
pillar_width  =  30.0e-07
pillar_thick  =   1.5e-07
pillar_area   = np.pi*pillar_length*pillar_width/4.0
pillar_vol    = pillar_area*pillar_thick

Nxx, Nyy, Nzz = demagCylinder(pillar_length, pillar_width, pillar_thick, cgs=True)

# Other Constants
Ms          = 1000.0
temperature = 4.0
timeUnit    = 1.0/(gamma*Ms)
realdt      = 1.0e-13
dt          = realdt/timeUnit  # time in units of $\gamma M_s$
sqrtDt      = np.sqrt(dt)
damping     = 0.05
pol_stt     = 0.13
pol_flt     = 0.00
nu          = np.sqrt(2.0*damping*kB*temperature/(pillar_vol*Ms*Ms))
sttPre      = (pol_stt*hbar*0.1)/(2.0*ech*Ms*Ms*pillar_thick)
fltPre      = (pol_flt*hbar*0.1)/(2.0*ech*Ms*Ms*pillar_thick)

# Pulse characteristics
nTron_pulse       = False
minCurrent        = 0.0
maxCurrent        = 13.0e8
minTilt           = 0.0
maxTilt           = 180
pulse_duration    = 0.1e-9
pause_before      = 0.3e-9
pause_after       = 8.0e-9
total_time        = pulse_duration + pause_before + pause_after
total_steps       = int(total_time/realdt)
normalizeInterval = 50

def gaussian_pulse_shape(time, rise_time=65.0e-12, fall_time=100e-12):
    # HWHM to sigma
    rise_time = rise_time/(2.0*2.3548)
    fall_time = fall_time/(2.0*2.3548)
    if time<pause_before:
        return np.exp(-((time-pause_before)**2)/(2*rise_time**2))
    elif time<(pause_before+pulse_duration):
        return 1.0
    elif time>(pause_before+pulse_duration):
        return np.exp(-((time-pause_before-pulse_duration)**2)/(2*fall_time**2))
    else:
        return 0.0

# Import nTron pulses, which starts at 1.0ns and continues to 10ns
n_times, n_voltages = np.loadtxt('ntron.dat', unpack = True)
n_times             = n_times*1e-12
n_amps              = n_voltages/n_voltages.max()
n_cutoff            = np.where(n_times>=1.0e-9)[0][0]
i_times             = n_times[n_cutoff:]-1.0e-9
i_amps              = n_amps[n_cutoff:]
nTron_pulse_interp  = interpolate.interp1d(i_times, i_amps)
def nTron_pulse_shape(time):
    if (time>pause_before) and (time<(pause_before+i_times[-1])):
        return nTron_pulse_interp(time-pause_before)
    else:
        return 0.0

# External Field
hExtX     = 0.0
hExtY     = 0.0
hExtZ     = 0.0

# Perpendicular Anisotropy
hPMA      = 2000.0
Nzz       = Nzz - Nzz - hPMA/Ms # from compensation

# Kernel Configuration Constants
threadsPerBlock    = 256  # Averages over different thermal realizations
blocksX            = 256
blocksY            = 256
N                  = threadsPerBlock*blocksX*blocksY

# Render the CUDA module metaprogram template
modRend = modTemp.render(
    alpha           = damping,
    dt              = dt,
    sqrtDt          = sqrtDt,
    nuSqrtDt        = nu*sqrtDt,
    nu              = nu,
    nu2             = nu*nu,
    nxx             = Nxx,
    nyy             = Nyy,
    nzz             = Nzz,
    hx              = hExtX/Ms,
    hy              = hExtY/Ms,
    hz              = hExtZ/Ms,
    stt             = sttPre/damping,
    flt             = fltPre,
    minCurrent      = minCurrent,
    maxCurrent      = maxCurrent,
    currentRange    = (maxCurrent-minCurrent),
    minTilt         = minTilt*np.pi/180.0,
    maxTilt         = maxTilt*np.pi/180.0,
    tiltRange       = (maxTilt-minTilt)*np.pi/180.0,
    )

# Load Module Function Pointers   
mod                    = SourceModule(modRend)
evolve_current_vs_tilt = mod.get_function("evolve_current_vs_tilt")
normalizeM             = mod.get_function("normalizeM")
resetM                 = mod.get_function("resetM")
reduceM                = mod.get_function("reduceM")

# Random number generator
rgen   = pycuda.curandom.XORWOWRandomNumberGenerator()

# Initialization...
m0      = np.zeros((N,4))
m0[:,:] = [0.0, 0.0, 1.0, 0.0]

# For the simulations
m_gpu        = gpua.to_gpu(m0.astype(np.float32))
dW_gpu       = gpua.zeros((threadsPerBlock,4), dtype=np.float32)

# For storing the resulting phase diagram
phase_gpu    = gpua.empty((blocksX*blocksY,1), dtype=np.float32)

def current_tilt():
    """For current in mA, duration in seconds, field_x in Oe, 
    anisotropy in Oe (not w.r.t. compensation)"""

    resetM(m_gpu, np.float32(np.sqrt(0.0)), np.float32(0.0), np.float32(1.00),
            grid=(blocksX,blocksY), block=(threadsPerBlock,1,1))
    
    for i, t in enumerate(tqdm(np.linspace(0.0, total_time, total_steps), desc='Evolving', leave=True)):
        if temperature > 0:
            rgen.fill_normal(dW_gpu)
        pulse_value = np.float32(nTron_pulse_shape(t) if nTron_pulse else gaussian_pulse_shape(t))
        evolve_current_vs_tilt(m_gpu, dW_gpu, np.float32(t), pulse_value,
                               grid=(blocksX,blocksY), block=(threadsPerBlock,1,1))
        if (i%(normalizeInterval)==0):
            normalizeM(m_gpu, grid=(blocksX,blocksY), block=(threadsPerBlock,1,1))

    reduceM(m_gpu, phase_gpu, np.int32(threadsPerBlock), grid=(blocksY,1), block=(blocksX,1,1))
    diagram        = phase_gpu.get()
    return np.transpose(np.reshape(diagram, (blocksY, blocksX))) #, np.transpose(np.reshape(timing_diagram, (blocksY, blocksX)))

if __name__ == '__main__':

    phase_diagram = current_tilt()

    np.savetxt("PhaseDiagram-TiltCurrent-%0.2fns-Hpma%.2fOe-%dK%s.txt" % (pulse_duration*1e9, hPMA, temperature, '-nTron' if nTron_pulse else ''),  np.transpose(phase_diagram))

    pt = pulse_duration*1e9
    maxCurrent = maxCurrent*1e-8
    minCurrent = minCurrent*1e-8

    extent = (minTilt, maxTilt, minCurrent, maxCurrent)
    aspect = (maxTilt-minTilt)/(maxCurrent-minCurrent)
    plt1 = plt.figure(1)
    plt.imshow(phase_diagram, origin='lower', extent=extent, aspect=aspect, cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt1.suptitle('Switching Probability %0.1fns %s pulse' % (pt, ' nTron' if nTron_pulse else 'square'))
    plt1.gca().set_xlabel('Polarizer Tilt (deg)')
    plt1.gca().set_ylabel(r'Current Density (10$^8$A/cm$^2$)')

    # Extract the boundary surface at 50%
    intersections_tilt = np.where(np.diff(np.sign(np.transpose(phase_diagram)-0.5)))[0]
    intersections_curr = np.where(np.diff(np.sign(np.transpose(phase_diagram)-0.5)))[1]

    currents = np.linspace(minCurrent, maxCurrent, blocksX)[intersections_curr]
    tilts = np.linspace(minTilt, maxTilt, blocksY)[intersections_tilt]
    plt1.gca().set_xlim(minTilt, maxTilt)
    plt1.gca().set_ylim(minCurrent, maxCurrent)
    plt.show(1)
    plt.savefig('PhaseDiagram-TiltCurrent-%0.2fns-Hpma%.2fOe-%dK%s.png' % (pt, hPMA, temperature, '-nTron' if nTron_pulse else ''))

    print "Done..."

    
