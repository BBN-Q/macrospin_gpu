#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
import pyopencl.clrandom as ran
from   jinja2 import Template
import matplotlib.pyplot as plt
from   demag import demagCylinder
from   tqdm import *
import time

ctx   = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf    = cl.mem_flags

# Define random number generator
rg = ran.RanluxGenerator(queue) 

# Load kernel
with open("kernel.cl") as template_file:
    kernel_template = Template(template_file.read())

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
minCurrent        = 0.0e9
maxCurrent        = 2.8e8
minTilt           = 0.0
maxTilt           = 90.0
pulse_duration    = 0.8e-9
pause_before      = 0.5e-9
pause_after       = 0.5e-9
total_time        = pulse_duration + pause_before + pause_after
total_steps       = int(total_time/realdt)
normalizeInterval = 50

# External Field
hExtX     = 0.0
hExtY     = 0.0
hExtZ     = 0.0

# Perpendicular Anisotropy
hPMA      = 4000.0
Nzz       = Nzz - hPMA/Ms # from compensation

# Render the CUDA module metaprogram template
kernel = kernel_template.render(
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
    )

# Initialize program and setup argument types for kernels
prg = cl.Program(ctx, kernel).build()
evolve = prg.evolve
evolve.set_scalar_arg_dtypes([None, None, None, None, np.float32])
reduce_m = prg.reduce_m
reduce_m.set_scalar_arg_dtypes([None, None, np.int32])
normalize_m = prg.normalize_m

# Data dimensions
realizations  = 1  # Averages over different thermal realizations
current_steps = 64
tilt_steps    = 32
N             = current_steps*tilt_steps*realizations

m             = cl.array.zeros(queue, N, cl.array.vec.float4)
dW            = cl.array.zeros(queue, N, cl.array.vec.float4)
phase_diagram = cl.array.zeros(queue, current_steps*tilt_steps, np.float32)

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

def current_tilt():

    initial_m = np.zeros(N, dtype=cl.array.vec.float4)
    initial_m[:] = (1,0,0,0)
    cl.enqueue_copy(queue, m.data, initial_m)

    tilts_np    = np.linspace(minTilt, maxTilt, tilt_steps).astype(np.float32)*np.pi/180.0
    currents_np = np.linspace(minCurrent, maxCurrent, current_steps).astype(np.float32)

    tilts       = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=tilts_np)
    currents    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=currents_np)
    
    ms = []
    # evolve(queue, (current_steps*realizations, tilt_steps), (realizations, 1), m.data, dW.data, currents, tilts, 0.0).wait()

    for i, t in enumerate(tqdm(np.linspace(0.0, total_time, total_steps), desc='Evolving', leave=True)):
    #   for t in np.linspace(0.0, 1.0e-12, 10):
        # Fill out the Weiner process
        if temperature > 0:
            rg.fill_normal(dW)

        # Where are we in the pulse?
        pulse_value = gaussian_pulse_shape(t)
        # Actually evolve
        evolve(queue, (current_steps*realizations, tilt_steps), (realizations, 1), m.data, dW.data, currents, tilts, pulse_value).wait()

        if (i%(normalizeInterval)==0):
            ms.append(m.get()['x'][0])
            normalize_m(queue, (current_steps*realizations, tilt_steps), (realizations, 1), m.data,).wait()

    # plt.plot(ms)
    # plt.show()
    reduce_m(queue, (current_steps*realizations, tilt_steps), (realizations, 1), m.data, phase_diagram.data, realizations).wait()
    return m.get()['x'].reshape( tilt_steps, current_steps).transpose()
    # return ms

if __name__ == '__main__':
    phase_diagram = current_tilt()
    # print("ALSKDjalskdjasl")
    # plt.plot(phase_diagram)
    # plt.show()
    # np.savetxt("PhaseDiagram-TiltCurrent-%0.2fns-Hpma%.2fOe-%dK%s.txt" % (pulse_duration*1e9, hPMA, temperature, '-nTron' if nTron_pulse else ''),  np.transpose(phase_diagram))

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
    plt.show()

# rg.fill_normal(dW)

# m_local = m.get()
# plt.hist(m_local['x'], 50, normed=1, facecolor='red', alpha=0.25)
# plt.hist(m_local['y'], 50, normed=1, facecolor='green', alpha=0.25)
# plt.hist(m_local['z'], 50, normed=1, facecolor='blue', alpha=0.25)
# do_something = prg.do_something
# do_something.set_scalar_arg_dtypes([None, np.float32])
# do_something(queue, m.shape, None, m.data, 4.0)
# m_local = m.get()
# plt.hist(m_local['x'], 50, normed=1, facecolor='red', alpha=0.25)
# plt.hist(m_local['y'], 50, normed=1, facecolor='green', alpha=0.25)
# plt.hist(m_local['z'], 50, normed=1, facecolor='blue', alpha=0.25)


# res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
# prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

# res_np = np.empty_like(a_np)
# cl.enqueue_copy(queue, res_np, res_g)

# n, bins, patches = plt.hist(c_np.get(), 100, normed=1, facecolor='green', alpha=0.75)
# print(bins)

# # Check on CPU with Numpy:
# print(res_np - (a_np + b_np))
# print(np.linalg.norm(res_np - (a_np + b_np)))
# plt.show()