#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np

# PyQt4 imports
from   PyQt4 import QtGui, QtCore, QtOpenGL
from   PyQt4.QtOpenGL import QGLWidget

# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.arrays.vbo as glvbo

# PyOpenCL imports
import pyopencl as cl
import pyopencl.clrandom as ran
from   pyopencl.tools import get_gl_sharing_context_properties
from   jinja2 import Template

# Physics imports
from   demag import demagCylinder

# Other imports
import matplotlib.pyplot as plt
from   tqdm import *
import time

# Define the GPU Context
ctx   = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf    = cl.mem_flags

# Define random number generator
rg = ran.RanluxGenerator(queue) 

# Load kernel
with open("costm-amp-dur.cl") as template_file:
    kernel_template = Template(template_file.read())

# Constants in CGS, so watch out for AbAmps and such!
ech   = 1.6022e-20
hbar  = 6.6261e-27 / (2.0*np.pi)
muB   = 9.2740e-21
kB    = 1.3807e-16
g     = 2.0
gamma = g*muB / hbar

# Geometry in cm 
pillar_length = 100.0e-07
pillar_width  = 50.0e-07
pillar_thick  = 2.0e-07
pillar_area   = np.pi*pillar_length*pillar_width/4.0
pillar_vol    = pillar_area*pillar_thick

# Other Constants
Ms          = 640.0             # Saturation magnetization (emu/cm^3)
temperature = 40.0               # System temperature (K)
timeUnit    = 1.0/(gamma*Ms)    # Reduced units for numerical convnience
realdt      = 2.0e-13           # Actual time step (s)
dt          = realdt/timeUnit   # time in units of $\gamma M_s$
sqrtDt      = np.sqrt(dt)       # Convenience
damping     = 0.05              # Gilbert damping factor
pol_op      = 0.16              # Spin torque polarization OP
pol_ip      = 0.37              # Spin torque polarization IP
lambda_op   = 1.49              # Spin torque asymmetry OP
lambda_ip   = 1.49              # Spin torque asymmetry IP
nu          = np.sqrt(2.0*damping*kB*temperature/(pillar_vol*Ms*Ms)) # Width of thermal distribution

# Spin torque prefactors, things crammed together for numerical efficiency
stt_op_pre  = 2.0*(lambda_op**2)*pol_op*hbar*0.1/(2.0*ech*Ms*Ms*pillar_thick*damping) # 0.1 for Amps->abAmps
stt_ip_pre  = 2.0*(lambda_ip**2)*pol_ip*hbar*0.1/(2.0*ech*Ms*Ms*pillar_thick*damping) # 0.1 for Amps->abAmps

Nxx, Nyy, Nzz = demagCylinder(pillar_length, pillar_width, pillar_thick, cgs=True)

# Pulse characteristics
min_current       = 0.0e8 # A/cm^2
max_current       = 0.8e8
min_duration      = 0.0e-12
max_duration      = 1.4e-9
pause_before      = 0.3e-9
pause_after       = 2.0e-9
total_time        = max_duration + pause_before + pause_after
total_steps       = int(total_time/realdt)
normalizeInterval = 50

# External Field
hExtX     = 0.0
hExtY     = 0.0
hExtZ     = 0.0

# Perpendicular Anisotropy
hPMA      = 0000.0
Nzz       = Nzz - hPMA/Ms # from compensation

# Render the CUDA module metaprogram template
kernel = kernel_template.render(
        alpha             = damping,
        dt                = dt,
        sqrtDt            = sqrtDt,
        nuSqrtDt          = nu*sqrtDt,
        nu                = nu,
        nu2               = nu*nu,
        nxx               = Nxx,
        nyy               = Nyy,
        nzz               = Nzz,
        hx                = hExtX/Ms,
        hy                = hExtY/Ms,
        hz                = hExtZ/Ms,
        stt_op            = stt_op_pre,
        stt_ip            = stt_ip_pre,
        lambda2_plus1_op  = lambda_op**2 + 1.0,
        lambda2_plus1_ip  = lambda_ip**2 + 1.0,
        lambda2_minus1_op = lambda_op**2 - 1.0,
        lambda2_minus1_ip = lambda_ip**2 - 1.0,
        rise_time         = 65.0e-12,
        fall_time         = 100.0e-12,
        pause_before      = pause_before,
        pause_after       = pause_after,
    )
with open('rendered-kernel.cl', 'w') as f:
    f.write(kernel)

# Initialize program and setup argument types for kernels
prg = cl.Program(ctx, kernel).build()
evolve = prg.evolve
evolve.set_scalar_arg_dtypes([None, None, None, None, np.float32])
reduce_m = prg.reduce_m
reduce_m.set_scalar_arg_dtypes([None, None, np.int32])
normalize_m = prg.normalize_m

# Data dimensions
realizations   = 64 # Averages over different realizations of the noise process
current_steps  = 64
duration_steps = 64
N              = current_steps*duration_steps*realizations
time_points    = 20 # How many points to store as a function of time
time_interval  = 200 # The interval between adding new points

# Declare the GPU bound arrays
m             = cl.array.zeros(queue, N, cl.array.vec.float4)
m_of_t        = cl.array.zeros(queue, N*time_points, cl.array.vec.float4)
dW            = cl.array.zeros(queue, N, cl.array.vec.float4)
phase_diagram = cl.array.zeros(queue, current_steps*duration_steps, np.float32)

def amp_dur():

    initial_m = np.zeros(N, dtype=cl.array.vec.float4)
    initial_m[:] = (1,0,0,0)
    cl.enqueue_copy(queue, m.data, initial_m)

    # Create the GPU buffers that contain the phase diagram parameters
    durations_np  = np.linspace(min_duration, max_duration, duration_steps).astype(np.float32)
    currents_np   = np.linspace(min_current, max_current, current_steps).astype(np.float32)
    durations     = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=durations_np)
    currents      = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=currents_np)
    
    # Time evolution taken care of by the host, GPU integrates individual time steps.
    for i, t in enumerate(tqdm(np.linspace(0.0, total_time, total_steps), desc='Evolving', leave=True)):

        # Generate random numbers for the stochastic process
        if temperature > 0:
            rg.fill_normal(dW)

        # Run a single LLG evolution step
        evolve(queue, (current_steps*realizations, duration_steps), (realizations, 1), m.data, dW.data, currents, durations, t).wait()

        # Periodic Normalizations
        if (i%(normalizeInterval)==0):
            normalize_m(queue, (current_steps*realizations, duration_steps), (realizations, 1), m.data,).wait()

    reduce_m(queue, (current_steps, duration_steps), (1,1), m.data, phase_diagram.data, realizations).wait()
    return phase_diagram.get().reshape(duration_steps, current_steps).transpose()


if __name__ == '__main__':
    phase_diagram = amp_dur()

    min_duration = min_duration*1e9
    max_duration = max_duration*1e9
    max_current = max_current*1e-8
    min_current = min_current*1e-8

    extent = (min_duration, max_duration, min_current, max_current)
    aspect = (max_duration-min_duration)/(max_current-min_current)
    plt1 = plt.figure(1)
    plt.imshow(phase_diagram, origin='lower', extent=extent, aspect=aspect, cmap=plt.get_cmap('Blues'))
    plt.colorbar()
    plt1.suptitle('Switching Probability')
    plt1.gca().set_xlabel('Pulse Duration (ns)')
    plt1.gca().set_ylabel(r'Current Density (10$^8$A/cm$^2$)')
    plt.show()
