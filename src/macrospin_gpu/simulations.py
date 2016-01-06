#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Numerical libraries
import numpy as np

# PyOpenCL imports
import pyopencl as cl
import pyopencl.clrandom as ran

# Progress bar
from   tqdm import *

# Python builtins
import sys
import os
import time

class Simulation2D(object):
    """docstring for Simulation2D"""
    def __init__(self, macrospin_object):
        super(Simulation2D, self).__init__()

        self.mo = macrospin_object
        kernel_text = self.mo.render_kernel()

        self.dirname = os.path.dirname(__file__)
        with open(self.dirname+'/templates/rendered-kernel.cl', 'w') as f:
            f.write(kernel_text)

        self.ctx    = cl.create_some_context()
        self.queue  = cl.CommandQueue(self.ctx)
        self.prg    = cl.Program(self.ctx, kernel_text).build()
        
        self.evolve = self.prg.evolve
        self.evolve.set_scalar_arg_dtypes([None, None, None, None, np.float32])

        self.reduce_m = self.prg.reduce_m
        self.reduce_m.set_scalar_arg_dtypes([None, None, np.int32])

        self.normalize_m = self.prg.normalize_m

        # Define random number generator
        self.ran_gen = ran.RanluxGenerator(self.queue, luxury=0)

        # Declare the GPU bound arrays
        self.m             = cl.array.zeros(self.queue, self.mo.N, cl.array.vec.float4)
        self.dW            = cl.array.zeros(self.queue, self.mo.N, cl.array.vec.float4)
        self.phase_diagram = cl.array.zeros(self.queue, self.mo.pixels, np.float32)

        # Fill out the magnetization initial conditions and push to card
        self.initial_m = np.zeros(self.mo.N, dtype=cl.array.vec.float4)
        self.initial_m[:] = tuple(self.mo.initial_m)
        cl.enqueue_copy(self.queue, self.m.data, self.initial_m)

        # Phase diagram values
        self.first_vals     = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mo.first_vals_np)
        self.second_vals    = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mo.second_vals_np)

    def run(self):
        # Time evolution taken care of by the host, GPU integrates individual time steps.
        for i, t in enumerate(tqdm(np.linspace(0.0, self.mo.total_time, self.mo.total_steps), desc='Evolving', leave=True)):

            # Generate random numbers for the stochastic process
            if self.mo.temperature > 0:
                self.ran_gen.fill_normal(self.dW)

            # Run a single LLG evolution step
            self.evolve(self.queue,
                        (self.mo.first_val_steps*self.mo.thermal_realizations, self.mo.second_val_steps),
                        (self.mo.thermal_realizations, 1),
                        self.m.data, self.dW.data, self.first_vals, self.second_vals, t).wait()

            # Periodic Normalizations
            if (i%(self.mo.normalize_interval)==0):
                self.normalize_m(self.queue,
                                 (self.mo.first_val_steps*self.mo.thermal_realizations, self.mo.second_val_steps),
                                 (self.mo.thermal_realizations, 1),
                                 self.m.data).wait()

    def get_phase_diagram(self):
        self.reduce_m(self.queue,
                      (self.mo.first_val_steps, self.mo.second_val_steps),
                      (1,1),
                      self.m.data, self.phase_diagram.data, self.mo.thermal_realizations).wait()

        return self.phase_diagram.get().reshape(self.mo.second_val_steps, self.mo.first_val_steps).transpose()


class Simulation2D_Dipolar(object):
    """docstring for Simulation2D_Dipolar"""
    def __init__(self, macrospin_object):
        super(Simulation2D_Dipolar, self).__init__()

        self.mo = macrospin_object
        kernel_text = self.mo.render_kernel()

        self.dirname = os.path.dirname(__file__)
        with open(self.dirname+'/templates/rendered-kernel.cl', 'w') as f:
            f.write(kernel_text)

        self.ctx    = cl.create_some_context()
        self.queue  = cl.CommandQueue(self.ctx)
        self.prg    = cl.Program(self.ctx, kernel_text).build()
        
        self.evolve = self.prg.evolve
        self.evolve.set_scalar_arg_dtypes([None, None, None, None, None, np.float32])

        self.reduce_m = self.prg.reduce_m
        self.reduce_m.set_scalar_arg_dtypes([None, None, np.int32])

        self.normalize_m = self.prg.normalize_m

        # Define random number generator
        self.ran_gen = ran.RanluxGenerator(self.queue, luxury=0)

        # Declare the GPU bound arrays
        self.m             = cl.array.zeros(self.queue, self.mo.N, cl.array.vec.float4)
        self.m_ref         = cl.array.zeros(self.queue, self.mo.N, cl.array.vec.float4)
        self.dW            = cl.array.zeros(self.queue, self.mo.N, cl.array.vec.float4)
        self.phase_diagram = cl.array.zeros(self.queue, self.mo.pixels, np.float32)

        # Fill out the magnetization initial conditions and push to card
        self.initial_m = np.zeros(self.mo.N, dtype=cl.array.vec.float4)
        self.initial_m[:] = tuple(self.mo.initial_m)
        cl.enqueue_copy(self.queue, self.m.data, self.initial_m)

        self.initial_m_ref = np.zeros(self.mo.N, dtype=cl.array.vec.float4)
        self.initial_m_ref[:] = tuple(self.mo.initial_m_ref)
        cl.enqueue_copy(self.queue, self.m_ref.data, self.initial_m_ref)

        # Phase diagram values
        self.first_vals     = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mo.first_vals_np)
        self.second_vals    = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.mo.second_vals_np)

    def run(self):
        # Time evolution taken care of by the host, GPU integrates individual time steps.
        for i, t in enumerate(tqdm(np.linspace(0.0, self.mo.total_time, self.mo.total_steps), desc='Evolving', leave=True)):

            # Generate random numbers for the stochastic process
            if self.mo.temperature > 0:
                self.ran_gen.fill_normal(self.dW)

            # Run a single LLG evolution step
            self.evolve(self.queue,
                        (self.mo.first_val_steps*self.mo.thermal_realizations, self.mo.second_val_steps),
                        (self.mo.thermal_realizations, 1),
                        self.m.data, self.m_ref.data, self.dW.data, self.first_vals, self.second_vals, t).wait()

            # Periodic Normalizations
            if (i%(self.mo.normalize_interval)==0):
                self.normalize_m(self.queue,
                                 (self.mo.first_val_steps*self.mo.thermal_realizations, self.mo.second_val_steps),
                                 (self.mo.thermal_realizations, 1),
                                 self.m.data, self.m_ref.data).wait()

    def get_phase_diagram(self):
        self.reduce_m(self.queue,
                      (self.mo.first_val_steps, self.mo.second_val_steps),
                      (1,1),
                      self.m.data, self.phase_diagram.data, self.mo.thermal_realizations).wait()

        return self.phase_diagram.get().reshape(self.mo.second_val_steps, self.mo.first_val_steps).transpose()

