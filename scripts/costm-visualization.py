#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import absolute_import, print_function
import numpy as np

# PyQt4 imports
from   PyQt4 import QtGui, QtCore, QtOpenGL
from   PyQt4.QtOpenGL import QGLWidget

# PyOpenGL imports
import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import OpenGL.arrays.vbo as vbo

# PyOpenCL imports
import pyopencl as cl
import pyopencl.clrandom as ran
from   pyopencl.tools import get_gl_sharing_context_properties
from   jinja2 import Template

# Physics imports
from   macrospin_gpu.demag import demagCylinder

# Other imports
from   tqdm import *
import time

def clinit():
    """Initialize OpenCL with GL-CL interop.
    """
    plats = cl.get_platforms()
    # handling OSX
    if sys.platform == "darwin":
        ctx = cl.Context(properties=get_gl_sharing_context_properties(),
                             devices=[])
    else:
        ctx = cl.Context(properties=[
                            (cl.context_properties.PLATFORM, plats[0])]
                            + get_gl_sharing_context_properties())
    queue = cl.CommandQueue(ctx)
    return ctx, queue

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
realdt      = 1.0e-13           # Actual time step (s)
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
min_current        = 0.0e8 # A/cm^2
max_current        = 1.2e8
min_duration       = 0.6e-9
max_duration       = 0.6e-9
pause_before       = 0.3e-9
pause_after        = 2.0e-9
total_time         = max_duration + pause_before + pause_after
total_steps        = int(total_time/realdt)
normalize_interval = 50
display_interval   = 5

# External Field
hExtX     = 0.0
hExtY     = 0.0
hExtZ     = 0.0

# Perpendicular Anisotropy
hPMA      = 0000.0
Nzz       = Nzz - hPMA/Ms # from compensation


class GLPlotWidget(QGLWidget):
    # default window size
    width, height = 1280, 1024

    def initialize_buffers(self):
        """Initialize OpenGL and OpenCL buffers and interop objects,
        and compile the OpenCL kernel.
        """
        
        # Load kernel
        with open("src/macrospin_gpu/templates/costm-amp-dur-gl.cl") as template_file:
            kernel_template = Template(template_file.read())

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
        self.ctx, self.queue = clinit()
        self.prg = cl.Program(self.ctx, kernel).build()
        self.evolve = self.prg.evolve
        self.evolve.set_scalar_arg_dtypes([None, None, None, None, np.float32])
        self.reduce_m = self.prg.reduce_m
        self.reduce_m.set_scalar_arg_dtypes([None, None, np.int32])
        self.normalize_m = self.prg.normalize_m
        self.update_m_of_t = self.prg.update_m_of_t
        self.update_m_of_t.set_scalar_arg_dtypes([None, None, None, np.int32, np.int32, np.int32])

        # release the PyOpenCL queue
        self.queue.finish()

        # Define random number generator
        self.rg = ran.RanluxGenerator(self.queue)

        # Data dimensions
        self.realizations   = 8 # Averages over different realizations of the noise process
        self.current_steps  = 256
        self.duration_steps = 4
        self.N              = self.current_steps*self.duration_steps*self.realizations
        self.time_points    = 64 # How many points to store as a function of time

        # Current state
        self.current_iter      = 0
        self.current_time      = 0.0
        self.current_timepoint = 0

        # Declare the GPU bound arrays
        self.m             = cl.array.zeros(self.queue, self.N, cl.array.vec.float4)
        self.dW            = cl.array.zeros(self.queue, self.N, cl.array.vec.float4)
        self.phase_diagram = cl.array.zeros(self.queue, self.current_steps*self.duration_steps, np.float32)

        # Create the GPU buffers that contain the phase diagram parameters
        self.durations_np  = np.linspace(min_duration, max_duration, self.duration_steps).astype(np.float32)
        self.currents_np   = np.linspace(min_current, max_current, self.current_steps).astype(np.float32)
        self.m_of_t_np     = np.ndarray((self.current_steps*self.duration_steps*self.time_points, 4), dtype=np.float32)
        self.colors_np     = np.ndarray((self.current_steps*self.duration_steps*self.time_points, 4), dtype=np.float32)
        self.durations     = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.durations_np)
        self.currents      = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.currents_np)

        # Load initial magnetization state
        initial_m = np.zeros(self.N, dtype=cl.array.vec.float4)
        initial_m[:] = (1,0,0,0)
        cl.enqueue_copy(self.queue, self.m.data, initial_m)

        self.colors_np[:,:] = [1.,1.,1.,1.] # White particles
        self.colbuf = vbo.VBO(data=self.colors_np, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.colbuf.bind()

        # For the glMultiDraw command we need an array of offsets and draw lengths
        self.start_indices = np.arange( 0, self.current_steps*self.duration_steps*self.time_points, self.time_points, dtype=np.int32 )
        self.draw_lengths  = self.time_points * np.ones(self.current_steps*self.duration_steps, dtype=np.int32)

        # Declare an empty OpenGL VBO and bind it
        self.glbuf = vbo.VBO(data=self.m_of_t_np, usage=gl.GL_DYNAMIC_DRAW, target=gl.GL_ARRAY_BUFFER)
        self.glbuf.bind()

        # create an interop object to access to GL VBO from OpenCL
        self.glclbuf = cl.GLBuffer(self.ctx, cl.mem_flags.READ_WRITE, int(self.glbuf.buffers[0]))
        self.colclbuf = cl.GLBuffer(self.ctx, cl.mem_flags.READ_WRITE, int(self.colbuf.buffers[0]))

    def execute(self):
        """Execute the OpenCL kernel.
        """
        # Beginning of execute loop
        start_time = time.time()

        # get secure access to GL-CL interop objects
        cl.enqueue_acquire_gl_objects(self.queue, [self.glclbuf, self.colclbuf])
        
        # Execute the kernel for the duration of the timer cycle
        while time.time()-start_time < 40.0e-3:
            if self.current_iter == 0:
                for i in range(self.time_points):
                        self.update_m_of_t(self.queue,(self.current_steps, self.duration_steps),
                                              (1,1),
                                              self.m.data, self.glclbuf, self.colclbuf,
                                              self.time_points, self.realizations,
                                              i%self.time_points).wait()

            self.current_time += realdt
            self.current_iter += 1
            # Generate random numbers for the stochastic process
            if temperature > 0:
                self.rg.fill_normal(self.dW)

            # Run a single LLG evolution step
            self.evolve(self.queue, (self.current_steps*self.realizations, self.duration_steps),
                                    (self.realizations, 1), 
                                    self.m.data, self.dW.data, self.currents,
                                    self.durations, self.current_time).wait()

            # Periodic Normalizations
            if (self.current_iter%(normalize_interval)==0):
                self.normalize_m(self.queue, (self.current_steps*self.realizations, self.duration_steps), 
                                             (self.realizations, 1),
                                             self.m.data).wait()

            # Push data to the gl buffer
            if (self.current_iter%(display_interval)==0):
                self.update_m_of_t(self.queue,(self.current_steps, self.duration_steps),
                                              (1,1),
                                              self.m.data, self.glclbuf, self.colclbuf,
                                              self.time_points, self.realizations,
                                              self.current_timepoint%self.time_points).wait()
                self.current_timepoint += 1


        # release access to the GL-CL interop objects
        cl.enqueue_release_gl_objects(self.queue, [self.glclbuf, self.colclbuf])
        self.queue.finish()
        gl.glFlush()

    def update_buffer(self):
        """Update the GL buffer from the CL buffer
        """
        # execute the kernel before rendering
        self.execute()
        gl.glFlush()

    def initializeGL(self):
        """Initialize OpenGL, VBOs, upload data on the GPU, etc.
        """
        # Rotation amount
        self.delta_x  = 0.0
        self.delta_y  = 0.0
        self.distance = -7.0
        # initialize OpenCL first
        self.initialize_buffers()
        # set background color
        gl.glClearColor(0,0,0,0)
        # update the GL buffer from the CL buffer
        self.update_buffer()

    def paintGL(self):
        """Paint the scene.
        """
        self.execute()
        # clear the GL scene
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # Move back
        gl.glTranslatef(0.0, 0.0, self.distance)
        gl.glRotatef(-90.0, 1, 0, 0)
        gl.glRotatef(self.delta_y, 1, 0, 0) 
        gl.glRotatef(self.delta_x, 0, 0, 1)

        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glLineWidth(1.0)

        
        self.colbuf.bind()
        gl.glColorPointer(4, gl.GL_FLOAT, 0, self.colbuf)
        self.glbuf.bind()
        gl.glVertexPointer(4, gl.GL_FLOAT, 0, self.glbuf)
        gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
        gl.glEnableClientState(gl.GL_COLOR_ARRAY)

        gl.glMultiDrawArrays(gl.GL_LINE_STRIP, 
                             self.start_indices, self.draw_lengths,
                             self.current_steps*self.duration_steps)

        gl.glDisableClientState(gl.GL_COLOR_ARRAY)
        gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        gl.glDisable(gl.GL_BLEND)

    def resizeGL(self, width, height):
        """Called upon window resizing: reinitialize the viewport.
        """
        # update the window size
        self.width, self.height = width, height
        # paint within the whole window
        gl.glViewport(0, 0, width, height)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glFrustum(-1.0, +1.0, -1.0, 1.0, 5.0, 60.0)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Space:
           self.current_iter = 0
           self.current_time = 0.0

    def mousePressEvent(self, event):
        self.last_pos = event.posF()

    def mouseMoveEvent(self, event):
        dxy = event.posF() - self.last_pos
        dx = dxy.x()
        dy = dxy.y()

        if event.buttons() & QtCore.Qt.LeftButton:
            if event.modifiers() & QtCore.Qt.ControlModifier:
                pass
            else:
                self.delta_x += 0.2*dx
                self.delta_y += 0.2*dy

        self.last_pos = event.posF()

    def wheelEvent(self, event):
        self.distance += 0.002*event.delta()

if __name__ == '__main__':
    import sys
    import numpy as np

    # define a Qt window with an OpenGL widget inside it
    class TestWindow(QtGui.QMainWindow):
        def __init__(self):
            super(TestWindow, self).__init__()

            # initialize the GL widget
            self.widget = GLPlotWidget()

            # put the window at the screen position (100, 100)
            self.setGeometry(100, 100, self.widget.width, self.widget.height)
            self.setCentralWidget(self.widget)

            # Set up timer
            timer = QtCore.QTimer(self)
            timer.timeout.connect(self.widget.updateGL)
            timer.start(50)
            self.show()

        def keyPressEvent(self, event):
            self.widget.keyPressEvent(event)

    # create the Qt App and window
    app = QtGui.QApplication(sys.argv)
    window = TestWindow()
    window.show()
    app.exec_()