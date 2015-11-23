# From this package
from macrospin_gpu.kernels import Macrospin_2DPhaseDiagram
from macrospin_gpu.simulations import Simulation2D

# Numerical libraries
import numpy as np

# Plotting
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mk = Macrospin_2DPhaseDiagram()
    mk.set_magnetic_properties(Ms=1200.0, damping=0.05, Hpma=0, initial_m=[1,0,0])
    mk.set_external_field([0,0,100])
    mk.set_evolution_properties()
    mk.set_geometry(100,50,3)
    mk.add_spin_torque([0.0,0.0,1.0], 0.2, 1.49)
    mk.add_spin_torque([-1.0,0.0,0.0], 0.4, 1.49)
    mk.add_thermal_noise(4.0, 16)
    mk.define_phase_diagram("current_density", np.linspace(0, 2e8,128),
                            "pulse_duration",  np.linspace(0.05e-9, 1.0e-9,128))

    sim = Simulation2D(mk)
    sim.run()
    phase_diagram = sim.get_phase_diagram()

    min_duration = 0.05
    max_duration = 1.00
    min_current  = 0
    max_current  = 1e8

    extent = (min_duration, max_duration, min_current, max_current)
    aspect = (max_duration-min_duration)/(max_current-min_current)

    plt1 = plt.figure(1)
    plt.imshow(phase_diagram, origin='lower', cmap=plt.get_cmap('Blues'),
                 extent=extent, aspect=aspect)
    plt.colorbar()
    plt1.suptitle('Switching Probability')
    plt1.gca().set_xlabel('Pulse Duration (ns)')
    plt1.gca().set_ylabel(r'Current Density (10$^8$A/cm$^2$)')
    plt.show()