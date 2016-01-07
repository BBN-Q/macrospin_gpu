# From this package
from macrospin_gpu.kernels import Macrospin_2DPhaseDiagram
from macrospin_gpu.simulations import Simulation2D

# Numerical libraries
import numpy as np

# Plotting
import matplotlib.pyplot as plt

if __name__ == '__main__':
    min_duration = 0.001
    max_duration = 0.5
    min_current  = 0    
    max_current  = 2e8

    mk = Macrospin_2DPhaseDiagram()
    mk.set_magnetic_properties(Ms=640.0, damping=0.1, Hpma=000, initial_m=[-1,0,0])
    mk.set_external_field([0,0,0])
    mk.set_evolution_properties(dt=1e-12)
    mk.set_geometry(100,50,3)
    mk.add_spin_torque([0.0,0.0,1.0], 0.16, 1.49)
    mk.add_spin_torque([-1.0,0.0,0.0], 0.00, 1.49)
    mk.add_thermal_noise(0.0, 1)
    mk.define_phase_diagram("current_density", np.linspace(min_current, max_current, 64),
                            "pulse_duration",  np.linspace(min_duration*1e-9, max_duration*1e-9, 64))
    mk.store_time_traces()

    sim = Simulation2D(mk)
    sim.run()
    phase_diagram = sim.get_phase_diagram()

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