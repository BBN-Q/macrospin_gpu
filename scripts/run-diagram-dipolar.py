# From this package
from macrospin_gpu.kernels import Macrospin_2DPhaseDiagram_Dipolar
from macrospin_gpu.simulations import Simulation2D_Dipolar

# Numerical libraries
import numpy as np

# Plotting
import matplotlib.pyplot as plt

if __name__ == '__main__':
    min_duration = 0.05
    max_duration = 1.4
    min_current  = 0    
    max_current  = 4e8

    mk = Macrospin_2DPhaseDiagram_Dipolar()
    mk.set_magnetic_properties(Ms=1280.0, damping=0.1, Hpma=0, initial_m=[1,0,0])
    mk.set_external_field([384,0,0])
    mk.set_evolution_properties(dt=1e-12)
    mk.set_geometry(100,50,3)
    mk.add_reference_layer(initial_m=[1,0,0])
    mk.add_spin_torque([0.0,0.0,-1.0], 0.2, 1.49, square_pulse=False)
    mk.add_spin_torque([-1.0,0.0,0.0], 0.05, 1.49, square_pulse=False)
    mk.add_thermal_noise(100.0, 1)
    mk.define_phase_diagram("current_density", np.linspace(min_current, max_current, 64),
                            "pulse_duration",  np.linspace(min_duration*1e-9, max_duration*1e-9, 64))

    sim = Simulation2D_Dipolar(mk)
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