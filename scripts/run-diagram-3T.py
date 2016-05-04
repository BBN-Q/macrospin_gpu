# From this package
from macrospin_gpu.kernels import Macrospin_2DPhaseDiagram
from macrospin_gpu.simulations import Simulation2D

# Numerical libraries
import numpy as np

# Plotting
import matplotlib.pyplot as plt

if __name__ == '__main__':
    min_duration = 0.001
    max_duration = 3.0
    min_current  = 0    
    max_current  = 4.5e8

    mk = Macrospin_2DPhaseDiagram()
    mk.set_magnetic_properties(Ms=1200.0, damping=0.015, Hpma=10000, initial_m=[1,0,0])
    mk.set_external_field([2.0,5.0,3.0])
    mk.set_evolution_properties(dt=5e-13, total_time=max_duration*1.2e-9, normalize_interval=50)
    mk.set_geometry(170,75,1.3)
    # mk.add_spin_torque([0.0,0.0,1.0], 0.2, 1.49)
    mk.add_spin_torque([1.0,0.0,0.0], 0.1, 1.0)
    # mk.enable_oersted_field(field_direction=[-1,0,0])
    mk.add_thermal_noise(4, 32)
    mk.define_phase_diagram("current_density", np.linspace(min_current, max_current, 128),
                            "pulse_duration",  np.linspace(min_duration*1e-9, max_duration*1e-9, 128))
    # mk.store_time_traces(interval=2e-12)

    sim = Simulation2D(mk)
    sim.run()
    phase_diagram = sim.get_phase_diagram()

    extent = (min_duration, max_duration, min_current, max_current)
    aspect = (max_duration-min_duration)/(max_current-min_current)

    plt1 = plt.figure(1)
    plt.imshow(phase_diagram, origin='lower', cmap=plt.get_cmap('RdGy'),
                 extent=extent, aspect=aspect)
    plt.colorbar()
    plt1.suptitle('Switching Probability')
    plt1.gca().set_xlabel('Pulse Duration (ns)')
    plt1.gca().set_ylabel(r'Current Density (10$^8$A/cm$^2$)')

    # plt2 = plt.figure(2)
    # time_data = sim.get_time_traces()
    # ni, nj, n_blah = time_data.shape
    # for i in range(0,ni,16):
    #     for j in range(0,nj,16):
    #         plt.plot(time_data[i,j,:]['x'])
    plt.show()

