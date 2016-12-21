# From this package
from macrospin_gpu.kernels import Macrospin_2DPhaseDiagram
from macrospin_gpu.simulations import Simulation2D

# Numerical libraries
import numpy as np

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    min_duration = 0.0
    max_duration = 1.0
    min_current  = 0e8    
    max_current  = 1.0e8

    mk = Macrospin_2DPhaseDiagram()
    mk.set_magnetic_properties(Ms=1500.0, damping=0.03, Hpma=10000, Hd=1500*4*np.pi, Hk=50.0, initial_phi=np.pi)
    mk.set_external_field([2.0,5.0,3.0])
    mk.set_evolution_properties(dt=1e-12, total_time=max_duration*2.0e-9)
    mk.set_geometry(170,75,1.3)
    mk.add_spin_torque([0.0,0.0,1.0], 0.15, 1.0)
    mk.add_spin_torque([1.0,0.0,0.0], 0.08, 1.0)
    # mk.enable_oersted_field(field_direction=[-1,0,0])
    mk.add_thermal_noise(16, 16)
    # mk.store_time_traces(interval=1.0e-12)
    mk.define_phase_diagram("current_density", np.linspace(min_current, max_current, 64),
                            "pulse_duration",  np.linspace(min_duration*1e-9, max_duration*1e-9, 64))
    # mk.store_time_traces(interval=2e-12)

    sim = Simulation2D(mk)
    sim.run()
    phase_diagram = sim.get_phase_diagram()

    extent = (min_duration, max_duration, min_current, max_current)
    aspect = (max_duration-min_duration)/(max_current-min_current)

    np.savetxt("PhaseDiagram.txt", phase_diagram.T, header="{}, {}, {}, {}".format(*extent))

    plt1 = plt.figure(1)
    plt.imshow(1.0-phase_diagram, origin='lower', cmap=plt.cm.Blues, #cmap=plt.get_cmap('RdGy'),
                 extent=extent, aspect=aspect, interpolation='nearest')
    plt.colorbar()
    plt1.suptitle('Switching Probability')
    plt1.gca().set_xlabel('Pulse Duration (ns)')
    plt1.gca().set_ylabel(r'Current Density (10$^8$A/cm$^2$)')

    # plt2 = plt.figure(2)
    # ax2 = Axes3D(plt2)
    # time_data = sim.get_time_traces()
    # ni, nj, n_blah = time_data.shape
    # for i in range(0,ni,int(ni/8)):
    #     for j in range(0,nj,int(nj/8)):
    #         ax2.plot(time_data[i,j,:]['x'], time_data[i,j,:]['y'], time_data[i,j,:]['z'])
    # ax2.set_xlim(-1.1,1.1)
    # ax2.set_ylim(-1.1,1.1)
    # ax2.set_zlim(-1.1,1.1)
    plt.show()

