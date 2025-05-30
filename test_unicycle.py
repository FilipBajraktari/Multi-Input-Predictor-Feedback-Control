import matplotlib.pyplot as plt
import numpy as np

from unicycle import Unicycle
from utils import simulate_system

from . import *


def main():

    # Simulation
    X0 = np.ones(3, dtype=np.float32)*0.5
    init_inputs = [np.ones(NDi, dtype=np.float32) for NDi in NDs]
    ml_predictor_path = "models/const_distinct_delays_15_epochs.pth"
    unicycle = Unicycle(X0, init_inputs, dt, ml_predictor_path)

    import time
    start_time = time.time()
    states, controls, P, P_hat = simulate_system(unicycle, t)
    print(f"Elapsed time: {time.time() - start_time} seconds")

    # Plot Graphics
    fig = plt.figure(figsize=(20, 15))
    gs =  fig.add_gridspec(4, 6)

    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])

    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])

    ax6 = fig.add_subplot(gs[2, 0:2])
    ax7 = fig.add_subplot(gs[2, 2:4])
    ax8 = fig.add_subplot(gs[2, 4:6])
    ax9 = fig.add_subplot(gs[3, 0:2])
    ax10 = fig.add_subplot(gs[3, 2:4])
    ax11 = fig.add_subplot(gs[3, 4:6])

    # X coordinate
    ax1.plot(t, states[0])
    ax1.set_xlabel("t")
    ax1.set_ylabel("X(t)")
    ax1.grid(True)

    # Y coordinate
    ax2.plot(t, states[1])
    ax2.set_xlabel("t")
    ax2.set_ylabel("Y(t)")
    ax2.grid(True)

    # Theta
    ax3.plot(t, states[2])
    ax3.set_xlabel("t")
    ax3.set_ylabel(r"$\theta(t)$")
    ax3.grid(True)

    # Anuglar velocity - U1
    ax4.plot(t[:-1], controls[0])
    ax4.set_xlabel("t")
    ax4.set_ylabel(r"$\omega(t)$")
    ax4.grid(True)

    # Linear velocity - U2
    ax5.plot(t[:-1], controls[1])
    ax5.set_xlabel("t")
    ax5.set_ylabel(r"$\nu(t)$")
    ax5.grid(True)

    # P1_x
    ax6.plot(t, P_hat[0, 0])
    ax6.set_xlabel("t")
    ax6.set_ylabel("$P_{1X}$(t)")
    ax6.grid(True)

    # P1_y
    ax7.plot(t, P_hat[1, 0])
    ax7.set_xlabel("t")
    ax7.set_ylabel("$P_{1Y}$(t)")
    ax7.grid(True)

    # P1_theta
    ax8.plot(t, P_hat[2, 0])
    ax8.set_xlabel("t")
    ax8.set_ylabel(r"$P_{1\theta}(t)$")
    ax8.grid(True)

    # P2_x
    ax9.plot(t, P_hat[0, 1])
    ax9.set_xlabel("t")
    ax9.set_ylabel("$P_{2X}$(t)")
    ax9.grid(True)

    # P2_y
    ax10.plot(t, P_hat[1, 1])
    ax10.set_xlabel("t")
    ax10.set_ylabel("$P_{2Y}$(t)")
    ax10.grid(True)

    # P2_theta
    ax11.plot(t, P_hat[2, 1])
    ax11.set_xlabel("t")
    ax11.set_ylabel(r"$P_{2\theta}(t)$")
    ax11.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()