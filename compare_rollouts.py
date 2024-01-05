"""
Plot a saved trace for viewing.

"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def main():
    spga_data = load_data("results/spga_trace.pkl")
    srl_data = load_data("results/srl_trace.pkl")
    
    spga_x = []
    spga_th = []
    for i, j in spga_data:
        spga_x.append(i)
        spga_th.append(j)
        
    srl_x = []
    srl_th = []
    for i, j in srl_data:
        srl_x.append(i)
        srl_th.append(j)
        
    plt.figure(1)
    plt.plot(srl_x, np.linspace(0, len(srl_x)-1, len(srl_x)), label='srl')
    plt.plot(spga_x, np.linspace(0, len(spga_x)-1, len(spga_x)), label='spga')
    
    plt.plot(np.ones((1000,))*1.5, np.linspace(0, 999, 1000), 'r-.', label='Unsafe')
    plt.plot(np.ones((1000,))*-1.5, np.linspace(0, 999,1000), 'r-.')
    plt.plot(np.ones((1000,))*0.75, np.linspace(0, 999, 1000), color='orange', linestyle='-.')
    plt.plot(np.ones((1000,))*-0.75, np.linspace(0, 999,1000), color='orange', linestyle='-.', label='Training Bound')
    # Horizontal green bar
    plt.hlines(1000, -0.75, 0.75, color=(0.2, 0.9, 0.2), linewidth=3, label='Goal')
    plt.axvspan(-2.4, -1.5, color='red', alpha=0.2);
    plt.axvspan(1.5, 2.4, color='red', alpha=0.2);
    plt.axvspan(-1.5, -0.75, color='orange', alpha=0.2);
    plt.axvspan(0.75, 1.5, color='orange', alpha=0.2);
    # plt.add_patch(Rectangle((-1.5, 1000), 3, 6,
    #                      facecolor = mcolors.cnames['lime'],
    #                      alpha=0.5,
    #                      fill=True, label="Goal"))
    plt.ylabel('Time')
    plt.xlabel('Position (x)')
    plt.text(
            -1.9, 50, "Time", ha="center", va="center", rotation=90, size=15,
            bbox=dict(boxstyle="rarrow, pad=0.25", fc="cyan", ec="b", lw=2))
    plt.xlim([-2.4, 2.4])
        
    plt.tight_layout()
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()