"""Plotting functions for DONUT geometry."""

import matplotlib.pyplot as plt
import config as cfg

def poloidal_cross_section(x, y, ctrl_x, ctrl_y):
    """Plot the poloidal cross section of the geometry.
    Args:        x: list of x coordinates for each section
        y: list of y coordinates for each section
        ctrl_x: list of x coordinates of control points for each section
        ctrl_y: list of y coordinates of control points for each section
    """
    print("Plotting poloidal cross section...")
    for i, x_section in enumerate(x):
        plt.plot(x_section, y[i], label=f"Section {i + 1}", color = cfg.color[i])
        plt.scatter(ctrl_x[i], ctrl_y[i], label=f"Control Points {i + 1}",
                    color = cfg.color[i], marker='x')
    plt.grid(linestyle='--', color='gray', linewidth=0.2)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.suptitle("Poloidal Cross Section")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("poloidal_cross_section.pdf")
