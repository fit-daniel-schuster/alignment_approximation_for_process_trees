# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def create_surface_plot(data):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x = range(1, 11)
    y = range(1, 11)
    x, y = np.meshgrid(x, y)

    z = np.array([[10, 2, 1, 1, 2, 3, 1, 2, 3], [2, 2, 2, 1, 2, 3, 1, 2, 3], [6, 2, 3, 1, 2, 3, 1, 2, 3],
                  [10, 2, 1, 1, 2, 3, 1, 2, 3], [2, 2, 2, 1, 2, 3, 1, 2, 3], [6, 2, 3, 1, 2, 3, 1, 2, 3],
                  [10, 2, 1, 1, 2, 3, 1, 2, 3], [2, 2, 2, 1, 2, 3, 1, 2, 3], [6, 2, 3, 1, 2, 3, 1, 2, 3]])
    z = data
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.jet,
                           linewidth=0, antialiased=True)
    ax.set_xticks(range(1, 11))
    ax.set_yticks(range(1, 11))
    ax.set_xlabel('process tree height')
    ax.set_ylabel('trace length')
    ax.set_zlabel('avg. computation time')

    surf = ax.plot_surface(x, y, z, cmap=cm.jet,
                           linewidth=1, antialiased=True)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf)

    plt.show()


if __name__ == "__main__":
    create_surface_plot()
