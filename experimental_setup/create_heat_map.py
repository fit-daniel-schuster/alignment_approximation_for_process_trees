from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc


def create_heat_map(data=None, title="", save=False, x_labels=None, y_labels=None):
    # fig, ax = plt.subplots(figsize=(6, 3))
    fig, ax = plt.subplots(figsize=(5, 3))
    if data is not None:
        if x_labels is None:
            x_labels = [str(i) for i in range(1, len(data) + 1)]
        if y_labels is None:
            y_labels = [str(i) for i in range(1, len(data) + 1)]
    else:
        x_labels = list(range(1, 11))
        y_labels = list(range(1, 11))
    if data is None:
        data = np.array([[0.8, 2.4, 2.5, 3.9, 0.0],
                         [2.4, 0.0, 4.0, 1.0, 2.7],
                         [1.1, 2.4, 0.8, 4.3, 1.9],
                         [0.6, 0.0, 0.3, 0.0, 3.1],
                         [0.1, 2.0, 0.0, 1.4, 0.0]])

    ax = sns.heatmap(data, annot=True, xticklabels=x_labels, yticklabels=y_labels, linecolor="w",
                     linewidths=.5, fmt=".2f")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(title)
    plt.ylabel("tree height (TH)")
    plt.xlabel("trace length (TL)")
    plt.tight_layout()
    plt.show()
    if save:
        plt.draw()
        fig.savefig("plot" + str(datetime.now()).replace(":", "-").replace(" ", "-").replace(".", "-") + ".pdf",
                    dpi=500, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format="pdf",
                    transparent=False, bbox_inches=None, pad_inches=0.1)


if __name__ == "__main__":
    create_heat_map()
