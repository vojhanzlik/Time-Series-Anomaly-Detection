import matplotlib.pyplot as plt
import numpy as np

METHODS_LABEL_DICT = {0: "Signals as a distance matrix",
                      1: "DTW-aligned SAD distance (L1 norm)",
                      2: "DTW-aligned euclidean distance (L2 norm)",
                      3: "Barycenter as a distance matrix",
                      4: "DTW-aligned Cosine distance",
                      5: "Unaligned euclidean distance (L2 norm)",
                      6: "Unaligned SAD distance (L1 norm)",
                      None: ""}


def plot_nd_signal(n_dim, signal1, signal2):
    fig = plt.figure()
    axes = [fig.add_subplot(n_dim, 2, i) for i in range(1, 2 * n_dim + 1)]
    n = 0
    for i in range(0, n_dim):
        axes[i + n].plot(signal1[:, 0], signal1[:, i + 1], color="red", linewidth=1)
        axes[i + n + 1].plot(signal2[:, 0], signal2[:, i + 1], color="red", linewidth=1)
        n += 1
    plt.show()


def plot_one_6dim_signal(signal1, save_fig=None):
    fig, ax = plt.subplots(6, 1, figsize=(6.5, 5))
    ylabels = ["Force x",
               "Force y",
               "Force z",
               "Torque x",
               "Torque y",
               "Torque z"
               ]
    xaxis = np.arange(np.shape(signal1)[0])
    for i in range(0, 6):
        ax[i].plot(xaxis, signal1[:, i], color="red", linewidth=1)
        ax[i].set_ylabel(ylabels[i])
        if i < 5:
            ax[i].set_xticklabels([])
    ax[5].set_xlabel("Time")
    # fig.suptitle("Sample signal of a process", fontweight="bold", fontsize = 12)
    fig.align_ylabels()
    if save_fig is not None:
        name = str(save_fig) if len(save_fig) > 4 and str(save_fig)[-4:] == ".pdf" else str(save_fig) + ".pdf"
        plt.savefig(name, format='pdf', bbox_inches='tight')
    plt.show()


def plot_samples(correct_points, wrong_points, barycenters, ebarycenters=[[], []], method=[2, 5], title=""):
    scatter_if_not_empty(correct_points, color="blue", marker="+")
    scatter_if_not_empty(wrong_points, color="red", marker="x")
    scatter_if_not_empty(barycenters, color="green", marker="o")
    scatter_if_not_empty(ebarycenters, color="orange", marker="o")
    plt.grid(True)
    plt.xlabel(METHODS_LABEL_DICT[method[0]])
    plt.ylabel(METHODS_LABEL_DICT[method[1]])
    plt.title(title)
    plt.legend(["Successfull process",
                "Failed process",
                "Barycenter distance",
                "Euclidean barycenter distance"], loc="lower right")
    plt.show()


def scatter_if_not_empty(points, color, marker):
    if points:
        plt.scatter(*points, color=color, marker=marker)


if __name__ == "__main__":
    import pickle

    with open("./data_to_load/new_data1.pkl", 'rb') as file:
        loaded_data = pickle.load(file)
    new_data = []
    for i in loaded_data:
        time = np.arange(np.shape(i)[0])[:, np.newaxis]
        cursig = i[:, 3:]
        new_data.append(cursig)
    plot_one_6dim_signal(new_data[0])
