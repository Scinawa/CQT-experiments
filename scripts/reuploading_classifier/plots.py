import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

def plot_reuploading_classifier(result_json, output_path="build/", show=False):
    # Retrieve relevant data
    train_x = np.array(result_json['x_train'])
    train_y = np.array(result_json['train_predictions'])
    test_x = np.array(result_json['x_test'])
    test_y = np.array(result_json['test_predictions'])
    loss_history = result_json['loss_history']
    
    fig = plt.figure(figsize=(8, 6), dpi=120)
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])  # 2 rows, 2 columns
    
    # Train plot (top-left)
    ax_train = fig.add_subplot(gs[0, 0])
    for label in np.unique(train_y):
        data_label = np.transpose(train_x[np.where(train_y == label)])
        ax_train.scatter(data_label[0], data_label[1])
    ax_train.set_title("Train predictions")
    ax_train.set_xlabel(r"$x$")
    ax_train.set_ylabel(r"$y$")
    circle_train = plt.Circle((0, 0), np.sqrt(2 / np.pi), edgecolor='k', linestyle='--', fill=False)
    ax_train.add_patch(circle_train)
    
    # Test plot (top-right)
    ax_test = fig.add_subplot(gs[0, 1])
    for label in np.unique(test_y):
        data_label = np.transpose(test_x[np.where(test_y == label)])
        ax_test.scatter(data_label[0], data_label[1])
    ax_test.set_title("Test predictions")
    ax_test.set_xlabel(r"$x$")
    ax_test.set_ylabel(r"$y$")
    circle_test = plt.Circle((0, 0), np.sqrt(2 / np.pi), edgecolor='k', linestyle='--', fill=False)
    ax_test.add_patch(circle_test)
    
    # Loss plot (bottom row spanning both columns)
    ax_loss = fig.add_subplot(gs[1, :])
    ax_loss.plot(loss_history)
    ax_loss.set_title("Loss plot")
    ax_loss.set_xlabel(r"$Iteration$")
    ax_loss.set_ylabel(r"$Loss$")
    
    plt.tight_layout()
    if show:
        plt.show()
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, "reuploading_classifier_plots.pdf"), bbox_inches='tight', dpi=300)
    plt.close(fig)


# Individual plotting functions
def plot_predictions(x, y, title="scatter_plot", outdir=".", show=False):
    x = np.array(x)
    y = np.array(y)
    plt.figure(figsize=(4, 4 * 6 / 8), dpi=120)
    for label in np.unique(y):
        data_label = np.transpose(x[np.where(y==label)])
        plt.scatter(data_label[0], data_label[1])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$y$")
    circle = plt.Circle((0, 0), np.sqrt(2 / np.pi), edgecolor ='k', linestyle='--', fill=False)
    plt.gca().add_patch(circle)
    os.makedirs(outdir, exist_ok=True)
    if show:
        plt.show()
    plt.savefig(os.path.join(outdir, f"{title}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

def plot_loss_history(x, y, title="loss_plot", outdir=".", show=False):
    plt.figure(figsize=(4, 4 * 6 / 8), dpi=120)
    plt.plot(x, y)
    plt.xlabel(r"$Iteration$")
    plt.ylabel(r"$Loss$")
    os.makedirs(outdir, exist_ok=True)
    if show:
        plt.show()
    plt.savefig(os.path.join(outdir, f"{title}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()