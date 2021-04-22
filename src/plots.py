import os
from typing import List

import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn import metrics

import utils


def _directory(plot, metric, method) -> str:
    dir_path = os.path.join(utils.PLOTS_DIR, plot, method, metric)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def histogram(
        x: np.array,
        dataset: str,
        metric: str,
        method: str,
):
    """ Plot a histogram of the given outlier scores.

    :param x: 1-d array of anomaly scores to plot in a histogram.
    :param dataset: The dataset which was scored.
    :param metric: The metric used with the dataset.
    :param method: The method used for generating the scores.
    """
    fig = plt.figure(figsize=(16, 10), dpi=200)
    fig.add_subplots(111)
    plt.hist(x=x, color='#0504aa', rwidth=0.85)
    plt.xlabel('Outlier Score')
    plt.ylabel('Counts')
    plot_path = os.path.join(_directory('histograms', metric, method), f'{dataset}.png')
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    return


def roc_curve(
        y_true: np.array,
        y_pred: np.array,
        dataset: str,
        metric: str,
        method: str,
):
    """ Plot the Roc-Curve, given the labels and predictions.

    :param y_true: True Labels, 1-d array.
    :param y_pred: Predicted scores, 1-d array.
    :param dataset: The dataset which was scored.
    :param metric: The metric used with the dataset.
    :param method: The method used for generating the scores.
    """
    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)

    fig = plt.figure(figsize=(16, 10), dpi=200)
    fig.add_subplots(111)
    plt.plot(fpr, tpr, color='darkorange', lw=2., label=f'area: {auc:.6f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2., linestyle='--')
    plt.xlim([0, 1.05]), plt.ylim([0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{dataset}-{metric}-{method}')
    plt.legend(loc='lower right')
    plot_path = os.path.join(_directory('roc_curves', metric, method), f'{dataset}.png')
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    return


def scatter_2d(
        data: np.array,
        labels: List[int],
        plot_path: str,
):
    """ Make a 2d scatter plot of the data

    :param data: Numpy array with shape (2, n) with rows representing the x and y axes respectively.
    :param labels: A list of labels for each point.
    :param plot_path: The path where to save the plot.
    """
    if data.shape[0] != 2:
        raise ValueError(f'expected an array of shape (2, n). Got {data.shape} instead.')

    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplots(111)
    plt.scatter(data[0, :], data[1, :], c=labels, cmap='Dark2', s=5.)
    ax.set_xlim([np.min(data[0, :]), np.max(data[0, :])])
    ax.set_ylim([np.min(data[1, :]), np.max(data[1, :])])
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    return


def scatter_3d(
        data: np.array,
        labels: List[str],
        plot_path: str,
):
    """ Make a 3d scatter plot of the data

    :param data: Numpy array with shape (3, n) with rows representing the x and y axes respectively.
    :param labels: A list of labels for each point.
    :param plot_path: The path where to save the plot.
    """
    if data.shape[0] != 3:
        raise ValueError(f'expected an array of shape (2, n). Got {data.shape} instead.')

    fig = plt.figure(figsize=(16, 10), dpi=200)
    ax = fig.add_subplots(111)
    plt.scatter(data[0, :], data[1, :], data[2, :], c=labels, cmap='Dark2', s=5.)
    ax.set_xlim([np.min(data[0, :]), np.max(data[0, :])])
    ax.set_ylim([np.min(data[1, :]), np.max(data[1, :])])
    ax.set_zlim([np.min(data[2, :]), np.max(data[2, :])])
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.25)
    plt.close(fig)
    return


def embed_umap(
        data: np.array,
        n_components: int,
        n_neighbors: int,
        dataset: str,
        metric: str,
):
    """ Create and save a UMAP embedding of the data.

    :param data: raw data to be embedded.
    :param n_components: will create an n-dimensional embedding.
    :param n_neighbors: number of neighbors for umap reduction. Low numbers emphasize local structure.
    :param dataset: name of the dataset being reduced.
    :param metric: metric to use for the reduction.
    """
    umap_path = os.path.join(utils.UMAPS_DIR, f'{dataset}_{metric}.memmap')
    if not os.path.exists(umap_path):
        embedding: np.ndarray = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            metric=metric,
        ).fit_transform(data)

        os.makedirs(utils.UMAPS_DIR, exist_ok=True)
        saver: np.memmap = np.memmap(
            filename=umap_path,
            dtype=float,
            mode='w+',
            shape=(data.shape[0], n_components),
        )
        saver[:] = embedding[:]
        del saver

    return np.memmap(
        filename=umap_path,
        dtype=float,
        mode='r',
        shape=(data.shape[0], n_components),
    )
