import logging
import os
import random
from glob import glob

import click
import numpy as np
from pyclam import criterion
from pyclam.manifold import Manifold
from sklearn.metrics import roc_auc_score

from .datasets import DATASETS, get, read
from .methods import METHODS
from .plot import RESULT_PLOTS

np.random.seed(42)
random.seed(42)

SUB_SAMPLE = 10_000
NORMALIZE = False

METRICS = {
    'cosine': 'cosine',
    'euclidean': 'euclidean',
    'manhattan': 'cityblock',
}

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build'))


def _manifold_path(dataset, metric, min_points, graph_ratio) -> str:
    """ Generate proper path to manifold. """
    return os.path.join(
        BUILD_DIR,
        ':'.join(map(str, [dataset, metric, min_points, f'{graph_ratio}.pickle']))
    )


def _dataset_from_path(path):
    return os.path.basename(path).split(':')[0]


def _metric_from_path(path):
    return os.path.basename(path).split(':')[1]


def _meta_from_path(path):
    bundle = os.path.basename(path).split(':')
    # noinspection PyTypeChecker
    return {
        'dataset': bundle[0],
        'metric': bundle[1],
        'min_points': bundle[2],
        'graph_ratio': bundle[3].split('.pickle')[0],
    }

class State:
    """ This gets passed between chained commands. """
    def __init__(self, dataset=None, metric=None, manifold=None):
        self.dataset = dataset
        self.metric = metric
        self.manifold = manifold

    def __getattr__(self, name):
        return None


@click.group(chain=True)
@click.option('--dataset', type=click.Choice(DATASETS.keys()))
@click.option('--metric', type=click.Choice(METRICS.keys()))
@click.pass_context
def cli(ctx, dataset, metric):
    ctx.obj = State(dataset, metric)


@cli.command()
@click.option('--max-depth', type=int, default=100)
@click.option('--min-points', type=int, default=3)
@click.option('--graph-ratio', type=int, default=100)
@click.pass_obj
def build(state, max_depth, min_points, graph_ratio):
    # Pickup any values from state that were pre-populated
    metrics = [state.metric] if state.metric else METRICS.keys()
    for dataset in [state.dataset] if state.dataset else DATASETS.keys():
        get(dataset)
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)

        for metric in metrics:
            logging.info('; '.join([
                f'dataset: {dataset}',
                f'metric: {metric}',
                f'shape: {data.shape}',
                f'outliers: {labels.sum()}'
            ]))
            manifold = Manifold(data, METRICS[metric])

            min_points = max(3, min_points) if data.shape[0] > 10_000 else 3
            filepath = _manifold_path(dataset, metric, min_points, graph_ratio)
            if os.path.exists(filepath):
                with open(filepath, 'rb') as fp:
                    logging.info(f'loading manifold {filepath}')
                    manifold = manifold.load(fp, data)
                manifold.build_tree(
                    criterion.MaxDepth(max_depth),
                    criterion.MinPoints(min_points),
                )
                manifold.build_graphs()
                with open(filepath, 'wb') as fp:
                    logging.info(f'dumping manifold {filepath}')
                    manifold.dump(fp)
                    state.manifold = manifold
            else:
                manifold.build(
                    criterion.MaxDepth(max_depth),
                    criterion.MinPoints(min_points),
                )
                with open(filepath, 'wb') as fp:
                    logging.info(f'dumping manifold {filepath}')
                    manifold.dump(fp)
                    state.manifold = manifold


@cli.command()
@click.option('--method', type=click.Choice(METHODS.keys()))
@click.option('--min-points', type=str, default='*')
@click.option('--graph-ratio', type=str, default='*')
@click.pass_obj
def test(state, method, min_points, graph_ratio):
    manifolds = [state.manifold] if state.manifold else glob(_manifold_path(state.datset or '*', state.metric or '*', min_points, graph_ratio))
    methods = [method] if method else METHODS.keys()
    for manifold in manifolds:
        if state.dataset:
            data, labels = read(state.dataset)
        else:
            data, labels = read(str(_dataset_from_path(manifold)), normalize=NORMALIZE, subsample=SUB_SAMPLE)

        # Load the manifold.
        if type(manifold) is str:
            with open(manifold, 'rb') as fp:
                manifold = Manifold.load(fp, data)

        for method in methods:
            for depth in range(0, manifold.depth + 1, 1):
                # These methods are invariant to depth.
                if method in {'n_points_in_ball', 'k_nearest'} and depth < manifold.depth:
                    continue
                logging.debug(f'calling {method} with {manifold}')
                anomalies = METHODS[method](manifold.graphs[depth])
                try:
                    logging.info('; '.join([
                        f'depth: {depth:>3}',
                        f'subgraphs: {len(manifold.graphs[depth].subgraphs):>5}',
                        f'clusters: {len(manifold.graphs[depth].clusters.keys()):>5}',
                        f'AUC Score: {roc_auc_score(labels, list(anomalies.values())):03.2f}',
                        f'{method}'
                    ]))
                except Exception as e:
                    logging.exception(e)


@cli.command()
@click.option('--plot', type=click.Choice(RESULT_PLOTS.keys()))
@click.option('--method', type=click.Choice(METHODS.keys()))
@click.option('--starting-depth', type=int, default=0)
@click.option('--min-points', type=str, default='*')
@click.option('--graph-ratio', type=str, default='*')
@click.pass_obj
def plot_results(state, plot, method, starting_depth, min_points, graph_ratio):
    manifolds = [state.manifold] if state.manifold else glob(_manifold_path(state.datset or '*', state.metric or '*', min_points, graph_ratio))
    methods = [method] if method else METHODS.keys()
    plots = [plot] if plot else RESULT_PLOTS.keys()

    for manifold in manifolds:
        if not state.dataset:
            meta = _meta_from_path(manifold)
            dataset, metric = str(meta['dataset']), meta['metric']
        else:
            dataset, metric = state.dataset, state.metric
        data, labels = read(dataset, normalize=NORMALIZE, subsample=SUB_SAMPLE)
        
        if type(manifold) is str:
            with open(manifold, 'rb') as fp:
                logging.info(f'loading manifold {manifold}')
                manifold = Manifold.load(fp, data)
        else:
            metric = manifold.metric
        
        for method in methods:
            for depth in range(starting_depth, manifold.depth + 1):
                if method in {'n_points_in_ball', 'k_nearest'} and depth < manifold.depth:
                    continue
                for plot in plots:
                    logging.info(f'{dataset}, {metric}, {depth}/{manifold.depth}, {method}, {plot}')
                    RESULT_PLOTS[plot](
                        labels,
                        METHODS[method](manifold.graphs[depth]),
                        dataset,
                        metric,
                        method,
                        depth,
                        save=True
                    )


@cli.command()
@click.pass_obj
def svm(state):
    from sklearn.svm import OneClassSVM
    from sklearn.model_selection import train_test_split
    from joblib import dump, load
    import numpy as np

    datasets = [state.dataset] if state.dataset else DATASETS.keys()
    metrics = [state.metric] if state.metric else METRICS.keys()
    
    # Build.
    for dataset in datasets:
        get(dataset)
        data, labels = read(dataset, normalize=False)
        normal, anomalies = np.argwhere(labels == 0).flatten(), np.argwhere(labels == 1).flatten()
        indices = np.concatenate([
            np.random.choice(normal, size=len(anomalies) * 9), 
            anomalies
            ])
        train, test = train_test_split(indices, stratify=labels[indices])
        for metric in metrics:
            filepath = os.path.join(BUILD_DIR, 'svm', ':'.join([dataset, metric]))
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            if os.path.exists(filepath):
                logging.info(f'loading {filepath}')
                model = load(filepath)
            else:
                logging.info(f'building {filepath}')
                model = OneClassSVM()
                model.fit(data[train], y=labels[train])
                dump(model, filepath)

            # Test. 
            logging.info(f'scoring {filepath}')
            predicted = model.predict(data[test])
            score = roc_auc_score(labels[test], predicted)
            logging.info(':'.join(map(str, [dataset, metric, round(score, 2)])))
            RESULT_PLOTS['roc_curve'](labels[test], {i: s for i, s in enumerate(np.clip(predicted, a_min=0, a_max=1))}, dataset, metric, 'SVM', 0, True)
            

@cli.command()
def plot_data():
    # TODO
    raise NotImplementedError
    for dataset in datasets:
        normalize = dataset not in ['mnist']
        data, labels = read(dataset, normalize)
        min_points = 5 if data.shape[0] > 50_000 else 1
        for metric in metrics:
            for n_neighbors in [32]:
                for n_components in [3]:
                    filename = os.path.join(BUILD_DIR, dataset)
                    filename = f'../data/{dataset}/umap/{n_neighbors}-{n_components}d-{metric}.pickle'
                    if data.shape[1] > n_components:
                        embedding = data
                        # embedding = make_umap(data, n_neighbors, n_components, metric, filename)
                    else:
                        embedding = data
                    title = f'{dataset}-{metric}-{n_neighbors}'
                    if n_components == 3:
                        # folder = f'../data/{dataset}/frames/{metric}-'
                        # plot_3d(embedding, labels, title, folder)

                        pass
                    if n_components == 2:
                        # plot_2d(embedding, labels, title)
                        pass


@cli.command()
@click.option('--dataset', type=str, default='*')
@click.option('--metric', type=str, default='*')
def animate(dataset, metric):
    # TODO
    raise NotImplementedError
    run([
        'ffmpeg',
        '-framerate', '30',
        '-i', os.path.join(dataset, 'frames', f'{metric}-%03d.png'),
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-crf', '20',
        '-pix_fmt', 'yuv420p',
        os.path.join(dataset, f'{metric}-30fps.mp4'),
    ])


if __name__ == "__main__":
    os.makedirs(BUILD_DIR, exist_ok=True)
    cli(prog_name='src')
