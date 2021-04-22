import random
from typing import Dict
from typing import List

import numpy as np
from pyclam import CHAODA
from pyclam import Graph
from pyclam import Manifold
from pyclam.chaoda import ClusterScores
from pyclam.criterion import MetaMLSelect
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from datasets import read
from utils import *

SAMPLING_DATASETS = [
    'annthyroid',
    'cardio',
    'cover',
    'http',
    'mammography',
    'mnist',
    'musk',
    'optdigits',
    'pendigits',
    'satellite',
    'satimage-2',
    'shuttle',
    'smtp',
    'thyroid',
    'vowels',
]
NOT_NORMALIZED = {
    'shuttle',
}
FEATURE_NAMES = [
    'cardinality',
    'radius',
    'lfd',
]
FEATURE_NAMES.extend([f'{_name}_ema' for _name in FEATURE_NAMES])


def extract_dt(tree: DecisionTreeRegressor, metric: str, method: str) -> str:
    """ Just a bit of meta-programming. Don't question the black magic wielded here. """
    # noinspection PyProtectedMember
    from sklearn.tree import _tree

    feature_name = [
        FEATURE_NAMES[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.tree_.feature
    ]
    tree_code: List[str] = [
        f'def from_dt_{metric}_{method}(ratios: np.array) -> float:',
        f'    {", ".join(FEATURE_NAMES)} = tuple(ratios)',
    ]

    def extract_lines(node, depth):
        indent = "    " * depth
        if tree.tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree.tree_.threshold[node]
            tree_code.append(f'{indent}if {name} <= {threshold:.6e}:')

            extract_lines(tree.tree_.children_left[node], depth + 1)
            tree_code.append(f'{indent}else:')

            extract_lines(tree.tree_.children_right[node], depth + 1)
        else:
            tree_code.append(f'{indent}return {tree.tree_.value[node][0][0]:.6e}')

    extract_lines(0, 1)
    return '\n'.join(tree_code)


def extract_lr(model: LinearRegression, metric: str, method: str):
    coefficients = [f'{float(c):.6e}' for c in model.coef_]
    return '\n'.join([
        f'def from_lr_{metric}_{method}(ratios: np.array) -> float:',
        f'    return float(np.dot(np.asarray(',
        f'        a=[{", ".join(coefficients)}],',
        f'        dtype=float,',
        f'    ), ratios))'
    ])


def _build_data(method_function, graph, manifold, labels):
    cluster_scores: ClusterScores = method_function(graph)
    train_x = np.zeros(shape=(len(cluster_scores), 6), dtype=np.float32)
    train_y = np.zeros(shape=(len(cluster_scores),))
    for i, (cluster, score) in enumerate(cluster_scores.items()):
        train_x[i] = manifold.cluster_ratios(cluster)
        y_pred = np.asarray([score for _ in cluster.argpoints], dtype=np.float32)
        y_true = np.asarray([labels[p] for p in cluster.argpoints], dtype=np.float32)
        loss = np.sum(np.mean(np.square(y_pred - y_true))) / cluster.cardinality
        train_y[i] = 1. - loss

    return train_x, train_y


def train_models(train_datasets: List[str], num_epochs: int) -> Dict[str, str]:
    meta_models = {
        metric: {
            method: (LinearRegression(), DecisionTreeRegressor(max_depth=3))
            for method in CHAODA.method_names
        }
        for metric in METRICS
    }
    train_x_dict = {
        metric: {method: list() for method in CHAODA.method_names}
        for metric in METRICS
    }
    train_y_dict = {
        metric: {method: list() for method in CHAODA.method_names}
        for metric in METRICS
    }

    for dataset in train_datasets:
        print('-' * 200)
        print(f'Dataset: {dataset}')
        print('-' * 200)

        normalization_mode = 'gaussian' if dataset in NOT_NORMALIZED else None
        data, labels = read(dataset, normalization_mode)

        chaoda: CHAODA = CHAODA(
            metrics=METRICS,
            max_depth=MAX_DEPTH,
            min_points=assign_min_points(data.shape[0]),
        )
        manifolds: List[Manifold] = chaoda.build_manifolds(data)
        # noinspection PyProtectedMember
        inner_methods = list(chaoda._names.items())

        for epoch in range(num_epochs):
            print('-' * 160)
            print(f'\tEpoch: {epoch + 1} of {num_epochs}')
            print('-' * 160)

            for manifold in manifolds:
                metric = manifold.metric

                # build the graphs
                if epoch == 0:
                    graphs = [layer.build_edges() for layer in manifold.layers[5::5]]
                else:
                    criteria = [
                        MetaMLSelect(lambda arr: meta_model.predict([arr]))
                        for models in meta_models[metric].values()
                        for meta_model in models
                    ]
                    graphs = [Graph(*criterion(manifold.root)).build_edges() for criterion in criteria]

                # filter large graphs for slow methods
                method_function_graph_triplets = list()
                for method, function in inner_methods:
                    for graph in graphs:
                        if graph.cardinality > 128 and method in chaoda.slow_methods:
                            continue
                        method_function_graph_triplets.append((method, function, graph))

                # build the data
                for method, function, graph in method_function_graph_triplets:
                    print('-' * 120)
                    print(f'\t\tDataset: {dataset}, Epoch: {epoch + 1} of {num_epochs}, Metric: {metric}, Method: {method}')
                    print('-' * 120)
                    x, y = _build_data(function, graph, manifold, labels)
                    train_x_dict[metric][method].append(x), train_y_dict[metric][method].append(y)

                # train the meta-models
                for method, _, _ in method_function_graph_triplets:
                    train_x = np.concatenate(train_x_dict[metric][method], axis=0)
                    train_y = np.concatenate(train_y_dict[metric][method], axis=0)
                    lr_model, dt_model = meta_models[metric][method]
                    lr_model.fit(train_x, train_y)
                    dt_model.fit(train_x, train_y)

    model_codes: Dict[str, str] = dict()
    for metric in METRICS:
        for method in CHAODA.method_names:
            lr_model, dt_model = meta_models[metric][method]
            model_codes[f'lr_{metric}_{method}'] = extract_lr(lr_model, metric, method)
            model_codes[f'dt_{metric}_{method}'] = extract_dt(dt_model, metric, method)

    return model_codes


def write_meta_models(model_codes: Dict[str, str], out_path: str):
    with open(out_path, 'w') as fp:
        fp.write('import numpy as np\n\n')

        for code in model_codes.values():
            fp.write(f'\n{code}\n\n')

        fp.write('\nMETA_MODELS = {\n')
        for name in model_codes.keys():
            fp.write(f'    \'from_{name}\': from_{name},\n')
        fp.write('}\n')
    return


if __name__ == '__main__':
    np.random.seed(42), random.seed(42)
    # _train_datasets = ['vertebral', 'wine']  # for testing
    _train_datasets = list(sorted(np.random.choice(SAMPLING_DATASETS, 6, replace=False)))
    # print(_train_datasets)
    # exit(1)
    write_meta_models(
        train_models(_train_datasets, 10),
        os.path.join(SRC_DIR, 'meta_models.py')
    )
