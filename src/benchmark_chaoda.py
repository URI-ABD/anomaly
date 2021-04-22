import random
from time import time
from typing import List
from typing import Optional

import numpy as np
from pyclam import CHAODA
from sklearn.metrics import roc_auc_score

import datasets
# You can replace this next import with your own generated file to verify results.
#  Remember to change the parsing for META_MODELS just below.
# import meta_models_trained
import meta_models
from utils import *

_NORMALIZATIONS = [
    None,
    'linear',
    'gaussian',
    'sigmoid',
]

META_MODELS = [
    # tuple of (metric, method, function)
    (name.split('_')[2], '_'.join(name.split('_')[3:]), method)
    for name, method in meta_models.META_MODELS.items()
]


def _score_dataset(dataset: str, fast: bool, report_individual: Optional[str]) -> Tuple[float, float]:
    """ Runs CHAODA on the dataset and returns the tuple of (auc-score, time-taken).

    :param dataset: Name of the dataset to benchmark
    :param fast: Whether to use the speed heuristic
    :param report_individual:
    """
    data, labels = datasets.read(dataset, None, None)
    name = 'CHAODA-Fast' if fast else 'CHAODA'
    print_blurb(name, dataset, data.shape)
    speed_threshold = max(128, int(np.sqrt(data.shape[0]))) if fast else None
    if fast:
        print(f"Speed threshold set to {speed_threshold}.")

    start = time()
    detector: CHAODA = CHAODA(
        metrics=METRICS,
        max_depth=MAX_DEPTH,
        min_points=assign_min_points(data.shape[0]),
        meta_ml_functions=META_MODELS,
        speed_threshold=speed_threshold,
    ).fit(data=data)

    if report_individual is not None:
        # Print individual method scores.

        if not os.path.exists(report_individual):
            with open(report_individual, 'w') as fp:
                columns = ','.join([
                    'dataset',
                    'cardinality',
                    'dimensionality',
                    'num_components',
                    'num_clusters',
                    'num_edges',
                    'min_depth',
                    'max_depth',
                    'method',
                    'auc_roc',
                ])
                fp.write(f'{columns}\n')

        index = 0
        # noinspection PyProtectedMember
        for method_name, graph in detector._graphs:
            # noinspection PyProtectedMember
            scores = detector._individual_scores[index]
            auc = roc_auc_score(labels, scores)
            index += 1

            with open(report_individual, 'a') as fp:
                features = ','.join([
                    dataset,
                    f'{data.shape[0]}',
                    f'{data.shape[1]}',
                    f'{len(graph.components)}',
                    f'{graph.cardinality}',
                    f'{len(graph.edges)}',
                    f'{graph.depth_range[0]}',
                    f'{graph.depth_range[1]}',
                    method_name,
                    f'{auc:.2f}',
                ])
                fp.write(f'{features}\n')

    # Carry on with the ensemble.
    return float(roc_auc_score(labels, detector.scores)), float(time() - start)


def run_chaoda(dataset_names: List[str], fast: bool, report_individual: str = None):
    """ Calculates anomaly scores for all datasets using the CHAODA class.
    """
    scores_df, times_df = get_dataframes()
    model_name = 'CHAODA_FAST' if fast else 'CHAODA'
    for i, dataset in enumerate(dataset_names):
        s, t = _score_dataset(dataset, fast, report_individual)
        scores_df.at[model_name, dataset] = s
        times_df.at[model_name, dataset] = t
        scores_df.to_csv(SCORES_PATH, float_format='%.2f')
        times_df.to_csv(TIMES_PATH, float_format='%.2e')

    return


if __name__ == '__main__':
    np.random.seed(42), random.seed(42)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    _individual_scores_path = os.path.join(RESULTS_DIR, 'individual_scores.csv')
    _datasets = datasets.DATASET_NAMES
    # _datasets = ['vertebral']

    run_chaoda(
        dataset_names=_datasets,
        fast=True,  # Set this to False to ignore the speed heuristic
        report_individual=_individual_scores_path,
    )
