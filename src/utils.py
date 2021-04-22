import os
from typing import Tuple

SRC_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
PLOTS_DIR = os.path.join(ROOT_DIR, 'plots')
UMAPS_DIR = os.path.join(ROOT_DIR, 'umaps')

SCORES_PATH = os.path.join(RESULTS_DIR, 'scores.csv')
TIMES_PATH = os.path.join(RESULTS_DIR, 'times.csv')

METRICS = ['cityblock', 'euclidean']
NORMALIZE = 'gaussian'
SUB_SAMPLE = 64_000  # for testing the implementation
MAX_DEPTH = 50  # even though no dataset reaches this far


def get_dataframes():
    import datasets
    import pandas as pd

    if not os.path.exists(SCORES_PATH):
        scores_df = pd.DataFrame(columns=datasets.DATASET_NAMES)
        scores_df.index.name = 'model'
        times_df = pd.DataFrame(columns=datasets.DATASET_NAMES)
        times_df.index.name = 'model'
    else:
        scores_df = pd.read_csv(SCORES_PATH, index_col='model')
        times_df = pd.read_csv(TIMES_PATH, index_col='model')
    return scores_df, times_df


def print_blurb(model: str, dataset: str, shape: Tuple[int, int]):
    print()
    print('-' * 80)
    print(f'Running model {model} on dataset \'{dataset}\' with shape {shape}.')
    print('-' * 80)
    print()
    return


def assign_min_points(num_points: int) -> int:
    """ Improves runtime speed without impacting scores. set to 1 if willing to wait. """
    return max((num_points // 1000), 1)
