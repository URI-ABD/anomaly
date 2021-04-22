import os
from typing import List

import pandas as pd

import datasets as chaoda_datasets
import utils

TRAIN_DATASETS: List[str] = list(sorted([
    'annthyroid',
    'mnist',
    'pendigits',
    'satellite',
    'shuttle',
    'thyroid',
]))
TEST_DATASETS: List[str] = list(sorted([
    _d for _d in chaoda_datasets.DATASET_LINKS.keys()
    if _d not in TRAIN_DATASETS
]))


def bold_best(values: List[str], *, margin: float = 0.02, high: bool = True) -> List[str]:
    """ Highlight the best value in the given values. """
    to_indices, ex_indices = list(), list()
    for i, v in enumerate(values):
        if v == 'TO':
            to_indices.append(i)
        elif v == 'EX':
            ex_indices.append(i)
        else:
            continue

    for i in to_indices:
        values[i] = '-1'
    for i in ex_indices:
        values[i] = '-1'

    values: List[float] = [float(v) for v in values]
    if high:
        threshold: float = max(values) - margin
        values: List[str] = [''.join(['\\textbf{', f'{v:.2f}', '}']) if v >= threshold else f'{v:.2f}' for v in values]
    else:
        threshold: float = max(min(values), 0.1) * (1 + margin)
        values: List[str] = [''.join(['\\textbf{', f'{v:.2f}', '}']) if v <= threshold else f'{v:.2f}' for v in values]

    for i in range(len(values)):
        if i in to_indices:
            values[i] = '\\textit{TO}'
        elif i in ex_indices:
            values[i] = '\\textit{EX}'
        else:
            continue

    return values


def bold_column(column: List[str]) -> List[str]:
    return [''.join(['\\textbf{' + c + '}']) for c in column]


# noinspection DuplicatedCode
def parse_csv(mode: str, datasets: List[str]):
    if mode not in ['scores', 'times']:
        raise ValueError(f'mode must be \'scores\' or \'times\'. Got {mode} instead.')

    high = mode == 'scores'
    path = utils.SCORES_PATH if high else utils.TIMES_PATH
    raw_df: pd.DataFrame = pd.read_csv(path, dtype=str)

    new_df: pd.DataFrame = pd.DataFrame()
    models = list(sorted(raw_df['model'].tolist()))
    if 'CHAODA' in models:
        models[models.index('CHAODA')] = models[0]
        models[0] = 'CHAODA'
    if 'CHAODA-Fast' in models:
        models[models.index('CHAODA-Fast')] = models[1]
        models[1] = 'CHAODA-Fast'

    new_df['model'] = models
    for dataset in datasets:
        short = chaoda_datasets.SHORT_NAMES[dataset]
        new_df[short] = bold_best(raw_df[dataset].tolist(), high=high)

    return new_df, bold_column(list(new_df.columns))


def get_latex(mode: str, datasets: List[str]):
    df, columns = parse_csv(mode, datasets)

    latex_string: str = df.to_latex(
        header=columns,
        index=False,
        column_format='|' + 'c|' * len(columns),
        escape=False,
    )
    latex_list = latex_string.split('\n')
    latex_list.pop(1)
    latex_list.pop(2)
    latex_list.pop(-3)

    for i in range(len(latex_list[:-2]), 0, -1):
        latex_list.insert(i, '\\hline')

    return '\n'.join(latex_list)


def write_tables():
    out_path = os.path.join(utils.RESULTS_DIR, 'latex')

    def _write_tables(name: str, datasets: List[str]):
        path = f'{out_path}_scores_{name}.txt'
        with open(path, 'w') as fp:
            fp.write(get_latex('scores', datasets))

        path = f'{out_path}_times_{name}.txt'
        with open(path, 'w') as fp:
            fp.write(get_latex('times', datasets))

        return

    half_num = len(TEST_DATASETS) // 2

    _write_tables('train', TRAIN_DATASETS)
    _write_tables('test_1', TEST_DATASETS[:half_num])
    _write_tables('test_2', TEST_DATASETS[half_num:])
    # _write_tables('test_3', chaoda_datasets.OTHER_DATASETS)
    return


if __name__ == '__main__':
    write_tables()
