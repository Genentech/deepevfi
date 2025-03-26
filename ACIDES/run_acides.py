import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy
from importlib import reload
import os
import argparse

import sys
sys.path.append(".")
sys.path.append("/home/shenm19/prj/df-manuscript/ACIDES/module/")

from module import ACIDES_module
from module.ACIDES_module import ACIDES
from module.ACIDES_module import ACIDES_FIGURE_AVERAGE

import warnings
warnings.filterwarnings("ignore")


def main(args: dict):
    data_set = pd.read_csv(args['csv'], index_col = 0)

    round_cols = args['round_cols'].split(args['round_cols_delimiter'])
    rounds_df = data_set[round_cols]
    print(f'Using {round_cols=}')
    print(f'df shape: {rounds_df.shape}')

    # init times
    if args['times'] == 'default':
        times = list(range(len(round_cols)))
    else:
        times = [float(v) for v in args['times'].split(',')]
    t_rounds = np.array(times)
    print(f'Using {t_rounds=}')

    ## prepare a random number generator object 
    seed_num = int(args['random_seed'])
    rng = np.random.RandomState(seed_num)

    ## make a directory in which the results will be saved
    os.makedirs(os.path.dirname(args['output_folder']), exist_ok = True)

    fit = ACIDES(
        Inference_type = 'Negbin',
        theta_term_yes = 'yes',
        t_rounds = t_rounds.copy(),
        folder_name = args['output_folder'],
        random_num = rng,
        para_n_jobs = 6
    )
    fit.fit(
        _data_set_ = rounds_df.copy(),
        negbin_iterate = 1, 
        # Fixed_abneg = [0.87,0.25]
    )
    fit.fit_after_fixing_parameters(average_range_0 = 1)

    result = ACIDES_FIGURE_AVERAGE(
        Inference_type = 'Negbin',
        howmany_average = 1,
        save_folder = args['output_folder'],
        t_rounds = t_rounds.copy(),
        n_jobs = 4
    )
    results_dict = result.dummy_dictionary['data_all']
    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(args['output_folder'] + '/results.csv')

    # merge with input csv
    gt_col = args.get('genotype_col')
    results_df['index'] = results_df.index
    data_set['index'] = data_set.index

    mdf = data_set.merge(results_df, on = 'index', how = 'outer')
    mdf['ACIDES inferred fitness'] = np.exp(mdf['a_inf'])

    mdf.to_csv(args['output_folder'] + '/merged.csv')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """ Runs ACIDES. """
    )
    parser.add_argument('--csv',
        # required = True,
        default = '/home/shenm19/prj/df-manuscript/ACIDES/data/genotype_col/Data-B.csv')
    parser.add_argument('--genotype_col',
        default = 'Genotype')
    parser.add_argument('--round_cols', 
        help = 'Columns for time-series rounds, interpreted sequentially',
        default = 'c_0,c_18,c_37,c_45'
    )
    parser.add_argument('--times',
        help = 'Comma-delimited list of float time for each round',
        default = 'default',
    )
    parser.add_argument('--round_cols_delimiter', default = ',', 
        help = 'Delimiter character'
    )
    parser.add_argument('--random_seed', default = '0', 
        help = 'Random seed'
    )
    parser.add_argument('--output_folder', 
        # required = True,
        default = '/home/shenm19/prj/df-manuscript/ACIDES/output/'
    )

    args = parser.parse_args()

    main(vars(args))