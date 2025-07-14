"""
    Wrapper script to run Enrich2. Converts CLI args into json config,
    then calls enrich CLI tool.
    Must be run in enrich2 conda env. 
"""
import matplotlib
matplotlib.use('Agg')
import argparse, os, subprocess, json
from collections import defaultdict
import pandas as pd


def split_csv_to_tsvs(args):
    """ Returns dict of timepoint (int): filename of tsv. """
    df = pd.read_csv(args['csv'])
    genotype_col = args['genotype_col']
    round_cols = args['round_cols'].split(args['round_cols_delimiter'])

    t_to_fn = {}
    for t, round_col in enumerate(round_cols):
        sub_df = df[[genotype_col, round_col]]
        sub_df = sub_df.set_index(genotype_col)
        sub_df = sub_df.rename(columns = {round_col: 'count'})
        sub_df.index.name = None

        fn = args['output_folder'] + '/' + str(t) + '.tsv'
        sub_df.to_csv(fn, sep = '\t')
        t_to_fn[t] = fn
    return t_to_fn


def write_json_config(args, t_to_fn):
    json_config_fn = args['output_folder'] + '/enrich2_config.json'

    dd = defaultdict(dict)
    dd['name'] = 'name'
    dd['output directory'] = args['output_folder']
    dd['conditions'] = [{
        'name': 'conditions',
        'selections': [{
            'name': 'selections',
            'libraries': []
        }]
    }]

    for t, fn in t_to_fn.items():
        round_dict = {
            'name': str(t),
            'timepoint': t,
            'counts file': fn,
            'identifiers': {},
        }
        dd['conditions'][0]['selections'][0]['libraries'].append(round_dict)

    with open(json_config_fn, 'w') as f:
        f.write(json.dumps(dict(dd), indent = 2))
    return json_config_fn


def main(args):
    try:
        os.makedirs(os.path.dirname(args['output_folder']))
    except OSError:
        pass

    # split CSV into multiple tsvs
    t_to_fn = split_csv_to_tsvs(args)

    json_config_fn = write_json_config(args, t_to_fn)

    # call enrich2
    round_cols = args['round_cols'].split(args['round_cols_delimiter'])
    if len(round_cols) > 2:
        scoring_method = 'WLS'
    elif len(round_cols) == 2:
        scoring_method = 'ratios'

    command = 'enrich_cmd ' + json_config_fn + ' ' + scoring_method + ' full --no-plots'
    subprocess.check_output(command, shell = True)

    # merge
    out_dir = args.get('output_folder')
    gt_col = args.get('genotype_col')
    inp_df = pd.read_csv(args.get('csv'))
    out_df = pd.read_csv(out_dir + '/tsv/selections_sel/main_identifiers_scores.tsv', sep = '\t')

    inp_df['index'] = inp_df.index
    out_df['index'] = out_df.index
    mdf = inp_df.merge(out_df, on = 'index', how = 'outer')
    import numpy as np
    mdf['Enrich2 inferred fitness'] = np.exp(mdf['score'])
    mdf.to_csv(out_dir + 'merged.csv')

    subprocess.check_output('rm -rf ' + out_dir + '/tsv/', shell = True)
    subprocess.check_output('rm -rf ' + out_dir + '/*tsv', shell = True)
    subprocess.check_output('rm -rf ' + out_dir + '/*h5', shell = True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """ Runs enrich2 on count data """
    )
    parser.add_argument('--csv',
        # required = True,
        default = 'data/Data-B.csv')
    parser.add_argument('--genotype_col', 
        default = 'Genotype'
    )
    parser.add_argument('--round_cols', 
        help = 'Columns for time-series rounds, interpreted sequentially',
        default = 'c_0,c_18,c_37,c_45'
    )
    parser.add_argument('--round_cols_delimiter', default = ',', 
        help = 'Delimiter character'
    )
    parser.add_argument('--output_folder', 
        # required = True,
        default = 'output/test/'
    )

    args = parser.parse_args()

    main(vars(args))