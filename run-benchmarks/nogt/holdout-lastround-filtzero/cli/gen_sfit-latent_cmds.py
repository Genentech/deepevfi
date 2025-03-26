import os
import argparse
import pandas as pd

import config

layout_df = pd.read_csv(config.DATA_DIR + 'layout.csv')

NUM_REPLICATES = 5

METHOD_NAME = 'sfit-latent'
METHOD_CMD = f'python -m deepfitness.scripts.train_simple_latent'


def get_out_folder(dataset_name: str, replicate: int) -> str:
    return f'{config.OUT_DIR}/{dataset_name}/{METHOD_NAME}/rep-{replicate}/'


def get_commands(force_rerun: bool) -> list[str]:
    """ If force_rerun, generate all commands.
        Otherwise, only generate commands with missing output folders.
    """
    cmds = []
    for idx, row in layout_df.iterrows():
        dataset_name = row['Dataset']
        inp_csv = os.path.join(config.DATA_FOLDER, row['Filename'])
        round_cols_without_last = row['Round cols without last round']
        steps_without_last = row['Steps per round without last round']

        print(dataset_name)

        for replicate in range(NUM_REPLICATES):
            out_folder = get_out_folder(dataset_name, replicate)

            if not force_rerun and os.path.isdir(out_folder):
                continue

            cmd = f'ts -G 1 {METHOD_CMD} ' + \
                f'--csv {inp_csv} --round_cols [{round_cols_without_last}] ' + \
                f'--genotype_col Genotype ' + \
                f'--simple.loss_func dirichlet_multinomial ' + \
                f'--steps_per_round [{steps_without_last}] ' + \
                f'--tsngs.skip_filters True ' + \
                f'--random_seed {replicate} ' + \
                f'--output_folder {out_folder}'
            cmds.append(cmd)
    return cmds


def main(args: dict):
    commands = get_commands(force_rerun = args['force_rerun'])
    out_shell = f'run_{METHOD_NAME}.sh'
    with open(out_shell, 'w') as f:
        f.write('\n'.join(commands))
    print(f'Wrote {len(commands)} commands to {out_shell}')
    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """ Generates commands.
            By default, only generates commands
            for runs with missing output folders.
        """
    )
    parser.add_argument('--force-rerun',
        help = 'Generate all commands',
        action = 'store_true'
    )
    args = parser.parse_args()

    main(vars(args))