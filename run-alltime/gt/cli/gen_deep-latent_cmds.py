import os
import argparse
import yaml
from hackerargs import args


CONFIG_DIR = '/evfi-manuscript-public/run-alltime/gt/config/'

NUM_REPLICATES = 1
METHOD_NAME = 'deep-latent'
METHOD_CMD = 'python -m deepfitness.scripts.train_deep_latent'


dataset_to_config = {
    'C': os.path.join(CONFIG_DIR, 'deep_latent_c.yaml'),
    'D': os.path.join(CONFIG_DIR, 'deep_latent_d.yaml'),
    'F': os.path.join(CONFIG_DIR, 'deep_latent_f.yaml'),
    'FGFR1-AHOent0.25': os.path.join(CONFIG_DIR, 'deep_latent_fgfr1aho.yaml'),
    'TEAD-1fc-p2tl': os.path.join(CONFIG_DIR, 'deep_latent_tead_1fc_p2tl.yaml'),
}


def get_out_folder(dataset_name: str, replicate: int) -> str:
    config = dataset_to_config[dataset_name]
    with open(config, 'r') as f:
        d = yaml.safe_load(f)
    return os.path.join(d['project_output_folder'], f'rep-{replicate}/')


def get_commands(force_rerun: bool) -> list[str]:
    """ If force_rerun, generate all commands.
        Otherwise, only generate commands with missing output folders.
    """
    cmds = []
    for dataset_name, config_yaml in dataset_to_config.items():
        for replicate in range(NUM_REPLICATES):
            out_folder = get_out_folder(dataset_name, replicate)

            if not force_rerun and os.path.isdir(out_folder):
                continue

            cmd = f'ts -G 1 {METHOD_CMD} ' + \
                f'--config {config_yaml} ' + \
                f'--random_seed {replicate} ' + \
                f'--project_output_folder {out_folder} ' + \
                f'--wandb.use False'
            cmds.append(cmd)
    return cmds


def main():
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
    args.parse_args(parser)

    main()