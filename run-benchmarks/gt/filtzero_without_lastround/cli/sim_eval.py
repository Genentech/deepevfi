import pandas as pd
from loguru import logger
from tqdm import tqdm
import os, json
import argparse
import subprocess
import pandas as pd
from hackerargs import args
from collections import defaultdict

OUT_DIR = '/evfi-manuscript-public/run-benchmarks/gt/filtzero_without_lastround/out/'

PRJ_DIR = '/evfi-manuscript-public/'
DATA_DIR = PRJ_DIR + 'datasets/'
deepfitness_dir = os.path.join(PRJ_DIR, 'deepfitness')

layout_df = pd.read_csv(os.path.join(DATA_DIR, 'layout.csv'))
dataset_names = list(layout_df['Dataset'])

methods = ['deep_latent']
method_to_fitness_col = {
    'deep_latent': 'Deep inferred fitness',
}
method_to_inpfn = {
    'deep_latent': 'train_rounds_train_gts.csv',
}
datasets = [
    'C',
    'D',
    'F',
    'FGFR1-AHOent0.25',
    'TEAD-1fc-p2tl',
]

# only run simeval-latent on `latent` methods:
latent_methods = ['deep_latent']


"""
    Crawling
"""
class InferencePath:
    def __init__(self, path: str):
        """ Inference path is the output folder of an inference method:
            {OUT_DIR}/{dataset}/{method}/rep-{replicate}/
        """
        assert OUT_DIR in path
        [dataset, method, rep_str] = path.replace(OUT_DIR, '').strip('/').split('/')
        assert dataset in dataset_names
        assert method in methods
        assert 'rep-' in rep_str
        self.path = path
        self.dataset = dataset
        self.method = method
        self.rep_str = rep_str
        self.replicate = int(rep_str.replace('rep-', ''))

    """
        Basic
    """
    @staticmethod
    def get_from(dataset: str, method: str, rep: int) -> str:
        """ Get inference path from dataset, method, rep. """
        assert dataset in dataset_names
        assert method in methods
        return os.path.join(OUT_DIR, dataset, method, f'rep-{rep}')

    def get_simeval_out_path(self, simeval_name: str) -> str:
        return os.path.join(self.path, simeval_name) + '/'
    
    """
        Finished status
    """
    def simeval_run_done(self, simeval_name: str) -> bool:
        """ Returns True if simeval run can be skipped:
            simeval output folder exists, and its files were created
            at a time later than inference files.
        """
        out_path = self.get_simeval_out_path(simeval_name)
        if os.path.isdir(out_path):
            if len(self.get_files(out_path)) == 0:
                return False
            earliest_out_ctime = min(self.get_file_creation_times(out_path))
            inference_ctime = self.get_inference_finish_time()
            if earliest_out_ctime > inference_ctime:
                return True
        return False
    
    def get_files(self, path: str) -> list[str]:
        return [file for file in os.listdir(path)
                if os.path.isfile(os.path.join(path, file))]

    def get_file_creation_times(self, path: str) -> list[int]:
        """ Get list of creation times for each file in path. """
        return [os.path.getctime(os.path.join(path, file))
                for file in self.get_files(path)]

    def get_inference_finish_time(self) -> int:
        inference_csv = os.path.join(self.path, method_to_inpfn[self.method])
        return os.path.getctime(inference_csv)


def expected_paths() -> list[str]:
    """ Enumerate expected paths, using datasets and methods. """
    return [os.path.join(OUT_DIR, dataset, method) for dataset in dataset_names
            for method in methods]


def crawl() -> list[InferencePath]:
    """
        Crawl working directory, enumerating inference folders
        written for each dataset, method, and replicate.
    """
    data_method_paths = expected_paths()
    logger.info(f'Checking all expected paths exist ... ')
    for data_method_path in data_method_paths:
        if not os.path.isdir(data_method_path):
            logger.warning(f'Expected path but does not exist:\n\t{data_method_path}')
    data_method_paths = [p for p in data_method_paths if os.path.isdir(p)]

    # find replicate folders by matching 'rep-' pattern
    logger.info(f'Finding replicate paths ... ')
    is_rep_folder = lambda folder: folder[:4] == 'rep-'
    inf_paths = []
    for dm_path in data_method_paths:
        rep_folders = sorted([f for f in os.listdir(dm_path) if is_rep_folder(f)])
        logger.info(f'{dm_path}: Found {len(rep_folders)} replicates.')
        inf_paths.extend([InferencePath(os.path.join(dm_path, f))
                          for f in rep_folders])
    return inf_paths


"""
    Running commands
"""
"""
    Get command
"""
def get_simeval_extrapolative_command(inference_path: InferencePath) -> str:
    """ Build command for simeval, extrapolative. """
    dataset = inference_path.dataset
    method = inference_path.method

    data_row = layout_df[layout_df['Dataset'] == dataset].iloc[0]
    round_before = data_row['Second to last round']
    round_after = data_row['Last round']

    steps_per_round = data_row['Steps per round']
    step_to_last_round = float(steps_per_round.split(',')[-1])

    output_folder = os.path.join(inference_path.path, 'simeval-extrapolate') + '/'

    cmd = ' '.join([
        'python -m deepfitness.scripts.research.simulate_and_eval',
        f'--csv {os.path.join(inference_path.path, method_to_inpfn[method])}',
        f'--genotype_col Genotype',
        f'--fitness_col \"{method_to_fitness_col[method]}\"',
        f'--round_before {round_before}',
        f'--round_after {round_after}',
        f'--steps {step_to_last_round}',
        f'--output_folder {output_folder}'
    ])
    return cmd


def get_simeval_latent_command(inference_path: InferencePath) -> str:
    """ Build command for simeval, extrapolative. """
    dataset = inference_path.dataset
    method = inference_path.method

    assert method in latent_methods
    
    data_row = layout_df[layout_df['Dataset'] == dataset].iloc[0]
    round_before = data_row['Second to last round']
    round_after = data_row['Last round']
    round_cols = data_row['Round cols']
    steps_from_r0 = data_row['Time to last round']

    output_folder = os.path.join(inference_path.path, 'simeval-latent') + '/'

    cmd = ' '.join([
        'python -m deepfitness.scripts.research.sim_eval_latent',
        f'--csv {os.path.join(inference_path.path, method_to_inpfn[method])}',
        f'--genotype_col Genotype',
        f'--fitness_col \"{method_to_fitness_col[method]}\"',
        f'--log_abundance_col \"log_abundance\" '
        f'--round_cols [{round_cols}]',
        f'--steps_from_r0 {steps_from_r0}',
        f'--output_folder {output_folder}'
    ])
    return cmd


def run_all_simevals(
    inference_paths: list[InferencePath], 
    force_rerun: bool = False
) -> None:
    cmds = []
    for inf_path in inference_paths:        
        if force_rerun or not inf_path.simeval_run_done('simeval-extrapolate'):
            cmds.append(get_simeval_extrapolative_command(inf_path))

        if inf_path.method in latent_methods:
            if force_rerun or not inf_path.simeval_run_done('simeval-latent'):
                cmds.append(get_simeval_latent_command(inf_path))
    logger.info(f'Found {len(cmds)} commands ...')

    pcs = []
    for cmd in cmds:
        logger.info(cmd)
        # subprocess.check_output(cmd, cwd = deepfitness_dir, shell = True)
        pc = subprocess.Popen(cmd, cwd = deepfitness_dir, shell = True)
        pcs.append(pc)

    import time
    wait_time = 0.1
    get_n_done = lambda pcs: sum(pc.poll() != None for pc in pcs)
    n_done = get_n_done(pcs)
    with tqdm(total = len(pcs)) as pbar:
        while get_n_done(pcs) < len(pcs):
            new_n_done = get_n_done(pcs)
            pbar.update(new_n_done - n_done)
            n_done = new_n_done
            time.sleep(wait_time)
        new_n_done = get_n_done(pcs)
        pbar.update(new_n_done - n_done)

    return


def collate(inference_paths: list[InferencePath]) -> pd.DataFrame:
    """ Crawl simeval output folders, and collate results into
        dataframe, saving to file.
    """
    dd = defaultdict(list)
    for inf_path in inference_paths:
        dataset = inf_path.dataset
        method = inf_path.method
        replicate = inf_path.replicate

        metrics_json = os.path.join(inf_path.path, 'simeval-extrapolate', 'metrics.json')
        with open(metrics_json) as f:
            d = json.load(f)

        dd['Dataset'].append(dataset)
        dd['Method'].append(method)
        dd['Replicate'].append(replicate)
        dd['Evaluation type'].append('Extrapolate')
        for key in d:
            dd[key].append(d[key])

        if inf_path.method in latent_methods:

            metrics_json = os.path.join(inf_path.path, 'simeval-latent', 'metrics.json')
            with open(metrics_json) as f:
                d = json.load(f)
            
            dd['Dataset'].append(dataset)
            dd['Method'].append(method)
            dd['Replicate'].append(replicate)
            dd['Evaluation type'].append('Latent')
            for key in d:
                dd[key].append(d[key])

    df = pd.DataFrame(dd)
    df = df.sort_values(['Dataset', 'Method', 'Evaluation type', 'Replicate'])
    df = df.reset_index(drop = True)
    return df


def main(args: dict):
    inference_paths = crawl()
    run_all_simevals(inference_paths, force_rerun = args['force_rerun'])

    collated_df = collate(inference_paths)
    out_csv = os.path.join(OUT_DIR, 'benchmark-gt-filtzero.csv')
    collated_df.to_csv(out_csv)
    logger.success(f'Wrote {collated_df.shape=} to {out_csv}')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = """ 
            Crawl inference output paths, and run simulation/evaluation
            on held-out last round.
        """
    )
    parser.add_argument('--force-rerun',
        help = 'Generate all commands',
        action = 'store_true'
    )
    args = parser.parse_args()

    main(vars(args))