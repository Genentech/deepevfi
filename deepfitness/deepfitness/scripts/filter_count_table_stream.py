"""
    Applies standard filters to csv: readcount, min fq, min max read

    Streaming - used to read very large CSVs tar gzs
"""
from loguru import logger
import subprocess
import argparse
import pandas as pd
from tqdm import tqdm
import numpy as np

from hackerargs import args


class ChunkWriter:
    """ Write to CSV in chunks. """
    def __init__(self, filename):
        self.filename = filename
        self.header_written = False
    
    def write(self, chunk):
        if self.header_written:
            chunk.to_csv(self.filename, mode='a', index=False, header=False)
        else:
            chunk.to_csv(self.filename, mode='w', index=False)
            self.header_written = True


def main():
    chunksize = 100000
    csv_file = args.get('csv')
    chunk_reader = pd.read_csv(
        csv_file, 
        chunksize = chunksize, 
        iterator = True
    )
    total = subprocess.check_output(f'wc -l {csv_file}', shell = True)
    total = int(total.split()[0])

    round_cols = [str(r) for r in args.get('round_cols')]
    genotype_col = args.get('genotype_col')
    rounds_before = round_cols[:-1]
    rounds_after = round_cols[1:]
    threshold = args.setdefault('filter_max_readcount_threshold', 10)

    writer = ChunkWriter(args.get('output_csv'))
    timer = tqdm(total = total)
    for df in chunk_reader:
        df.loc[:, round_cols] = df.loc[:, round_cols].fillna(0)

        # filter max readcount
        crit = (df[round_cols].max(axis='columns') >= threshold)
        df = df[crit]

        # filter consecutive
        if args.setdefault('filt_consec', True):
            has_consec = np.zeros(len(df), dtype = bool)
            for r0, r1 in zip(rounds_before, rounds_after):
                has_consec |= (df[r0] > 0) & (df[r1] > 0)
            df = df[has_consec]

        # filter duplicate genotypes in chunk - does not guarantee full output df has no duplicates
        df = df[~df[genotype_col].duplicated()]

        # save
        writer.write(df)
        timer.update(len(df))
    timer.close()
    return


if __name__ == '__main__':
    """
        To run in package, use
        > python -m deepfitness.scripts.filter_count_table_stream
    """
    parser = argparse.ArgumentParser(
        description = """
            Applies standard data processing filters to count table: readcounts,
            min fq, min max reads.
            
            Tip: To head on tar gz file, use:
            tar -xzOf some_huge_file.tar.gz | head
        """
    )
    parser.add_argument('--csv', required = True,
        help = 'csv file, can be compressed'
    )
    parser.add_argument('--genotype_col', required = True,
        help = 'Name of column containing genotypes'
    )
    parser.add_argument('--round_cols', 
        help = 'Columns for time-series rounds, interpreted sequentially'
    )
    parser.add_argument('--rounds_before', 
        help = 'Columns for time-series rounds; before[i] -> after[i]'
    )
    parser.add_argument('--rounds_after', 
        help = 'Columns for time-series rounds; before[i] -> after[i]'
    )
    parser.add_argument('--output_csv', required = True)
    args.parse_args(parser)

    main()
