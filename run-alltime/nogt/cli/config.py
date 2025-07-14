import os
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRJ_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

DATA_DIR = PRJ_DIR + 'datasets/'
DATA_FOLDER = os.path.join(DATA_DIR, 'filtzero_without_lastround')

OUT_DIR = PRJ_DIR + 'run-alltime/nogt/out/'
