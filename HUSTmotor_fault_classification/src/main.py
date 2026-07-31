from load_data import load_data
from preprocessing import preprocess
from experiments import run_exp_1
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import metrics
from sklearn.dummy import DummyClassifier

data_by_id = load_data()



if (not Path("data/vibration_features.csv").exists() or
    not Path("data/sound_features.csv").exists()):
    preprocess(data_by_id)

#EXPERIMENT 1


#load data to dataframes
vibrations_df = pd.read_csv("data/vibration_features.csv")
sound_df = pd.read_csv("data/sound_features.csv")

#VIBRATION DATA ONLY

exp1_results, exp1_matrices = run_exp_1(vibrations_df)

