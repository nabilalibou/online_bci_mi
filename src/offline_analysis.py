"""
"""
import mne
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.preprocessing_utils import get_reported_bad_trials, offline_preprocess

matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")
mne.viz.set_browser_backend("qt")


doPrepro = True
data_repo = "../data"
save_prepro = f"{data_repo}/preprocessed/"  # None
subjects = ["na", "mp"]  # "na", "mp"
experiments = ["left_arm", "right_arm", "right_leg"]  # "left_arm", "right_arm", "right_leg"
paradigm = ["me", 'mi']  # 'me'  'mi'
l_freq, h_freq = 2, 50
doICA = "before_epoching"  # 'before_epoching'  'after_epoching'  None
work_on_sources = False  # if True, save ICs sources and their topomaps for each conditions
epoch_time = (-1.5, 5)  # (-1, 0)
bad_trials = get_reported_bad_trials()  # Trials reported as wrong by the subjects

count = 0
for subj in subjects:
    for exp in experiments:
        if doPrepro:
            epochs_prepro = offline_preprocess(
                subj,
                exp,
                data_repo=data_repo,
                l_freq=l_freq,
                h_freq=h_freq,
                epoch_time=epoch_time,
                doICA=doICA,
                work_on_sources=work_on_sources,
                bad_trials=bad_trials,
                save_prepro_repo=save_prepro,
            )
        else:
            if work_on_sources:
                prepro_repo = f"{save_prepro}ics_sources/"
                prefix = "sources"
            else:
                prepro_repo = save_prepro
                prefix = "prepro"
            for task in paradigm:
                epoch_pickle_path = (
                    f"{prepro_repo}{prefix}_{subj}_{exp}_{task}_filt({l_freq}_{h_freq})_"
                    f"basl{(epoch_time[0], 0)}_ICA{doICA}.pkl"
                )
                try:
                    with open(epoch_pickle_path, "rb") as file1:
                        epochs = pickle.load(file1)
                except FileNotFoundError as e:
                    raise FileNotFoundError(e)
                epoch_events = epochs.events[:, 2]
                if not count:
                    X = epochs.get_data(picks="eeg", tmin=epoch_time[0])
                    Y = epoch_events
                else:
                    X = np.concatenate(
                        (X, epochs.get_data(picks="eeg", tmin=epoch_time[0])), axis=0
                    )
                    Y = np.hstack((Y, epoch_events))
            count += 1

# Construct X, Y
# Extract features
# Cross-validation and Evaluate intra-subject, inter-subject
