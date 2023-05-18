import os
import mne
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyprep import NoisyChannels
from autoreject import get_rejection_threshold, read_reject_log
from utils.preprocessing_utils import (
    get_cond_id,
    get_reported_bad_trials,
    perform_ica,
    map_chan_type,
    annotate_by_markers,
    get_cond_epochs,
)

matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")
mne.viz.set_browser_backend("qt")


doPrepro = True
save_prepro = True
data_repo = "../data"
subjects = ["na", "mp"]  # "na", "mp"
experiments = [
    "left_arm",
    "right_arm",
    "right_leg",
]  # "left_arm", "right_arm", "right_leg"
l_freq, h_freq = 2, 50
doICA = "after_epoching"  # 'before_epoching'  'after_epoching'  None
work_on_sources = False  # if True, return ICs sources for each conditions
baseline_time = (-1.5, 0)  # (-1, 0)
basl_mode = "mean"  # ‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
bad_trials = get_reported_bad_trials()  # Trials reported as meh by the subjects


if doPrepro:
    for subj in subjects:
        session_bad_trials = []
        for exp in experiments:
            file = f"{data_repo}/{subj}_{exp}/{subj}_{exp}.fif"
            events_csv = f"{data_repo}/{subj}_{exp}/{subj}_{exp}.events.csv"
            me_event_id, mi_event_id = get_cond_id(exp)
            baseline = baseline_time
            raw_eeg = mne.io.read_raw_fif(file)
            raw_eeg.resample(sfreq=512)
            raw_eeg.load_data()  # need to load data to rereference etc
            # raw_eeg.plot()
            if save_prepro:
                if work_on_sources:
                    prepro_repo = f"{data_repo}/preprocessed/ics_sources/"
                    prefix = "sources"
                else:
                    prepro_repo = f"{data_repo}/preprocessed/"
                    prefix = "prepro"
                data_prepro_path = (
                    f"{prepro_repo}{prefix}_{subj}_{exp}_filt({l_freq}_{h_freq})_"
                    f"basl{baseline}{basl_mode}_ICA{doICA}.pkl"
                )
                topo_path = (
                    f"{prepro_repo}{prefix}_topo_{subj}_{exp}_filt({l_freq}_{h_freq})_"
                    f"basl{baseline}{basl_mode}_ICA{doICA}.png"
                )
            else:
                data_prepro_path = None
                topo_path = None

            # Set channel types
            mapping_type = map_chan_type(raw_eeg)
            raw_eeg.set_channel_types(mapping_type)
            # fig = raw_eeg.plot_sensors(show_names=True)
            # fig.show()

            # Filter
            # raw_eeg.notch_filter(50)
            raw_eeg.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose=False)
            # raw_filtered = butter_bandpass_filter(raw_eeg, l_freq, h_freq, raw_eeg.info['sfreq'])

            # Find bad/missing channels and interpolate them
            nc = NoisyChannels(raw_eeg)
            nc.find_bad_by_correlation(correlation_threshold=0.2)
            nc.find_bad_by_deviation(deviation_threshold=6.0)
            print(nc.get_bads())

            if raw_eeg.info["bads"]:
                raw_eeg = raw_eeg.interpolate_bads(reset_bads=True)

            raw_eeg.set_eeg_reference(ref_channels="average", ch_type="auto")

            if doICA == "before_epoching":
                if work_on_sources:
                    raw_eeg = perform_ica(
                        raw_eeg,
                        plot_topo=False,
                        topo_saveas=topo_path,
                        return_sources=True,
                    )
                else:
                    raw_eeg = perform_ica(raw_eeg, plot_topo=False)
            elif doICA == "after_epoching":
                baseline = None  # Baseline correction will be applied after ICA

            # Annotate according to events.csv markers time stamps
            raw_eeg, events = annotate_by_markers(
                raw_eeg, subj, events_csv, me_event_id, mi_event_id
            )
            # Make epochs of 5 seconds for the paradigm motor execution (me) and motor imagery (mi)
            session_bad_trials.extend(
                bad_trials[f"{subj}_{exp}"]["me"] + bad_trials[f"{subj}_{exp}"]["mi"]
            )
            for event_id in [me_event_id, mi_event_id]:
                list_bad_trials = session_bad_trials[event_id]
                epochs = get_cond_epochs(
                    raw_eeg,
                    events,
                    event_id,
                    epoch_tmin=-1.5,
                    epoch_tmax=5,
                    baseline=baseline,
                    basl_mode=basl_mode,
                    bad_trials=list_bad_trials,
                )
                if doICA != "after_epoching":
                    # Reject bad epochs using Autoreject
                    reject_dict = get_rejection_threshold(epochs, decim=1)
                    epochs.drop_bad(reject=reject_dict)
                epochs.load_data()
                if save_prepro:
                    epoch_prepro_path = (
                        f"{data_prepro_path.split(f'{exp}')[0]}{exp}"
                        f"{data_prepro_path.split(f'{exp}')[1]}"
                    )
                    topo_path = (
                        f"{topo_path.split(f'{exp}')[0]}{exp}{topo_path.split(f'{exp}')[1]}"
                    )
                else:
                    epoch_prepro_path = None

                if doICA == "after_epoching":
                    if work_on_sources:
                        epochs = perform_ica(
                            epochs,
                            plot_topo=False,
                            topo_saveas=topo_path,
                            return_sources=True,
                        )
                    else:
                        epochs = perform_ica(epochs, plot_topo=False)
                    # Once ICA applied, remove the baseline and reject bad epochs using Autoreject
                    baseline = baseline_time
                    # epochs.apply_baseline(baseline)
                    epochs_arr = mne.baseline.rescale(
                        epochs.get_data(),
                        raw_eeg.times,
                        baseline=baseline,
                        mode=basl_mode,
                        copy=True,
                    )
                    epochs = mne.EpochsArray(epochs_arr, epochs.info, verbose=False)
                    reject_dict = get_rejection_threshold(epochs, decim=1)
                    epochs.drop_bad(reject=reject_dict)

                if save_prepro:
                    with open(epoch_prepro_path, "wb") as f1:
                        pickle.dump(epochs, f1)
                        print(f"file {epoch_prepro_path} saved")
