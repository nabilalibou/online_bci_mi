import os
import mne
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pyprep import NoisyChannels
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
ICA = True
work_on_sources = False  # if True, ICA will be applied after the epoching
basl_time = (-1.5, 0)  # (-1, 0)
basl_mode = "mean"  # ‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
bad_trials = get_reported_bad_trials()  # Trials reported as meh by the subjects


if doPrepro:
    for subj in subjects:
        for exp in experiments:
            file = f"{data_repo}/{subj}_{exp}/{subj}_{exp}.fif"
            events_csv = f"{data_repo}/{subj}_{exp}/{subj}_{exp}.events.csv"
            me_event_id, mi_event_id = get_cond_id(exp)
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
                    f"basl{basl_time}_{basl_mode}_ICA{ICA}.pkl"
                )
                topo_path = (
                    f"{prepro_repo}{prefix}_topo_{subj}_{exp}_filt({l_freq}_{h_freq})_"
                    f"basl{basl_time}_{basl_mode}_ICA{ICA}.png"
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

            if ICA:
                raw_eeg = perform_ica(raw_eeg, plot_topo=False)
                baseline = basl_time
                if work_on_sources:
                    baseline = None  # Apply baseline removal after ICA

            # Annotate according to events.csv markers time stamps
            raw_eeg, events = annotate_by_markers(
                raw_eeg, subj, events_csv, me_event_id, mi_event_id
            )
            # Make epochs of 5 seconds for the paradigm motor execution (me) and motor imagery (mi)
            session_bad_trials = [
                bad_trials[f"{subj}_{exp}"]["me"],
                bad_trials[f"{subj}_{exp}"]["mi"],
            ]
            for event_id in [me_event_id, mi_event_id]:
                list_bad_trials = session_bad_trials[event_id]
                epochs = get_cond_epochs(
                    raw_eeg,
                    events,
                    event_id,
                    epoch_tmin=-1.5,
                    epoch_tmax=5,
                    baseline=(-1.5, 0),
                    basl_mode="mean",
                    bad_trials=list_bad_trials,
                )
                epochs.load_data()
                if save_prepro:
                    epoch_prepro_path = (
                        f"{data_prepro_path.split(f'{exp}')[0]}{exp}_me"
                        f"{data_prepro_path.split(f'{exp}')[1]}"
                    )
                    topo_path = (
                        f"{topo_path.split(f'{exp}')[0]}{exp}_me" f"{topo_path.split(f'{exp}')[1]}"
                    )
                else:
                    epoch_prepro_path = None
                if ICA:
                    if not work_on_sources:
                        epochs = perform_ica(epochs, plot_topo=False)
                    else:
                        epochs = perform_ica(
                            epochs,
                            plot_topo=False,
                            topo_saveas=topo_path,
                            return_sources=True,
                        )
                    baseline = basl_time
                    epochs = mne.baseline.rescale(
                        epochs.get_data(),
                        raw_eeg.times,
                        baseline=baseline,
                        mode=basl_mode,
                        copy=True,
                    )
                with open(epoch_prepro_path, "wb") as f1:
                    pickle.dump(epochs, f1)
                    print(f"file {epoch_prepro_path} saved")
