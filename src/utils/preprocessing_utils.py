import os
import mne
import csv
import numpy as np
import mne_icalabel
from autoreject import get_rejection_threshold, read_reject_log
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")
mne.viz.set_browser_backend("qt")


def get_cond_id(exp):
    """
    Return the id corresponding to each condition
    :param exp:
    :return:
    """
    match exp:
        case "left_arm":
            me_event_id = 0
            mi_event_id = 1
        case "right_arm":
            me_event_id = 2
            mi_event_id = 3
        case "right_leg":
            me_event_id = 4
            mi_event_id = 5
        case _:
            raise ValueError("'experiences' value not correct")
    return me_event_id, mi_event_id


def get_reported_bad_trials():
    """
    Return a dictionary containing the bad trials reported by the subjects
    :return:
    """
    bad_trials_dict = {
        "mp_left_arm": {"me": [], "mi": [22, 26, 36, 38]},
        "mp_right_arm": {"me": [17], "mi": [46]},
        "mp_right_leg": {"me": [], "mi": [49, 50]},
        "na_left_arm": {"me": [], "mi": [8, 36, 42, 49]},
        "na_right_arm": {"me": [39], "mi": [34]},
        "na_right_leg": {"me": [], "mi": []},
    }
    return bad_trials_dict


def map_chan_type(raw_eeg):
    """
    Return a dict mapping channel names to MNE types
    :param raw_eeg:
    :return:
    """
    chan_names = raw_eeg.ch_names
    mapping_type = {}
    for i in range(0, len(chan_names)):
        if "EOG" in chan_names[i]:
            mapping_type[chan_names[i]] = "eog"
        elif (
            "STATUS" in chan_names[i]
            or "TRIGGERS" in chan_names[i]
            or "Counter 2power24" in chan_names[i]
        ):
            mapping_type[chan_names[i]] = "stim"
        elif "M1" in chan_names[i] or "M2" in chan_names[i]:
            mapping_type[chan_names[i]] = "misc"
        else:
            mapping_type[chan_names[i]] = "eeg"
    return mapping_type


def get_cond_epochs(
    raw_eeg,
    events,
    event_id,
    epoch_tmin=-1.5,
    epoch_tmax=5,
    baseline=(-1.5, 0),
    basl_mode="mean",
    bad_trials=[],
):
    """
    Return epochs cleaned using AutoReject and input 'bad_trials'
    :param raw_eeg:
    :param events:
    :param event_id:
    :param epoch_tmin:
    :param epoch_tmax:
    :param baseline:
    :param basl_mode:
    :param bad_trials:
    :return:
    """
    if basl_mode == "mean":
        epochs = mne.Epochs(
            raw_eeg,
            events,
            event_id=event_id,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
            baseline=baseline,
        )
    else:
        epochs = mne.Epochs(
            raw_eeg,
            events,
            event_id=event_id,
            tmin=epoch_tmin,
            tmax=epoch_tmax,
            baseline=None,
        )
        epochs = mne.baseline.rescale(
            epochs.get_data(),
            raw_eeg.times,
            baseline=baseline,
            mode=basl_mode,
            copy=True,
        )
    reject_dict = get_rejection_threshold(epochs, decim=1)
    epochs.drop_bad(reject=reject_dict)
    idx_torem = np.array(bad_trials) - 1
    epochs.drop(idx_torem)
    return epochs


def annotate_by_markers(raw_eeg, subj, events_csv, me_event_id, mi_event_id):
    """
    Extract the 6 markers time stamp from the .events.csv file and convert them into annotations for
    the raw eeg signal. One marker was placed with TMSI amplifier on every 10 trials starting at
    trial 0.
    :param raw_eeg:
    :param subj:
    :param events_csv:
    :param mi_event_id:
    :param me_event_id:
    :return:
    """
    with open(events_csv, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        csv_dict = [row for row in csv_reader]
    markers_list = []
    for i, marker in enumerate(csv_dict):
        if marker["Events"] == "Marker 2":
            (h, m, s) = marker["Time"].split(":")
            (s, ms) = s.split(".")
            seconds = int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1e4
            markers_list.append(seconds)

    for i in range(len(markers_list)):
        # First marker for 'na' is on the start of the execution instead of the start of the
        # instruction
        if i == 0 and subj == "mp":
            me_start_sample = 5 + markers_list[i]
            mi_start_sample = 15 + markers_list[i]
        else:
            me_start_sample = markers_list[i]
            mi_start_sample = 10 + markers_list[i]
        if i < len(markers_list) - 1:
            if i == 0:
                me_stop_sample = me_start_sample + 25 * 9
                mi_stop_sample = mi_start_sample + 25 * 9
            else:
                me_stop_sample = me_start_sample + 25 * 10
                mi_stop_sample = mi_start_sample + 25 * 10
            me_events = mne.make_fixed_length_events(
                raw_eeg,
                id=me_event_id,
                start=me_start_sample,
                stop=me_stop_sample,
                duration=25,
            )
            mi_events = mne.make_fixed_length_events(
                raw_eeg,
                id=mi_event_id,
                start=mi_start_sample,
                stop=mi_stop_sample,
                duration=25,
            )
        else:
            me_sample_50 = me_start_sample * raw_eeg.info["sfreq"]
            mi_sample_50 = mi_start_sample * raw_eeg.info["sfreq"]
            me_events = np.array([int(me_sample_50), 0, me_event_id])
            mi_events = np.array([int(mi_sample_50), 0, mi_event_id])

        # merge the events and sort them by time sample
        events_segm = np.vstack((me_events, mi_events))
        # argsort() is passing back an array containing integer sequence of its parent
        events_segm = events_segm[events_segm[:, 0].argsort()]
        if not i:
            events = events_segm
        else:
            events = np.vstack((events, events_segm))
    annot_from_events = mne.annotations_from_events(
        events=events,
        event_desc=[me_event_id, mi_event_id],
        sfreq=raw_eeg.info["sfreq"],
    )
    return raw_eeg.set_annotations(annot_from_events), events


def perform_ica(raw_eeg, plot_topo=False, topo_saveas=None, return_sources=False, ic_min_var=0.01):
    """
    Perform Independent Component Analysis + automatically remove artifact with ICLabel
    :param raw_eeg:
    :param plot_topo:
    :param topo_saveas:
    :param return_sources:
    :param ic_min_var:
    :return:
    """
    nbr_ics = 15
    ica = mne.preprocessing.ICA(n_components=15, method="picard", max_iter="auto", random_state=97)
    ica.fit(raw_eeg)
    all_explained_var_ratio = ica.get_explained_variance_ratio(raw_eeg, ch_type="eeg")
    print(
        f"Fraction of variance explained by all the {nbr_ics} components: "
        f"{round(100 * all_explained_var_ratio['eeg'], 3)}%"
    )
    topo_fig = ica.plot_components(show=False)[0]
    if plot_topo:
        topo_fig.show()  # view topomaps of all ics
    if topo_saveas:
        path_ext = os.path.splitext(topo_saveas)[1]
        if path_ext not in [".png"]:
            raise ValueError(f"file path '{topo_saveas}' does not ends with format '.png'")
        try:
            topo_fig.savefig(topo_saveas, format=path_ext.split(".")[1], bbox_inches="tight")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{e}")

    # Check-up with mne find_bads_eog()
    if "eog" in raw_eeg.get_channel_types():
        ica.exclude = []
        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(raw_eeg)
        # # eog_indices, eog_scores = ica.find_bads_eog(raw_eeg, ch_name='Fpz')
        # ica.exclude = eog_indices

        # # barplot of ICA component "EOG match" scores
        # ica.plot_scores(eog_scores)
        #
        # # plot diagnostics
        # ica.plot_properties(raw_eeg, picks=eog_indices)
        #
        # # plot ICs applied to raw data, with EOG matches highlighted
        # ica.plot_sources(raw_eeg, show_scrollbars=False)

    # ICLabel classification of the IC components:
    ic_labels = mne_icalabel.label_components(raw_eeg, ica, method="iclabel")
    labels = ic_labels["labels"]

    if "eog" in raw_eeg.get_channel_types() and (ica.labels_["eog"] != eog_indices):
        raise ValueError("EOG ICs found by ICLabel and 'find_bads_eog' are different")

    exclude_dict = {}
    for idx, label in enumerate(labels):
        if label not in ["brain", "other"] and ic_labels["y_pred_proba"][idx] > 0.8:
            exclude_dict[idx] = {
                "label": label,
                "proba": ic_labels["y_pred_proba"][idx],
            }
    print(f"Excluding these ICA components:")
    for idx, val in exclude_dict.items():
        round_proba = round(int(100 * val["proba"]), 2)
        print(f"Component n°{idx} '{val['label']}' (probability: {round_proba}%)")
    print("Fraction of variance in EEG signal explained by the components that has been kept:")
    minor_brain_ics = []
    for i, ic in enumerate(ic_labels["labels"]):
        if i not in exclude_dict.keys():
            explained_var_ratio = ica.get_explained_variance_ratio(
                raw_eeg, components=[i], ch_type="eeg"
            )
            if explained_var_ratio["eeg"] < ic_min_var:
                minor_brain_ics.append(i)
            round_proba = int(100 * ic_labels["y_pred_proba"][i])
            round_var = 100 * explained_var_ratio["eeg"]
            print(
                "Component n°{} '{}' ({}%): {:.1f}%".format(i, ic, round_proba, round_var)
            )  # f"{}" cannot round
    # ica.plot_properties(raw_eeg, picks=kept_ics, verbose=False)
    if return_sources:
        ica.exclude.extend(minor_brain_ics)  # only keep the main sources to work on them
        return ica.get_sources(raw_eeg)
    else:
        return ica.apply(raw_eeg, exclude=list(exclude_dict.keys()))
