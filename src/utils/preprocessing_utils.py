"""
"""
import os
import warnings
import mne
import csv
from typing import Union
from pathlib import Path
import numpy as np
import mne_icalabel
import scipy
from pyprep import NoisyChannels
from autoreject import get_rejection_threshold, read_reject_log
from pyriemann.estimation import Covariances
from pyriemann.clustering import Potato
from pyriemann.utils.covariance import normalize
from autoreject import AutoReject, Ransac, get_rejection_threshold
import matplotlib
import matplotlib.pyplot as plt

# matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
# plt.switch_backend("Qt5Agg")
mne.viz.set_browser_backend("qt")


# Butterworth Bandpass Filter
# Source: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high], btype="bandpass", output='ba')
    return b, a


def butter_bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5) -> np.ndarray:
    """
    Applies a Butterworth bandpass filter to a 2D or 3D signal array.
    This function implements a digital Butterworth bandpass filter using the `scipy.signal.butter_bandpass`
    function and `scipy.signal.lfilter` for filtering. It takes the following arguments:

    Args:
        signal (np.ndarray): The input signal array. It can be either a 2D array
            representing a single channel or a 3D array representing multiple channels
            across trials. Supported array shapes are (n_time,) for 1D (deprecated),
            (n_chans, n_time) for 2D (single channel), and (n_trials, n_chans, n_time) for 3D (multiple channels).
        lowcut (float): Lower cutoff frequency of the filter in Hz.
        highcut (float): Upper cutoff frequency of the filter in Hz.
        fs (float): Sampling frequency of the signal in Hz.
        order (int, optional): The order of the filter (default: 5). Higher orders
            result in steeper roll-off at the cutoff frequencies but may also introduce
            phase distortion.

    Returns:
        np.ndarray: The filtered signal array with the same shape as the input signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    if len(signal.shape) == 2:
        n_chans, n_time = signal.shape
        n_trials = 1
    elif len(signal.shape) == 3:
        n_trials, n_chans, n_time = signal.shape
    else:
        raise ValueError("Wrong input signal shape. Need to be 2D or 3D")
    filtered = np.zeros(signal.shape)
    for j in range(n_trials):
        for i in range(n_chans):
            filtered[i] = scipy.signal.lfilter(b, a, signal[i])
    return filtered


def cut_into_windows(X: np.ndarray, y: np.ndarray, windows_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments a 3D time-series signal array (`X`) and its corresponding labels (`y`)
    into overlapping or non-overlapping windows.
    This function takes a 3D time-series signal array (`X`) with shape (n_samples, n_features, n_timesteps)
    and its corresponding labels (`y`)  and segments them into windows of a specified size (`windows_size`).
    It outputs two modified arrays:

    - Modified `X` array (shape can change): The function segments the time dimension (axis 2)
      of the input `X` array into windows of size `windows_size`. Overlapping windows are created
      if the original array length isn't perfectly divisible by `windows_size`.
    - Modified `y` array (reshaped): The function replicates the labels (`y`) for each window
      in the segmented `X` array.

    Args:
        X (np.ndarray): The 3D time-series signal array with shape (n_samples, n_features, n_timesteps).
        y (np.ndarray): The labels array corresponding to each sample in `X` (can have various shapes).
        windows_size (int): The size of the window to segment the time dimension of `X`.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the modified `X` and `y` arrays.

    Raises:
        ValueError: If `windows_size` is greater than 1 and the length of the time dimension
            in `X` isn't divisible by `windows_size`.
    """
    if windows_size > 1:
        if not X.shape[2] % windows_size == 0:
            raise ValueError(
                f"'{X.shape[2]}' not divisible by slide_windows_size value :'{windows_size}'"
            )
        # X = np.reshape(X, (slide_windows_size*X.shape[0], X.shape[1], -1))
        X_segm = np.zeros((windows_size * X.shape[0], X.shape[1], int(X.shape[2] / windows_size)))
        for i in range(X.shape[0]):
            for m in range(windows_size):
                k1 = m * int(X.shape[2] / windows_size)
                k2 = (m + 1) * int(X.shape[2] / windows_size)
                X_segm[i * windows_size + m, :, :] = X[i, :, k1:k2]
        X = X_segm
        y = []
        for i in range(0, len(y)):
            j = 0
            while j < windows_size:
                y.append(y[i])
                j += 1
        y = np.squeeze(y)
    return X, y


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
    epoch_time=(-1.5, 5),
    bad_trials=[],
):
    """
    Cut raw eeg signal into epochs and return them cleaned using input 'bad_trials'
    :param raw_eeg:
    :param events:
    :param event_id:
    :param epoch_time:
    :param bad_trials:
    :return:
    """
    epochs = mne.Epochs(
        raw_eeg,
        events,
        event_id=event_id,
        tmin=epoch_time[0],
        tmax=epoch_time[1],
        baseline=None,  # will apply baseline correction after
        verbose=False,
    )
    if bad_trials:
        idx_torem = np.array(bad_trials) - 1
        epochs.drop(idx_torem)
        print(f"dropped epochs corresponding to trials: {idx_torem}")
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


def get_good_eeg_chan(data: mne.io.Raw) -> list[str]:
    """
    Extracts a list of good EEG channel names from an MNE Raw object, excluding bad channels.
    This function identifies and returns a list of EEG channel names from an MNE Raw object
    that meet the following criteria:
    - **Channel Type:** The channel's `kind` attribute is 2, indicating EEG.
    - **Not a Bad Channel:** The channel name is not listed in the `bads` list within the Raw object's information.

    Args:
        data (mne.io.Raw): The MNE Raw object containing the EEG data.

    Returns:
        list[str]: A list of good EEG channel names.
    """
    good_eeg_chan = []
    for ch in data.info["chs"]:
        if ch["kind"] == 2 and ch["ch_name"] not in data.info["bads"]:
            good_eeg_chan.append(ch["ch_name"])
    return good_eeg_chan


def bad_by_peak_to_peak(data: np.ndarray, sfreq: float, window_secs: float = 1, reject_value: float = 100e-6) -> np.ndarray:
    """
    Identifies potentially bad channels based on median peak-to-peak amplitudes within non-overlapping windows.
    This function flags channels as potentially bad if their median peak-to-peak amplitude value
    within non-overlapping windows exceeds a specified rejection threshold. It works as follows:

    1. **Windowing and Peak-to-Peak Calculation:** The function divides the data into windows with a
       specified width (`window_secs`) and calculates the peak-to-peak amplitude for each window
       and channel.
    2. **Median Calculation:** For each channel, it calculates the median of the peak-to-peak amplitudes
       across all windows.
    3. **Thresholding:** It compares the median peak-to-peak value for each channel to a specified
       rejection threshold (`reject_value`). Channels with median peak-to-peak values above this threshold
       are considered potentially bad.

    Args:
        data (np.ndarray): The EEG data, expected to have shape (n_channels, n_timepoints).
        sfreq (float): The sampling frequency of the EEG data in Hz.
        window_secs (float, optional): The width of the non-overlapping windows in seconds (default: 1).
        reject_value (float, optional): The rejection threshold for median peak-to-peak amplitudes
            (default: 100e-6).

    Returns:
        np.ndarray: A boolean NumPy array of shape (n_channels,) indicating potentially bad channels
            (True for bad channels, False for good channels).
    """
    ch_deltas = np.zeros(data.shape[0])
    sfreq = int(sfreq)
    window = int(window_secs * sfreq)
    sum_deltas = np.zeros((data.shape[0], int(data.shape[1] / window)))
    for ch_idx in range(data.shape[0]):
        for s in range(int((data.shape[1] / window))):
            sum_deltas[ch_idx, s] = np.max(data[ch_idx, s * window : (s + 1) * window]) - np.min(
                data[ch_idx, s * window : (s + 1) * window]
            )
        ch_deltas[ch_idx] = np.median(sum_deltas[ch_idx, :])
    return np.asarray(np.greater(ch_deltas, reject_value))


def bad_by_PSD(data: Union[mne.io.BaseRaw, mne.Epochs], fmin: float = 0, fmax: float = np.inf, sd: float = 3) -> np.ndarray:
    """
    Identifies potentially bad channels based on deviations in their power spectral density (PSD) values.

    This function detects channels with unusually high or low PSD values compared to other channels
    or the expected PSD distribution, potentially indicating artifacts or recording issues.

    Args:
        data (Union[mne.io.BaseRaw, mne.Epochs]): The EEG data, either as an MNE Raw object or an MNE Epochs object.
        fmin (float, optional): The lower frequency bound for PSD calculation (default: 0 Hz).
        fmax (float, optional): The upper frequency bound for PSD calculation (default: infinity).
        sd (float, optional): The Z-score threshold for identifying potentially bad channels (default: 3).

    Returns:
        np.ndarray: A boolean NumPy array of shape (n_channels,) indicating potentially bad channels
            (True for bad channels, False for good channels).

    Raises:
        TypeError: If the `data` argument is not an MNE Raw or Epochs object.

    Notes:
        - Uses the Welch method for PSD calculation with Raw objects and the multitaper method with Epochs objects.
        - For Epochs objects, it compares each epoch's PSD to the median PSD across epochs for robust outlier detection.
        - Z-scores for PSD values are calculated to standardize comparisons across channels and frequencies.
    """
    if isinstance(data, mne.io.BaseRaw):
        method = "welch"
        data_size = len(get_good_eeg_chan(data))
        psd = data.compute_psd(method=method, fmin=fmin, fmax=fmax)
        log_psd = 10 * np.log10(psd.get_data())
        zscore_psd = scipy.stats.zscore(log_psd)
    elif isinstance(data, mne.Epochs):
        method = "multitaper"
        data_size = len(data)
        psd = data.compute_psd(method=method, fmin=fmin, fmax=fmax, adaptive=True)
        log_psd = 10 * np.log10(psd.get_data())
        psd_avg = np.median(log_psd, axis=0)
        zscore_psd = scipy.stats.zscore(np.sum(log_psd - psd_avg, axis=1))
    else:
        raise TypeError("data need to be a MNE Raw or Epochs object")
    mask = np.zeros(data_size, dtype=bool)
    for i in range(len(mask)):
        mask[i] = any(zscore_psd[i] > sd)
    return mask


def detect_badChan(
    raw_data, fmin=None, fmax=None, useRansac=False, keepEOG=False, Return_log=False, **kwargs
):
    """
    Detects bad or missing channels in MNE Raw data using various criteria.
    Todo: should not take into account 'bad' annotations like the functions from MNE (epoch, psd ...)
    This function identifies channels that exhibit characteristics indicative of
    noise or artifacts, including:

    - Flat signal (minimal activity)
    - Peak-to-peak amplitude.
    - High deviation from the mean
    - Insufficient correlation with other channels
    - Abnormal power spectral density (PSD) distribution

    Optionally, the RANSAC (RANdom SAmple Consensus) algorithm can be used
    for more robust bad channel detection.

    Parameters:
    - raw_data (MNE Raw): The MNE Raw data object containing the EEG channels.
    - fmin (float, optional): The lower frequency of interest for filtering. Defaults to None.
    - fmax (float, optional): The upper frequency of interest for filtering. Defaults to None.
    - useRansac (bool, optional): Whether to use RANSAC for bad channel detection. Defaults to False.
    - keepEOG (bool, optional): If True, attempt to retain frontal channels (starting with "F" or "AF")
        that might have been marked bad due to EOG artifacts. Defaults to False.
    - Return_log (bool, optional): If True, return a dictionary containing channel detection criteria
        and the associated bad channels. Defaults to False.
    - kwargs: Additional keyword arguments passed to the `pyprep.NoisyChannels` methods:
        - reject_peak_to_peak (windows=5, reject_value=100e-6): Peak-to-peak rejection parameters.
        - nc.find_bad_by_correlation (correlation_secs=1.0, correlation_threshold=0.2, frac_bad=0.01):
            Correlation rejection parameters.
        - nc.find_bad_by_deviation (deviation_threshold=6.0): Deviation rejection threshold.
        - bad_by_PSD (sd=3): PSD rejection standard deviation threshold.

    Returns:
    - list: A list of channel names identified as bad.
    - dict (optional): If `Return_log` is True, returns a dictionary with bad channels for each detection criterion.
    """
    assert isinstance(Return_log, bool)
    assert isinstance(keepEOG, bool)
    assert isinstance(useRansac, bool)
    deviation_threshold = kwargs.get("deviation_threshold", 6.0)
    correlation_threshold = kwargs.get("correlation_threshold", 0.2)
    raw_eeg = raw_data.copy().pick("eeg")
    config = {
        "global": {
            "ch_names": None,
            "low_freq": fmin,
            "high_freq": fmax,
            "deviation_threshold": deviation_threshold,
            "correlation_threshold": correlation_threshold,
        }
    }
    if keepEOG:
        config_eog = {
            "keepEOG": {
                "ch_names": [
                    ch
                    for ch in raw_eeg.info["ch_names"]
                    if ch.startswith("F") or ch.startswith("AF")
                ],
                "low_freq": 15,
                "high_freq": fmax,
                "deviation_threshold": deviation_threshold - 2,
                "correlation_threshold": correlation_threshold + 0.08,
            },
        }
        config.update(config_eog)
    log_dict = {"ptp": [], "flat": [], "correlation": [], "deviation": [], "psd": [], "ransac": []}
    bad_chans = []
    for k, v in config.items():
        raw_filtered = raw_eeg.copy().filter(
            v["low_freq"], v["high_freq"], v["ch_names"], verbose="ERROR"
        )

        # PTP rejection
        mask = bad_by_peak_to_peak(
            raw_filtered.get_data(), raw_filtered.info["sfreq"], reject_value=100e-6
        )
        channel_used = get_good_eeg_chan(raw_filtered)
        bad_by_ptp = np.asarray(channel_used)[mask]
        log_dict["ptp"] = list(bad_by_ptp)
        raw_filtered.info["bads"].extend(bad_by_ptp)
        if (
            len(bad_by_ptp) == channel_used
        ):  # if all channels bad by ptp no need to use NoisyChannels
            continue

        try:
            # Flat rejection
            nc = NoisyChannels(raw_filtered)  # auto run 'find_bad_by_nan_flat()' when instantiated
            log_dict["flat"].extend(nc.get_bads())

            # Correlation rejection
            nc.find_bad_by_correlation(correlation_threshold=v["correlation_threshold"])
            log_dict["correlation"].extend(nc.get_bads())

            # Deviation rejection
            nc.find_bad_by_deviation(deviation_threshold=v["deviation_threshold"])
            log_dict["deviation"].extend(nc.get_bads())
            raw_filtered.info["bads"].extend(nc.get_bads())

            # PSD rejection
            if not v["high_freq"]:
                if not raw_filtered.info["lowpass"]:
                    fmax = np.inf
                else:
                    fmax = raw_filtered.info["lowpass"]
            else:
                fmax = v["high_freq"]
            if not v["low_freq"]:
                if not raw_filtered.info["highpass"]:
                    fmin = 0
                else:
                    fmin = raw_filtered.info["highpass"]
            else:
                fmin = v["low_freq"]
            mask = bad_by_PSD(raw_filtered, fmin=fmin, fmax=fmax)
            channel_used = get_good_eeg_chan(raw_filtered)
            bad_by_psd = [chan for i, chan in enumerate(channel_used) if mask[i]]
            log_dict["psd"].extend(bad_by_psd)
            raw_filtered.info["bads"].extend(bad_by_psd)

            # Ransac rejection
            if useRansac:
                nc.find_bad_by_ransac()
                log_dict["ransac"].extend(nc.get_bads())
                print(f"Bads by ransac: {nc.bad_by_ransac}")
                raw_filtered.info["bads"].extend(nc.get_bads())
        except ValueError as e:  # all channels have been removed by ptp
            warnings.warn(
                f"ValueError: {e}. \nAll channels have been labelled bad by peak-to-peak."
            )
        bad_chans.extend(raw_filtered.info.get("bads", []))
    bad_chans = list(set(bad_chans))
    print(bad_chans)

    if Return_log:
        for k, v in log_dict.items():
            if v:
                log_dict[k] = list(set(v))
        return bad_chans, log_dict
    return bad_chans


def detect_badChan_by_ransac(epochs, ransac_corr=0.75, sample_prop=0.25):
    """
    Identifies bad channels in EEG epochs using the RANSAC algorithm.
    This function employs a modified RANSAC implementation from Autoreject
    that is specifically designed to handle EEG epochs (looks more stable and
    raise less random errors).

    Parameters:
    - epochs (Epochs): An MNE Epochs object containing preprocessed EEG epochs.
    - ransac_corr (float, optional): Minimum correlation coefficient required
       between predicted and actual signals for a channel to be considered good.
       Defaults to 0.75.
    - sample_prop (float, optional): Proportion of total channels to use for
       signal prediction in each RANSAC sample. Should be between 0 and 1,
       excluding those extremes. Defaults to 0.25.

    Returns:
    - list: A list of channel names identified as bad by RANSAC.
    """
    ransac = Ransac(min_corr=ransac_corr)
    ransac.fit(epochs)
    print(f"Additional bad channels found by RANSAC: {ransac.bad_chs_}")
    return ransac.bad_chs_


def perform_ica(eeg_data, plot_topo=False, topo_saveas=None, return_sources=False, ic_min_var=0.01):
    """
    Perform Independent Component Analysis + automatically remove artifact with ICLabel
    :param eeg_data:
    :param plot_topo:
    :param topo_saveas:
    :param return_sources:
    :param ic_min_var:
    :return:
    """
    nbr_ics = 15
    ica = mne.preprocessing.ICA(n_components=15, method="picard", max_iter="auto", random_state=97)
    ica.fit(eeg_data)
    all_explained_var_ratio = ica.get_explained_variance_ratio(eeg_data, ch_type="eeg")
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
    eog_indices = []
    if "eog" in eeg_data.get_channel_types():
        ica.exclude = []
        # find which ICs match the EOG pattern
        eog_indices, eog_scores = ica.find_bads_eog(eeg_data)
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
    ic_labels = mne_icalabel.label_components(eeg_data, ica, method="iclabel")
    labels = ic_labels["labels"]

    if "eog" in eeg_data.get_channel_types() and (ica.labels_["eog"] != eog_indices):
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
                eeg_data, components=[i], ch_type="eeg"
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
        return ica.get_sources(eeg_data)
    else:
        return ica.apply(eeg_data, exclude=list(exclude_dict.keys()))


def offline_preprocess(
    subject,
    experiment,
    data_repo="../data",
    l_freq=0.1,
    h_freq=50,
    epoch_time=(-1.5, 0),
    sfreq=512,
    doICA="after_epoching",
    work_on_sources=False,
    bad_trials=get_reported_bad_trials(),
    save_prepro_repo="../data/preprocessed",
):
    """
    Apply preprocessing on the raw eeg data extracted from the .fif files for each subject and
    experiments. If save_prepro=True, save the preprocessed files with pickle and the evoked
    erp/topomaps png picture.
    :param subject:
    :param experiment:
    :param data_repo:
    :param l_freq:
    :param h_freq:
    :param epoch_time:
    :param doICA:
    :param work_on_sources:
    :param bad_trials:
    :param save_prepro_repo:
    :return: epochs:
    """
    file = f"{data_repo}/{subject}_{experiment}/{subject}_{experiment}.fif"
    events_csv = f"{data_repo}/{subject}_{experiment}/{subject}_{experiment}.events.csv"
    me_event_id, mi_event_id = get_cond_id(experiment)
    cond_dict = {me_event_id: "me", mi_event_id: "mi"}
    raw_eeg = mne.io.read_raw_fif(file)
    raw_eeg.resample(sfreq=sfreq)
    raw_eeg.load_data()  # need to load data to rereference etc
    # raw_eeg.plot()

    # Set channel types
    mapping_type = map_chan_type(raw_eeg)
    raw_eeg.set_channel_types(mapping_type)
    # fig = raw_eeg.plot_sensors(show_names=True)
    # fig.show()

    # Filter
    # raw_eeg.notch_filter(50)
    raw_eeg.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # Find bad/missing channels and interpolate them
    raw_eeg.info["bads"], chan_drop_log = detect_badChan(
        raw_eeg, l_freq, h_freq, keepEOG=True, Return_log=True
    )
    if raw_eeg.info["bads"]:
        raw_eeg = raw_eeg.interpolate_bads(reset_bads=True)

    raw_eeg.set_eeg_reference(ref_channels="average", ch_type="auto")

    if doICA == "before_epoching":
        raw_eeg = perform_ica(raw_eeg, return_sources=work_on_sources)

    # Annotate according to events.csv markers time stamps
    raw_eeg, events = annotate_by_markers(raw_eeg, subject, events_csv, me_event_id, mi_event_id)

    # Make epochs of 5 seconds for the paradigm motor execution (me) and motor imagery (mi)
    for event_id in [me_event_id, mi_event_id]:
        list_bad_trials = bad_trials[f"{subject}_{experiment}"][cond_dict[event_id]]
        epochs = get_cond_epochs(
            raw_eeg,
            events,
            event_id,
            epoch_time=epoch_time,
            bad_trials=list_bad_trials,
        )
        epochs.load_data()
        if doICA == "after_epoching":
            topo_path = None
            if save_prepro_repo and work_on_sources:
                try:
                    Path(f"{save_prepro_repo}/ics_sources/").mkdir(parents=True, exist_ok=True)
                except FileExistsError:
                    pass
                topo_path = (
                    f"{save_prepro_repo}/ics_sources/sources_topo_{subject}_{experiment}_"
                    f"{cond_dict[event_id]}_filt({l_freq}_{h_freq})_basl{(epoch_time[0], 0)}_"
                    f"ICA{doICA}.png"
                )
            epochs = perform_ica(epochs, return_sources=work_on_sources, topo_saveas=topo_path)
        # Baseline removal
        epochs.apply_baseline()
        # Automatic epoch rejection using Autoreject
        reject_dict = get_rejection_threshold(epochs, decim=1)
        epochs.drop_bad(reject=reject_dict)

        if save_prepro_repo:
            prefix = "prepro"
            prepro_repo = save_prepro_repo
            if work_on_sources:
                prepro_repo = f"{save_prepro_repo}/ics_sources/"
                prefix = "sources"
            try:
                Path(f"{prepro_repo}").mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            prepro_file_path = (
                f"{prepro_repo}{prefix}_{subject}_{experiment}_{cond_dict[event_id]}_filt("
                f"{l_freq}_{h_freq})_basl{(epoch_time[0], 0)}_ICA{doICA}.fif"
            )
            erptopo_path = (
                f"{prepro_repo}{prefix}_{subject}_{experiment}_{cond_dict[event_id]}_filt("
                f"{l_freq}_{h_freq})_basl{(epoch_time[0], 0)}_ICA{doICA}.png"
            )
            topo_path = (
                f"{prepro_repo}{prefix}_topo_{subject}_{experiment}_{cond_dict[event_id]}_filt"
                f"({l_freq}_{h_freq})_basl{(epoch_time[0], 0)}_ICA{doICA}.png"
            )
            if not work_on_sources:
                evoked = epochs.average()
                erptopo_fig = evoked.plot_joint(
                    title=f"Subj:{subject} Exp:{experiment} Cond:" f"{cond_dict[event_id]}",
                    ts_args={"gfp": True},
                    show=False,
                )
                erptopo_fig.savefig(erptopo_path, format="png", bbox_inches="tight")
                times = np.arange(0, evoked.tmax, 0.1)
                topo_fig = evoked.plot_topomap(
                    times=times, average=0.050, ncols="auto", nrows="auto", show=False
                )
                topo_fig.savefig(topo_path, format="png", bbox_inches="tight")
            epochs.save(prepro_file_path, overwrite=True)
            print(f"file {prepro_file_path} along its evoked erp saved")

    return epochs
