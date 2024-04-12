"""
"""
import os
import warnings
import mne
import csv
from typing import Union, Optional
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
from utils.data_utils import auto_weight_chan_dict
from matplotlib import pyplot as plt

# matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
# plt.switch_backend("Qt5Agg")
# mne.viz.set_browser_backend("qt")


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


# def cut_into_windows(X: np.ndarray, y: np.ndarray, windows_size: int) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Segments a 3D time-series signal array (`X`) and its corresponding labels (`y`)
#     into overlapping or non-overlapping windows.
#     This function takes a 3D time-series signal array (`X`) with shape (n_samples, n_features, n_timesteps)
#     and its corresponding labels (`y`)  and segments them into windows of a specified size (`windows_size`).
#     It outputs two modified arrays:
#
#     - Modified `X` array (shape can change): The function segments the time dimension (axis 2)
#       of the input `X` array into windows of size `windows_size`. Overlapping windows are created
#       if the original array length isn't perfectly divisible by `windows_size`.
#     - Modified `y` array (reshaped): The function replicates the labels (`y`) for each window
#       in the segmented `X` array.
#
#     Args:
#         X (np.ndarray): The 3D time-series signal array with shape (n_samples, n_features, n_timesteps).
#         y (np.ndarray): The labels array corresponding to each sample in `X` (can have various shapes).
#         windows_size (int): The size of the window to segment the time dimension of `X`.
#
#     Returns:
#         tuple[np.ndarray, np.ndarray]: A tuple containing the modified `X` and `y` arrays.
#
#     Raises:
#         ValueError: If `windows_size` is greater than 1 and the length of the time dimension
#             in `X` isn't divisible by `windows_size`.
#     """
#     if windows_size > 1:
#         if not X.shape[2] % windows_size == 0:
#             raise ValueError(
#                 f"'{X.shape[2]}' not divisible by slide_windows_size value :'{windows_size}'"
#             )
#         # X = np.reshape(X, (slide_windows_size*X.shape[0], X.shape[1], -1))
#         X_segm = np.zeros((windows_size * X.shape[0], X.shape[1], int(X.shape[2] / windows_size)))
#         for i in range(X.shape[0]):
#             for m in range(windows_size):
#                 k1 = m * int(X.shape[2] / windows_size)
#                 k2 = (m + 1) * int(X.shape[2] / windows_size)
#                 X_segm[i * windows_size + m, :, :] = X[i, :, k1:k2]
#         X = X_segm
#         y = []
#         for i in range(0, len(y)):
#             j = 0
#             while j < windows_size:
#                 y.append(y[i])
#                 j += 1
#         y = np.squeeze(y)
#     return X, y


def cut_into_windows(X: np.ndarray, y: np.ndarray, window_size: int, stride=1) -> tuple[np.ndarray, np.ndarray]:
    """
    Segments a 3D time-series signal array (`X`) and its corresponding labels (`y`)
    into overlapping or non-overlapping windows.

    Args:
        X (np.ndarray): The 3D time-series signal array with shape (n_samples, n_features, n_timesteps).
        y (np.ndarray): The labels array corresponding to each sample in `X` (can have various shapes).
        window_size (int): The size of the window to segment the time dimension of `X`.
        stride (int, optional): The number of steps to slide between consecutive windows. Defaults to 1 (non-overlapping).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the modified `X` and `y` arrays.

    Raises:
        ValueError: If `window_size` is larger than the length of the time dimension in `X`.
    """

    if window_size > X.shape[2]:
        raise ValueError(f"Window size ({window_size}) cannot be larger than signal length ({X.shape[2]})")

    # Calculate the number of windows based on window size and stride
    n_windows = (X.shape[2] - window_size + 1) // stride

    # Use efficient sliding window view for window creation
    X_segm = np.lib.stride_tricks.sliding_window_view(X, window_shape=(X.shape[1], window_size))

    # Reshape to create the final segmented X array
    X_segm = X_segm.reshape(n_windows, X.shape[0], X.shape[1], window_size)

    # Handle labels (replicate for each window using tile)
    y_ = np.tile(y[:, np.newaxis], (1, n_windows))

    return X_segm, y_


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


def fix_montage_eeg(
    data: Union[mne.io.Raw, mne.Epochs],
    ref_chan: str = "",
    montage_ref: str = None,
    head_size: float = 0.095,
) -> Union[mne.io.Raw, mne.Epochs]:
    """
    Corrects channel names, assigns standard MNE channel types,
    and sets a specified montage (optional) for an MNE data object.
    This function takes an MNE Raw or Epochs object (`data`), optionally removes
    a reference channel name from channel names (`ref_chan`), assigns standard MNE
    channel types (`eeg`, `eog`, `misc`, `stim`) based on heuristics, and sets a
    standard or custom montage (`montage_ref`) with a specified head size
    (`head_size`).

    Args:
        data (Union[mne.io.Raw, mne.Epochs]): The MNE Raw or Epochs data object.
        ref_chan (str, optional): The reference channel name to remove from channel
            names (default: None).
        montage_ref (str, optional): The name of the standard montage to use
            (e.g., "biosemi-64-channel") or the path to a custom montage file
            (default: "").
        head_size (float, optional): The head size in meters to use for standard
            montage construction (default: 0.095).

    Returns:
        Union[mne.io.Raw, mne.Epochs]: The modified data object with corrected channels
            and optionally the specified montage.
    """
    data_ = data.copy()
    original_channel_names = data_.ch_names
    new_channel_names = [
        name.replace(f"-{ref_chan}", "") if ref_chan else name
        for name in original_channel_names
    ]
    channel_name_mapping = {}
    channel_type_mapping = {}
    for name in new_channel_names:
        channel_name_mapping[name] = name
        if ref_chan and ref_chan in name:
            channel_type_mapping[name] = "misc"
        elif "EOG" in name:
            channel_type_mapping[name] = "eog"
        elif any(
            term in name
            for term in ("STATUS", "TRIGGERS", "Counter 2power24")
        ):
            channel_type_mapping[name] = "stim"
        elif any(term in name for term in ("M1", "M2")):
            channel_type_mapping[name] = "misc"
        else:
            channel_type_mapping[name] = "eeg"
    allow_duplicates = False
    if len(data_.ch_names) != len(np.unique(data_.ch_names)) and not allow_duplicates:
        raise ValueError("New channel names are not unique, renaming failed")
    mne.rename_channels(data_.info, channel_name_mapping, allow_duplicates=True)
    data_.set_channel_types(channel_type_mapping)
    if montage_ref:
        try:
            montage = mne.channels.make_standard_montage(montage_ref, head_size=head_size)
        except ValueError:
            montage = mne.channels.read_custom_montage(montage_ref, head_size=head_size)
        data_.set_montage(montage, match_case=False)
    # Optional: data.plot_sensors(show_names=True)
    return data_


def get_cond_epochs(
    raw_eeg: mne.io.Raw,
    events: np.ndarray,
    event_id: int,
    epoch_tmin: float = -1.5,
    epoch_tmax: float = 5.0,
    detrend: Optional[int] = None,
    bad_trials: list[int] = [],
) -> mne.Epochs:
    """
    Creates condition-specific epochs from a raw EEG signal and event markers.
    This function extracts epochs around events with the specified `event_id` from the provided
    raw EEG data (`raw_eeg`) and event markers (`events`). Bad trials listed in `bad_trials`
    are excluded before returning the cleaned epochs.

    Args:
        raw_eeg (mne.io.Raw): The raw EEG data.
        events (np.ndarray): A NumPy array of event markers with time stamps.
        event_id (int): The event ID to extract epochs for.
        epoch_tmin (float, optional): The minimum time in seconds before the event
            to include in the epoch (default: -1.5).
        epoch_tmax (float, optional): The maximum time in seconds after the event
            to include in the epoch (default: 5.0).
        detrend (Optional[int], optional): The detrending mode to apply before epoching
            (e.g., 0 for no detrending, 1 for linear detrending). Defaults to None.
        bad_trials (list[int], optional): A list of trial indices (starting from 1)
            to be excluded due to bad data (default: []).

    Returns:
        mne.Epochs: The condition-specific epochs object.
    """
    epochs = mne.Epochs(
        raw_eeg,
        events,
        event_id=event_id,
        tmin=epoch_tmin,
        tmax=epoch_tmax,
        detrend=detrend,
        baseline=None,  # Baseline correction can be applied later
        verbose=False,
    )

    if bad_trials:
        # Convert trial indices to epoch indices (starting from 0)
        epochs = epochs.drop(np.array(bad_trials) - 1)
        print(f"Dropped epochs corresponding to trials: {bad_trials}")

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
    The function segments the EEG data (expected shape: (n_channels, n_timepoints)) into windows of a specified width
    (`window_secs`) and calculates the median peak-to-peak amplitude for each channel across all windows.
    Channels exceeding a specified rejection threshold (`reject_value`) in median peak-to-peak amplitude are flagged as
    potentially bad.

    Args:
        data (np.ndarray): The EEG data.
        sfreq (float): The sampling frequency of the EEG data in Hz.
        window_secs (float, optional): The width of the non-overlapping windows in seconds (default: 1).
        reject_value (float, optional): The rejection threshold for median peak-to-peak amplitudes (default: 100e-6).

    Returns:
        np.ndarray: A boolean NumPy array of shape (n_channels,) indicating potentially bad channels (True for bad, False for good).
    """
    window_samples = int(window_secs * sfreq)
    n_windows = data.shape[1] // window_samples
    peak_to_peak = np.ptp(data[:, :n_windows * window_samples].reshape(-1, window_samples), axis=1)
    median_ptp = np.median(peak_to_peak.reshape(data.shape[0], n_windows), axis=1)

    return median_ptp > reject_value


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
    if not isinstance(data, (mne.io.BaseRaw, mne.Epochs)):
        raise TypeError("data must be an MNE Raw or Epochs object")

    if isinstance(data, mne.io.BaseRaw):
        method = "welch"
    else:
        method = "multitaper"

    # Efficiently calculate PSD, convert to dB, compute Z-scores across all channels and identify channels > threshold
    psd = data.compute_psd(method=method, fmin=fmin, fmax=fmax)
    log_psd = 10 * np.log10(psd.get_data())
    zscore_psd = scipy.stats.zscore(log_psd, axis=0)
    bad_channels = np.any(zscore_psd > sd, axis=1)

    return bad_channels


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


def bad_epoch_ptp(epochs, reject_value=200e-6):
    """
    Identifies bad epochs in an MNE Epochs object based on peak-to-peak (ptp) values.

    Args:
        epochs (mne.Epochs): The MNE Epochs object containing EEG data.
        reject_value (float, optional): The threshold for ptp value to be considered a bad epoch. Defaults to 200e-6.

    Returns:
        ndarray: A boolean array with the same length as the number of epochs,
                where True indicates a bad epoch based on ptp exceeding the threshold.
    """
    epochs_data = epochs.get_data(picks="eeg", return_event_id=False)
    ptp_values = np.max(epochs_data, axis=2) - np.min(epochs_data, axis=2)
    bad_epochs = ptp_values > reject_value

    return bad_epochs


def autoreject_bad_epochs(
    epochs: mne.Epochs,
    interpolate: bool = True,
    n_interpolate: np.ndarray = np.array([1, 4, 32]),
    plot_reject: bool = False,
    Return_reject_log: bool = False,
) -> Union[mne.Epochs, np.ndarray]:
    """
    Applies Autoreject for local bad sensor detection and handling.
    Autoreject is a data-driven outlier-detection method combined with physics-driven
    channel repair, where parameters are calibrated using a cross-validation strategy
    robust to outliers.

    **Note:** Local Autoreject with interpolation is recommended for at least 30 channels.

    Args:
        epochs (Epochs): The MNE Epochs object to clean.
        interpolate (bool, optional): If True, bad sensors will be interpolated
            (default: True).
        n_interpolate (np.ndarray, optional): The number of worst channels to
            interpolate in case of bad trials (default: [1, 4, 32]).
        plot_reject (bool, optional): If True, plots raw bad epochs and reject log
            (default: False).
        return_reject_log (bool, optional): If True, returns the reject log instead
            of the cleaned epochs (default: False).

    Returns:
        Union[Epochs, np.ndarray]: The cleaned epochs (default) or the reject log
            as a NumPy array if `return_reject_log` is True.
    """
    print("Running Autoreject...")
    epochs_copy = epochs.copy()
    ar = AutoReject(n_interpolate=n_interpolate, random_state=11, n_jobs=1, verbose=False)

    if interpolate:
        epochs_clean, reject_log = ar.fit_transform(epochs_copy, return_log=True)
    else:
        ar.fit(epochs_copy)
        reject_log = ar.get_reject_log(epochs_copy)
        epochs_clean = epochs.drop(reject_log.bad_epochs)

    # Handle plotting logic and potential errors
    if plot_reject and reject_log.bad_epochs.any():
        try:
            # Attempt to plot epochs with scalings
            epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
        except (AttributeError, ValueError):
            # Catch potential errors related to plotting (e.g., missing scalings argument)
            print("Error occurred during plotting. Consider adjusting plot parameters.")
        reject_log.plot("horizontal")  # May still show bad channels

    if not reject_log.bad_epochs.any():
        print("0 bad epochs found")

    return reject_log.bad_epochs if Return_reject_log else epochs_clean


def reject_badEpoch(
    data: mne.Epochs,
    method: list[str] = ["potato", "local_ar"],
    ch_names: list[str] = None,
    fmin: float = None,
    fmax: float = None,
    ptp_reject: float = 150e-6,
    potato_zscore_thresh: float = 3.0,
    local_ar_coeff_mult: float = 3.0,
    psd_zscore_thresh: float = 3.0,
    global_ar_n_interpolate: np.ndarray = np.array([1, 4, 32]),
    mode: str = None,
    chan_weight_dict: dict[str, float] = None,
    Return_log: bool = False,
) -> Union[mne.Epochs, tuple[mne.Epochs, dict]]:
    """
    Automatically rejects bad epochs based on a combination of algorithms.
    This function rejects bad epochs from the provided MNE Epochs object (`data`)
    using the specified rejection methods (`method`).  Channe selection (`ch_names`),
    frequency bands of interest (`fmin`, `fmax`), and various parameters for each
    rejection method can be customized.
    todo: add **kwargs for all the parameters.

    Args:
        data (Epochs): The MNE Epochs object containing EEG data.
        method (list[str], optional): A list of rejection methods to apply.
            Available options include: "potato", "local_ar", "global_ar", "psd"
            (default: ["potato", "local_ar"]).
        ch_names (list[str], optional): A list of channel names to focus on
            during rejection (default: None, all channels used).
        fmin (float, optional): The lower frequency of interest in Hz (default: None).
        fmax (float, optional): The upper frequency of interest in Hz (default: None).
        ptp_reject (float, optional): Peak-to-peak rejection threshold (default: 150e-6).
        potato_zscore_thresh (float, optional): Z-score threshold for Potato
            rejection (default: 3.0).
        local_ar_coeff_mult (float, optional): Coefficient multiplier for local AR
            rejection threshold (default: 3.0).
        psd_zscore_thresh (float, optional): Z-score threshold for PSD-based
            rejection (default: 3.0).
        global_ar_n_interpolate (np.ndarray, optional): Parameters for global AR
            rejection: number of worst channels to interpolate (default: [1, 4, 32]).
        mode (str, optional): Pre-configured rejection mode (e.g., "conservative",
            "keepEOG") (default: None).
        chan_weight_dict (dict[str, float], optional): Dictionary with channel
            names as keys and importance weights as values (default: None).
        Return_log (bool, optional): If True, returns a dictionary with bad epoch
            indices for each method and configuration (default: False).

    Returns:
        Union[Epochs, tuple[Epochs, dict]]: The cleaned epochs object, or a
            tuple containing the cleaned epochs and a dictionary with bad epoch
            logs (if `return_log` is True).
    """
    assert isinstance(Return_log, bool)
    if not isinstance(method, list):
        method = [method]
    if not all(item in ["ptp", "potato", "local_ar", "global_ar", "asr", "psd"] for item in method):
        raise ValueError("input 'method' invalid")
    elif method == "asr":
        raise ValueError("method 'asr' not available yet")
    assert local_ar_coeff_mult > 0
    assert psd_zscore_thresh > 0
    epochs = data.copy().pick("eeg")
    if mode not in ["keepEOG", "conservative", "pre_ica", None]:
        raise ValueError("mode not recognised")
    elif mode in ["conservative"]:
        if not chan_weight_dict:
            chan_weight_dict = auto_weight_chan_dict(epochs.pick("eeg").ch_names)
            ch_names = [chan for chan, val in chan_weight_dict.items() if val == 2]
        elif isinstance(chan_weight_dict, dict):
            ch_names = [chan for chan, val in chan_weight_dict.items() if val == 2]
            if not ch_names:
                raise ValueError("'chan_weight_dict' does not contains channel with a value of 2.")
            if not set(ch_names).issubset(epochs.ch_names):
                raise ValueError(
                    "Channels from input 'chan_weight_dict' need to match names with channels in input "
                    "epoch object"
                )
        else:
            raise ValueError("Input 'chan_weight_dict' need to be None or dict.")
    original_rej = epochs.drop_log.count(("USER",))
    if mode == "keepEOG":
        config = {
            "keepEOG": {
                "ch_names": [
                    ch for ch in epochs.ch_names if ch.startswith("F") or ch.startswith("AF")
                ],
                "low_freq": 15,
                "high_freq": fmax,
                # 'cov_normalization': 'trace'  # trace-norm to be insensitive to power
            }
        }
    elif mode == "conservative":
        config = {
            "conservative": {
                "ch_names": ch_names,
                "low_freq": 3,
                "high_freq": 31,
            }
        }
    elif mode == "pre_ica":
        config = {
            "pre_ica": {
                "ch_names": ch_names,
                "low_freq": 15,
                "high_freq": fmax,
            }
        }
    else:
        config = {
            "user_config": {
                "ch_names": ch_names,
                "low_freq": fmin,
                "high_freq": fmax,
            }
        }
    potato = None
    if "potato" in method:
        potato = Potato(threshold=potato_zscore_thresh)

    dict_mask = {}
    bad_epochs = []
    mask = None
    # If peak-to-peak rejection is present, do it first
    if "ptp" in method:
        for k, v in config.items():
            config_epochs = epochs.copy().filter(
                v["low_freq"], v["high_freq"], v["ch_names"], verbose="ERROR"
            )
            mask = bad_epoch_ptp(config_epochs, reject_value=ptp_reject)
            dict_mask[f"ptp_{k}"] = mask
        final_mask = np.zeros(len(epochs.selection), dtype=bool)
        for k, m in dict_mask.items():
            final_mask = final_mask | m
        epochs.drop(final_mask)
    for k, v in config.items():
        config_epochs = epochs.copy().filter(
            v["low_freq"], v["high_freq"], v["ch_names"], verbose="ERROR"
        )
        for m in method:
            if m == "ptp":
                pass
            if m == "potato":
                cov_mats_ = Covariances(estimator="scm").fit_transform(
                    config_epochs.get_data(copy=True)
                )
                # if v.get('cov_normalization'):
                #     cov_mats_ = normalize(cov_mats_, v.get('cov_normalization'))
                try:
                    potato.fit(cov_mats_)
                    mask = np.invert(potato.predict(cov_mats_).astype(bool))
                except ValueError as e:
                    warnings.warn(
                        f"ValueError: {e}. \nEstimating a second time with shrunk Ledoit-Wolf covariance "
                        f"matrices for regularization ..."
                    )
                    cov_mats_ = Covariances(estimator="lwf").fit_transform(
                        config_epochs.get_data(copy=True)
                    )
                    try:
                        potato.fit(cov_mats_)
                        mask = np.invert(potato.predict(cov_mats_).astype(bool))
                    except ValueError as e:
                        warnings.warn(
                            f"ValueError: {e}. \nEstimating a third time with oracle approximating shrunk covariance "
                            f"matrices for regularization ..."
                        )
                        cov_mats_ = Covariances(estimator="oas").fit_transform(
                            config_epochs.get_data(copy=True)
                        )
                        try:
                            potato.fit(cov_mats_)
                            mask = np.invert(potato.predict(cov_mats_).astype(bool))
                        except ValueError as e:
                            warnings.warn(
                                f"ValueError: {e}. \nPotato rejection will be skipped."
                            )
                            mask = np.zeros(len(config_epochs.selection), dtype=bool)
            if m == "global_ar":
                reject_thresh = {
                    "eeg": get_rejection_threshold(config_epochs, ch_types="eeg", cv=5)["eeg"]
                    * local_ar_coeff_mult
                }
                print(f"The rejection threshold is {reject_thresh}")
                rpf_epochs_reshape_ = np.reshape(
                    config_epochs.get_data(copy=True), (len(config_epochs.selection), -1)
                )
                mask = np.zeros(len(config_epochs.selection), dtype=bool)
                for i in range(len(mask)):
                    mask[i] = any(rpf_epochs_reshape_[i] > reject_thresh["eeg"])
            if m == "local_ar":
                mask = autoreject_bad_epochs(
                    config_epochs,
                    interpolate=False,
                    n_interpolate=global_ar_n_interpolate,
                    plot_reject=False,
                    Return_reject_log=True,
                )
            if m == "psd":
                if not v["high_freq"]:
                    if not config_epochs.info["lowpass"]:
                        fmax = np.inf
                    else:
                        fmax = config_epochs.info["lowpass"]
                else:
                    fmax = v["high_freq"]
                if not v["low_freq"]:
                    if not config_epochs.info["highpass"]:
                        fmin = 0
                    else:
                        fmin = config_epochs.info["highpass"]
                else:
                    fmin = v["low_freq"]
                mask = bad_by_PSD(config_epochs, fmin=fmin, fmax=fmax, sd=psd_zscore_thresh)

            dict_mask[f"{m}_{k}"] = mask
            print(
                f'{dict_mask[f"{m}_{k}"].sum()} bad epochs found by method {m} for configuration {k}'
            )
            bad_epochs.append(np.asarray(dict_mask[f"{m}_{k}"]))

    final_mask = np.zeros(len(epochs.selection), dtype=bool)
    for k, m in dict_mask.items():
        if not k.startswith("ptp"):
            final_mask = final_mask | m

    epochs.drop(final_mask)
    print(
        f"Total = {epochs.drop_log.count(('USER',)) - original_rej} bad epochs found by methods: {method}"
    )

    if Return_log:
        log_dict = {}
        for method, mask in dict_mask.items():
            method_name, config = method.split("_", 1)[0], method.split("_", 1)[1]
            if method_name not in log_dict.keys():
                log_dict[method_name] = {}
            log_dict[method_name][config] = np.where(mask)[0]
        return epochs, log_dict

    return epochs


def perform_ica(
    eeg_data: Union[mne.Epochs, mne.io.Raw],
    nbr_ics: int = None,
    ic_min_var: float = 0.01,
    proba_thresh: float = 0.7,
    return_sources: bool = False,
    plot_topo: bool = False,
    topo_saveas: str = None,
    Return_log: bool = False,
) -> Union[mne.Epochs, mne.io.Raw, tuple[Union[mne.Epochs, mne.io.Raw], dict]]:
    """
    Performs Independent Component Analysis (ICA) and removes artifact components
    using ICLabel classification.
    This function performs ICA on the provided MNE data (`eeg_data`). It then uses
    ICLabel to classify the independent components (ICs) and removes those exceeding
    a specified probability threshold of being artifacts. Optionally, EOG channels
    can be used to aid in EOG IC detection.

    Args:
        eeg_data (Union[Epochs, Raw]): The MNE Epochs or Raw data object.
        nbr_ics (int, optional): The number of ICA components to compute. If None,
            defaults to the number of good EEG channels (default: None).
        ic_min_var (float, optional): Minimum variance ratio threshold for keeping
            an IC (default: 0.01).
        proba_thresh (float, optional): Probability threshold for ICLabel
            classification (default: 0.7). Must be between 0 and 1.
        return_sources (bool, optional): If True, returns the estimated ICA sources
            (default: False).
        plot_topo (bool, optional): If True, plots the topographies of all ICs
            (default: False).
        topo_saveas (str, optional): Path to save the topographies plot as a PNG
            image (default: None).
        Return_log (bool, optional): If True, returns a dictionary with information
            about kept ICs and explained variance (default: False).

    Returns:
        Union[Epochs, Raw, tuple[Union[Epochs, Raw], dict]]: The cleaned data object,
            or a tuple containing the cleaned data and a dictionary with ICA
            processing information (if `return_log` is True).

    Raises:
        ValueError: If `proba_thresh` is not between 0 and 1.
        ValueError: If `topo_saveas` path does not end with the '.png' extension.
        ValueError: If EOG ICs detected by ICLabel and MNE differ (when EOG channels
            are present).
    """
    if proba_thresh < 0 or proba_thresh > 1:
        raise ValueError("proba_thresh must be between 0 and 1")
    ica = mne.preprocessing.ICA(
        n_components=nbr_ics, method="picard", max_iter="auto", random_state=97
    )
    ica.fit(eeg_data)
    all_explained_var_ratio = ica.get_explained_variance_ratio(eeg_data, ch_type="eeg")
    if not nbr_ics:
        nbr_ics = len(get_good_eeg_chan(eeg_data))
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
    eog_indices = []
    if "eog" in eeg_data.get_channel_types():
        ica.exclude = []
        eog_indices, eog_scores = ica.find_bads_eog(eeg_data)  # detect eog ics

    # ICLabel classification of the IC components:
    ic_labels = mne_icalabel.label_components(eeg_data, ica, method="iclabel")
    labels = ic_labels["labels"]

    if "eog" in eeg_data.get_channel_types() and (ica.labels_["eog"] != eog_indices):
        raise ValueError("EOG ICs found by ICLabel and MNE 'find_bads_eog' are different")
    ica_logs = {"ics_kept": 0, "var_kept": 0}
    exclude_dict = {}
    for idx, label in enumerate(labels):
        if label not in ["brain", "other"] and ic_labels["y_pred_proba"][idx] > proba_thresh:
            exclude_dict[idx] = {
                "label": label,
                "proba": ic_labels["y_pred_proba"][idx],
            }
        else:
            ica_logs["ics_kept"] += 1
    print(f"Excluding these ICA components:")
    for idx, val in exclude_dict.items():
        round_proba = round(int(100 * val["proba"]), 2)
        print(f"Component n°{idx + 1} '{val['label']}' (probability: {round_proba}%)")
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
            ica_logs["var_kept"] += round_var
            print(
                "Component n°{} '{}' ({}%): {:.1f}%".format(i + 1, ic, round_proba, round_var)
            )  # f"{}" cannot round
    # ica.plot_properties(raw_eeg, picks=kept_ics, verbose=False)
    ica_logs["var_kept"] = round(ica_logs["var_kept"], 1)
    ica.apply(eeg_data, exclude=list(exclude_dict.keys()))
    if return_sources:
        ica.exclude.extend(minor_brain_ics)  # only keep the main sources to work on them
        if Return_log:
            return ica.get_sources(eeg_data), ica_logs
        return ica.get_sources(eeg_data)
    if Return_log:
        return eeg_data, ica_logs
    return eeg_data


def offline_preprocess(
    subject,
    experiment,
    data_repo="../data",
    l_freq=0.1,
    h_freq=50,
    epoch_duration=5,
    epoch_baseline=(-1.5,0),
    sfreq=512,
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
    :param epoch_duration:
    :param epoch_baseline:
    :param sfreq:
    :param work_on_sources:
    :param bad_trials:
    :param save_prepro_repo:
    :return: epochs:
    """
    freq_band = {
        "delta": [1, 4],
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30],
        "gamma": [30, 50],
    }
    file = f"{data_repo}/{subject}_{experiment}/{subject}_{experiment}.fif"
    events_csv = f"{data_repo}/{subject}_{experiment}/{subject}_{experiment}.events.csv"
    me_event_id, mi_event_id = get_cond_id(experiment)
    cond_dict = {me_event_id: "me", mi_event_id: "mi"}
    raw_eeg = mne.io.read_raw_fif(file)
    raw_eeg.resample(sfreq=sfreq)
    raw_eeg.load_data()  # need to load data to rereference etc
    # raw_eeg.plot()

    # Set channel types
    raw_eeg = fix_montage_eeg(raw_eeg)
    # fig = raw_eeg.plot_sensors(show_names=True)
    # fig.show()

    # Filter
    # raw_eeg.notch_filter(50)
    raw_eeg.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

    # Find bad/missing channels and interpolate them
    raw_eeg.info["bads"], chan_drop_log = detect_badChan(
        raw_eeg, l_freq, h_freq, keepEOG=True, Return_log=True
    )

    # Annotate according to events.csv markers time stamps
    raw_eeg, events = annotate_by_markers(raw_eeg, subject, events_csv, me_event_id, mi_event_id)

    # Make epochs of 5 seconds for the paradigm motor execution (me) and motor imagery (mi)
    for event_id in [me_event_id, mi_event_id]:
        list_bad_trials = bad_trials[f"{subject}_{experiment}"][cond_dict[event_id]]
        epochs = get_cond_epochs(
            raw_eeg,
            events,
            event_id,
            detrend=1,
            epoch_tmin=epoch_baseline[0],
            epoch_tmax=epoch_duration,
            bad_trials=list_bad_trials,
        )
        epochs.load_data()
        epochs.set_eeg_reference(ref_channels=['M1', 'M2'])
        if epochs.info["bads"]:
            epochs = epochs.interpolate_bads(reset_bads=True)

        topo_path = None
        if save_prepro_repo and work_on_sources:
            try:
                Path(f"{save_prepro_repo}/ics_sources/").mkdir(parents=True, exist_ok=True)
            except FileExistsError:
                pass
            topo_path = (
                f"{save_prepro_repo}/ics_sources/sources_topo_{subject}_{experiment}_"
                f"{cond_dict[event_id]}_filt({l_freq}_{h_freq})_basl{epoch_baseline}_.png"
            )
        epochs = perform_ica(epochs, proba_thresh=0.5, return_sources=work_on_sources, topo_saveas=topo_path)

        # Baseline removal
        epochs.apply_baseline(epoch_baseline)
        # epochs.plot()
        # epochs.plot_psd(0, 50)

        epochs = reject_badEpoch(
            epochs,
            fmin=1,
            fmax=h_freq,
            method=["potato"],
        )
        epochs = reject_badEpoch(
            epochs,
            fmin=1,
            fmax=h_freq,
            method=["psd"],
            mode="conservative",
        )
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
                f"{l_freq}_{h_freq})_basl{epoch_baseline}.fif"
            )
            erptopo_path = (
                f"{prepro_repo}{prefix}_{subject}_{experiment}_{cond_dict[event_id]}_filt("
                f"{l_freq}_{h_freq})_basl{epoch_baseline}.png"
            )
            topo_path = (
                f"{prepro_repo}{prefix}_topo_{subject}_{experiment}_{cond_dict[event_id]}_filt"
                f"({l_freq}_{h_freq})_basl{epoch_baseline}.png"
            )
            psd_topo_path = (
                f"{prepro_repo}{prefix}_psd_topo_{subject}_{experiment}_{cond_dict[event_id]}_filt"
                f"({l_freq}_{h_freq})_basl{epoch_baseline}.png"
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
                psd = epochs.compute_psd(method="multitaper", fmin=1, fmax=h_freq)
                fig = psd.plot_topomap(
                    freq_band, show=True
                )  # cmap='Spectral_r' vlim='joint' sphere=head_size
                fig.savefig(psd_topo_path, format="png", dpi=200)
                plt.close(fig)
            epochs.save(prepro_file_path, overwrite=True)
            print(f"file {prepro_file_path} along its evoked erp saved")

    return epochs
