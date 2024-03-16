""""""
from typing import Dict, Tuple
import numpy as np
from utils.preprocessing_utils import butter_bandpass, lfilter
from mne.time_frequency import (
    psd_array_welch,
    psd_array_multitaper,
    tfr_array_multitaper,
    tfr_array_morlet,
    tfr_array_stockwell,
)

EEG_FREQUENCY_BANDS = {
    "delta": [1, 4],
    "theta": [4, 8],
    "alpha": [8, 14],
    "beta": [14, 31],
    "gamma": [31, 49],
}


def get_freq_bands(
    data: np.ndarray,
    fs: int = 128,
    order: int = 5,
    band_dict: Dict[str, Tuple[int, int]] = EEG_FREQUENCY_BANDS,
):
    """
    :param data:
    :param fs:
    :param order:
    :param band_dict:
    :return:
    """
    band_list = []
    for low, high in band_dict.values():
        c_list = []
        for c in data:
            b, a = butter_bandpass(low, high, fs=fs, order=order)
            c_list.append(lfilter(b, a, c))
        c_list = np.array(c_list)
        band_list.append(c_list)
    return np.stack(band_list, axis=-1)


"""
- PSD, FFT, STFT, DFT, WT => add the possibility to have directly the spectrum (power*freq) and the
periodogram (estimated power on windows of time*freq). Have timefreqchan but with option on windows, overlap, method,
log, normalization (like mne-feature) ==> use power_spectrum from mne-feature utils. (welch, fft, multitaper).
"""
# compute_pow_spectr():
# mne.time_frequency.psd_array_welch
# => # https://mne.tools/stable/auto_tutorials/time-freq/20_sensors_time_frequency.html
# psds = 10 * np.log10(psds)


def get_spectrum(
    sfreq,
    data,
    fmin=0.0,
    fmax=256.0,
    psd_method="welch",
    welch_n_fft=256,
    welch_n_per_seg=None,
    welch_n_overlap=0,
    verbose=False,
):
    """Spectrum.
    The multitaper method, although more computationally intensive than Welch's method or FFT, should be preferred for
    'short' windows. Welch's method is more suitable for 'long' windows.
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
    data : ndarray, shape (..., n_times).
    fmin : float (default: 0.)
        Lower bound of the frequency range to consider.
    fmax : float (default: 256.)
        Upper bound of the frequency range to consider.
    psd_method : str (default: 'welch')
        Method used to estimate the PSD from the data. The valid values for
        the parameter ``method`` are: ``'welch'``, ``'fft'`` or
        ``'multitaper'``.
    welch_n_fft : int (default: 256)
        The length of the FFT used. The segments will be zero-padded if
        `welch_n_fft > welch_n_per_seg`. This parameter will be ignored if
        `method = 'fft'` or `method = 'multitaper'`.
    welch_n_per_seg : int or None (default: None)
        Length of each Welch segment (windowed with a Hamming window). If
        None, `welch_n_per_seg` is equal to `welch_n_fft`. This parameter
        will be ignored if `method = 'fft'` or `method = 'multitaper'`.
    welch_n_overlap : int (default: 0)
        The number of points of overlap between segments. Should be
        `<= welch_n_per_seg`. This parameter will be ignored if
        `method = 'fft'` or `method = 'multitaper'`.
    verbose : bool (default: False)
        Verbosity parameter. If True, info and warnings related to
        :func:`mne.time_frequency.psd_array_welch` or
        :func:`mne.time_frequency.psd_array_multitaper` are printed.
    Returns
    -------
    psd : ndarray, shape (..., n_freqs)
        Estimated PSD.
    freqs : ndarray, shape (n_freqs,)
        Array of frequency bins.
    """
    _verbose = 40 * (1 - int(verbose))
    _fmin, _fmax = max(0, fmin), min(fmax, sfreq / 2)
    if psd_method == "welch":
        _n_fft = min(data.shape[-1], welch_n_fft)
        return psd_array_welch(
            data,
            sfreq,
            fmin=_fmin,
            fmax=_fmax,
            n_fft=_n_fft,
            average=None,
            window="hamming",
            n_per_seg=welch_n_per_seg,
            n_overlap=welch_n_overlap,
            verbose=_verbose,
        )
    elif psd_method == "fft":
        n_times = data.shape[-1]
        m = np.mean(data, axis=-1)
        _data = data - m[..., None]
        spect = np.fft.rfft(_data, n_times)
        mag = np.abs(spect)
        freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
        psd = np.power(mag, 2) / (n_times**2)
        psd *= 2.0
        psd[..., 0] /= 2.0
        if n_times % 2 == 0:
            psd[..., -1] /= 2.0
        mask = np.logical_and(freqs >= _fmin, freqs <= _fmax)
        return psd[..., mask], freqs[mask]
    else:
        raise ValueError(
            "The given method (%s) is not implemented. Valid "
            "methods for the computation of the PSD are: "
            "`welch`, `fft` or `multitaper`." % str(psd_method)
        )


def get_tfr(
    data,
    sfreq,
    tfr_method="multitaper",
    freqs=np.arange(2, 40),
    n_cycles=7.0,
    time_bandwidth=4.0,
    width=1.0,
    decim=1,
    output="power",
):
    """
    Compute Time-Frequency Representation

    Feature ERSP/ERDS: https://mne.tools/stable/auto_examples/time_frequency/time_frequency_erds.html
    https://mne.tools/stable/generated/mne.time_frequency.tfr_multitaper.html#mne.time_frequency.tfr_multitaper
    https://github.com/cbrnr/bci_event_2021
    https://mne.tools/stable/auto_examples/time_frequency/time_frequency_simulated.html

    :param data:
    :param sfreq:
    :param tfr_method:
    :param freqs:
    :param n_cycles:
    :param time_bandwidth:
    :param width:
    :param decim:
    :param output:
    :return:
    """
    if tfr_method == "multitaper":
        tfr = tfr_array_multitaper(
            data, sfreq, freqs, n_cycles, time_bandwidth=time_bandwidth, decim=decim, output=output
        )
    elif tfr_method == "morlet":
        tfr = tfr_array_morlet(data, sfreq, freqs, n_cycles, decim=decim, output=output)
    elif tfr_method == "stockwell":
        tfr = tfr_array_stockwell(data, sfreq, width=width, decim=decim)
    return tfr
