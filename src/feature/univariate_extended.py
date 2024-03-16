"""
TODO Futures Features:
- Spatial Representation (paper+code DeepForest) => It is more of a data Transformer. Compute psd or others then put it
in topo_rpz.
=> For spatial rpz => better to use it as a Transformer directly in the classification pipeline. Because it will always
be the final feature. FeatureExtractor(sfreq=sfreq, selected_funcs=selected_funcs)
"""
import numpy as np
from scipy import stats as st
from mne_features.utils import _get_feature_funcs, _get_feature_func_names


def get_univariate_funcs_extended(sfreq):
    """Mapping between aliases and univariate feature functions.
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
    Returns
    -------
    univariate_funcs : dict
    """
    return _get_feature_funcs(sfreq, __name__)


def get_univariate_func_names_extended():
    """List of names of univariate feature functions.
    Returns
    -------
    univariate_func_names : list
    """
    return _get_feature_func_names(__name__)


def compute_raw(data):
    """return the raw data per channels.
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **raw**
    """
    return data


def compute_min(data):
    """Min of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **min**
    """
    return np.min(data, axis=-1)


def compute_max(data):
    """Max of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **max**
    """
    return np.max(data, axis=-1)


def compute_median(data):
    """Median of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **median**
    """
    return np.median(data, axis=-1)


def compute_mode(data):
    """Mode of the data (per channel). Mode refers to the most repeating element in the array.
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **mode**
    """
    return st.mode(data, axis=-1)


def compute_diff_entropy(data):
    """Differential Entropy of the data (per channel).
    Parameters
    ----------
    data : ndarray, shape (n_channels, n_times)
    Returns
    -------
    output : ndarray, shape (n_channels,)
    Notes
    -----
    Alias of the feature function: **diff_entropy**
    """
    return 1 / 2 * np.log2(2 * np.pi * np.e * np.std(data, axis=-1))
