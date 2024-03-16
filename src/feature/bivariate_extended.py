""""""
from mne_features.utils import _get_feature_funcs, _get_feature_func_names


def get_bivariate_funcs_extended(sfreq):
    """Mapping between aliases and bivariate feature functions.
    Parameters
    ----------
    sfreq : float
        Sampling rate of the data.
    Returns
    -------
    bivariate_funcs : dict
    """
    return _get_feature_funcs(sfreq, __name__)


def get_bivariate_func_names_extended():
    """List of names of bivariate feature functions.
    Returns
    -------
    bivariate_func_names : list
    """
    return _get_feature_func_names(__name__)


# def compute_dasm(data):
#     """Mode of the data (per channel). compute Differential Asymmetry
#     Parameters
#     ----------
#     data : ndarray, shape (n_channels, n_times)
#     Returns
#     -------
#     output : ndarray, shape (n_channels,)
#     Notes
#     -----
#     Alias of the feature function: **mode**
#     """
#
#     if asymmetry == "rational":
#         if not np.all(data):
#             raise Warning("data contains 0")
#             print("data contains 0, cannot compute rational data, replace by differential asymmetry instead")
#             asymmetry = "differential"
#         # left feature / right feature
#     elif asymmetry == "differential":
#         print("")
#         # left feature - right feature
#     if asymmetry:
#         for ch_pair in ch_pairs:
#             print("feature_matrix[] = feature_matrix[] - feature_matrix[]")
#
#     return st.mode(data, axis=-1)
#
#
# def compute_rasm(data):
#     """Mode of the data (per channel). Rational Asymmetry
#     Parameters
#     ----------
#     data : ndarray, shape (n_channels, n_times)
#     Returns
#     -------
#     output : ndarray, shape (n_channels,)
#     Notes
#     -----
#     Alias of the feature function: **mode**
#     """
#     return st.mode(data, axis=-1)
