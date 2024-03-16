""" Feature extraction functions """
from typing import Dict, Tuple
import numpy as np

try:
    from sklearn.externals import joblib
except (ImportError, ModuleNotFoundError):
    import joblib
from mne_features.bivariate import get_bivariate_funcs
from mne_features.univariate import get_univariate_funcs
from mne_features.feature_extraction import extract_features
from feature.univariate_extended import get_univariate_funcs_extended
from feature.bivariate_extended import get_bivariate_funcs_extended


def _check_all_funcs(selected, feature_funcs, feature_funcs_extended):
    """Selection checker.
    Checks if the elements of ``selected`` are either strings (alias of a
    feature function defined in mne-features) or tuples of the form
    ``(str, callable)`` (user-defined feature function).
    Parameters
    ----------
    selected : list of str or tuples
        Names of the selected feature functions.
    feature_funcs : dict
        Dictionary of the feature functions (univariate and bivariate)
        available in mne-features.
    Returns
    -------
    valid_funcs : list of tuples
    """

    valid_funcs = list()
    _intrinsic_mne_func_names = feature_funcs.keys()  # MNE-feature alias
    _intrinsic_func_names_extended = feature_funcs_extended.keys()  # NeuroKyma features alias
    all_func_names = {**feature_funcs, **feature_funcs_extended}.keys()
    for s in selected:
        if isinstance(s, str):
            if s in _intrinsic_mne_func_names:
                valid_funcs.append(s)
            elif s in _intrinsic_func_names_extended:
                valid_funcs.append((s, feature_funcs_extended[s]))
            else:
                raise ValueError(
                    f"The given alias {s} is not valid. The valid aliases for feature functions are "
                    f"the ones from: \nmne-features: {_intrinsic_mne_func_names} \n NeuroKyma "
                    f"features: {_intrinsic_func_names_extended}."
                )
        elif isinstance(s, tuple):
            if len(s) != 2:
                raise ValueError(
                    "The given tuple (%s) is not of length 2. "
                    "Each user-defined feature function should "
                    "be passed as a tuple of the form "
                    "`(str, callable)`." % str(s)
                )
            else:
                # Case of a user-defined feature function
                if s[0] in all_func_names:
                    raise ValueError(
                        f"A user-defined feature function was given an alias {s[0]} which is already "
                        "used by mne-features or NeuroKyma features. \n The list of aliases used by "
                        f"mne-features is: {_intrinsic_mne_func_names}. \n The list of aliases of the "
                        f"NeuroKyma features is: {_intrinsic_func_names_extended}."
                    )
                else:
                    valid_funcs.append(s)
        else:
            # Case where the element is neither a string, nor a tuple
            raise ValueError(
                "%s is not a valid feature function and cannot "
                "be interpreted as a user-defined feature "
                "function." % str(s)
            )
    if not valid_funcs:
        raise ValueError("No valid feature function was given.")
    else:
        return valid_funcs


def get_features(
    data,
    features_list,
    sfreq=128,
    funcs_params=None,
    n_jobs=1,
    ch_names=None,
    return_as_df=False,
    keep_original_shape=True,
):
    """
    Wrapper of mne_features.extract_features() function extended with features from feature.univariate_extended and
    feature.bivariate_extended.
    MNE-Features official documentation: https://mne.tools/mne-features/index.html
    :param data:
    :param sfreq:
    :param features_list:
    :param funcs_params:
    :param n_jobs:
    :param ch_names:
    :param return_as_df:
    :param keep_original_shape:
    :return:
    """

    univariate_funcs = get_univariate_funcs(sfreq)
    bivariate_funcs = get_bivariate_funcs(sfreq)
    feature_funcs = univariate_funcs.copy()
    feature_funcs.update(bivariate_funcs)

    univariate_funcs_extended = get_univariate_funcs_extended(sfreq)
    bivariate_funcs_extended = get_bivariate_funcs_extended(sfreq)
    feature_funcs_extended = univariate_funcs_extended.copy()
    feature_funcs_extended.update(bivariate_funcs_extended)

    sel_funcs = _check_all_funcs(features_list, feature_funcs, feature_funcs_extended)

    feature_matrix = extract_features(
        data, sfreq, sel_funcs, funcs_params, n_jobs, ch_names, return_as_df
    )

    if keep_original_shape:
        try:
            feature_matrix = feature_matrix.reshape(data.shape[0], data.shape[1], -1)
        except ValueError:
            raise Warning(
                f"Feature matrix with shape {np.shape(feature_matrix)} could not be reshaped according to "
                f"the input matrix original shape {np.shape(data)}. This may be because of the presence of"
                f"bivariate features among the selected features."
            )

    return feature_matrix
