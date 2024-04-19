""" Feature extraction functions """
from typing import Dict, Tuple
import warnings
import numpy as np
from collections import defaultdict
try:
    from sklearn.externals import joblib
except (ImportError, ModuleNotFoundError):
    import joblib
from mne_features.bivariate import get_bivariate_funcs
from mne_features.univariate import get_univariate_funcs
from mne_features.feature_extraction import extract_features
from feature.univariate_extended import get_univariate_funcs_extended
from feature.bivariate_extended import get_bivariate_funcs_extended


def _check_all_funcs(selected, feature_funcs, feature_funcs_extended=None):
    """
    Checks if selected features are valid function names or user-defined functions.

    Parameters:
    - selected (list): List of selected feature function names or tuples.
    - feature_funcs (dict): Dictionary of available MNE-Features functions.
    - feature_funcs_extended (dict, optional): Dictionary of available custom functions (if applicable).

    Returns:
    - valid_funcs (list): List of valid function names or tuples (user-defined functions).

    Raises:
    - ValueError: If an invalid function name, tuple format, or alias conflict is encountered.
    """
    valid_funcs = []
    mne_func_names = set(feature_funcs.keys())
    extended_func_names = set() if feature_funcs_extended is None else set(feature_funcs_extended.keys())
    all_func_names = mne_func_names.union(extended_func_names)

    for item in selected:
        if isinstance(item, str):
            # Check for MNE-Features function or extended function with matching alias
            if item in mne_func_names:
                valid_funcs.append(item)
            elif feature_funcs_extended and item in extended_func_names:
                valid_funcs.append((item, feature_funcs_extended[item]))
            else:
                raise ValueError(
                    f"Invalid function name: {item}. Valid names include:\n"
                    f"- MNE-Features: {', '.join(mne_func_names)}\n"
                    f"{(lambda d: ', '.join(d) if d else 'No custom functions provided')(extended_func_names)}"
                )
        elif isinstance(item, tuple) and len(item) == 2:
            # Check for user-defined function format and alias conflict
            alias, func = item
            if alias in all_func_names:
                raise ValueError(f"Alias conflict: {alias} already used by a built-in function.")
            valid_funcs.append(item)
        else:
            raise ValueError(f"Invalid feature function format: {item}")

    if not valid_funcs:
        raise ValueError("No valid feature functions provided.")

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
    Extracts features from MNE data using MNE-Features and custom univariate/bivariate functions.
    This function simplifies feature extraction by combining built-in MNE-Features functions
    with user-defined feature functions for univariate and bivariate analysis.

    Parameters:
    - data (mne.io.BaseRaw | mne.Epochs): The MNE data object.
    - features_list (list): A list of feature names (from MNE-Features or custom functions).
    - sfreq (float): The sampling frequency of the data (Hz).
    - funcs_params (dict, optional): A dictionary containing parameters for custom functions.
    - n_jobs (int): The number of parallel jobs to use (default: 1).
    - ch_names (list, optional): A list of channel names to use (default: all channels).
    - return_as_df (bool): Whether to return the features as a DataFrame (default: False).
    - keep_original_shape (bool): Whether to reshape the output to match data shape (default: True).

    Returns:
    - feature_matrix (np.ndarray | pd.DataFrame): The extracted feature matrix.

    Raises:
    - Warning: If reshaping fails due to bivariate features.
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
            # Reshape based on data dimensions, handle potential errors
            feature_matrix = feature_matrix.reshape(data.shape[0], data.shape[1], -1)
        except ValueError:
            message = (
                f"Reshaping failed for feature matrix ({np.shape(feature_matrix)}). "
                f"This likely occurs due to bivariate features presents among input features."
            )
            warnings.warn(message)

    return feature_matrix
