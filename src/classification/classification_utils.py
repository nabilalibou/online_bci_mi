"""
"""
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.utils import shuffle


# from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
# from ml_test.utils.math_utils import flatten3d, cartesian_product
# from ml_test.utils.data_utils import (
#     add_prefix_to_list,
#     add_suffix_to_list,
#     clean_list_of_str,
#     shuffle_X_lists,
# )
# from ml_test.classification.NN_models import reset_weights


def get_df_results_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average of each column and adds it as a new column named 'Avg_total' at the front of the DataFrame.
    Also adds a MultiIndex column with level-wise averages for DataFrames with MultiIndex columns.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: A new DataFrame with the calculated averages.
    Raises:
        ValueError: If the DataFrame has no columns.
    """
    if len(df.columns) == 0:
        raise ValueError("Cannot calculate average of an empty DataFrame")

    df_with_avg = df.copy()
    df_with_avg["Avg_total"] = df.mean(axis=1)  # axis = 1 or 0

    if isinstance(df.columns, pd.MultiIndex):
        if all(len(df.columns.levels[i]) > 1 for i in range(len(df.columns.levels))):
            for level in range(len(df.columns.levels)):
                df_avg = df.T.groupby(level=level, sort=False).mean().T
                df_with_avg = pd.concat(
                    [df_avg], axis=1, keys=[f"Avg_{df.columns.names[level]}"]
                ).join(
                    df_with_avg
                )  # concat so multiindex column are automatically put as first column

    df_with_avg.insert(0, "Avg_total", df_with_avg.pop("Avg_total"))  # make Avg_total the first column

    return df_with_avg.astype(float).round(3)


def evaluate(X: np.ndarray, y: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray,
             clf_dict: dict, score_dict: dict, nbr_runs: int = 1) -> np.ndarray:
    """
    Evaluates a dictionary of classifiers (`clf_dict`) using a dictionary of scoring functions (`score_dict`)
    on a training set (`X`, `y`) and an evaluation set (`X_eval`, `y_eval`).

    Args:
        X (np.ndarray): The training data features.
        y (np.ndarray): The training data target labels.
        X_eval (np.ndarray): The evaluation data features.
        y_eval (np.ndarray): The evaluation data target labels.
        clf_dict (dict): A dictionary containing classifier objects with names as keys.
        score_dict (dict): A dictionary containing scoring functions with names as keys.
        nbr_runs (int, optional): The number of times to repeat the evaluation for each
                                  classifier-scorer combination (default: 1).
    Returns:
        np.ndarray: A 2D array (vertical vector) containing the average scores for each classifier-scorer combination
                    after `nbr_runs` evaluations.
    Raises:
        Exception: If an error occurs while fitting a classifier in the pipeline.
    """
    vect_result = np.zeros([len(clf_dict.keys()) * len(score_dict.keys())])
    cnt_clf = 0
    for clf_name, clf_value in clf_dict.items():
        print(f"=> Currently fitting Pipeline '{clf_name}'")
        for run in range(0, nbr_runs):
            X_, y_ = shuffle(X, y)
            try:
                clf_value.fit(X_, y_)
            except Exception as e:
                raise Exception(f"\nError when going through the pipeline '{clf_name}'. {e}")
            y_pred = clf_value.predict(X_eval)
            cnt_score = 0
            for score_name, scorer in score_dict.items():
                index = cnt_clf * len(score_dict.keys()) + cnt_score
                vect_result[index] += np.mean(scorer(y_eval, y_pred))
                cnt_score += 1
        cnt_clf += 1
    vect_result /= nbr_runs

    return vect_result.round(3).reshape((-1, 1))


def evaluate_conditions(X: np.ndarray, y: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray,
                        clf_dict: dict, nbr_runs: int = 1) -> np.ndarray:
    """
    Evaluates a dictionary of classifiers (`clf_dict`) on a training set (`X`, `y`) and an
    evaluation set (`X_eval`, `y_eval`), focusing on performance for each unique class label (condition).

    Args:
        X (np.ndarray): The training data features.
        y (np.ndarray): The training data target labels.
        X_eval (np.ndarray): The evaluation data features.
        y_eval (np.ndarray): The evaluation data target labels.
        clf_dict (dict): A dictionary containing classifier objects with names as keys.
        nbr_runs (int, optional): The number of times to repeat the evaluation for each
                                  classifier (default: 1).
    Returns:
        np.ndarray: A 2D array of shape (n_classifiers, n_conditions) where each entry
                    represents the average proportion of correctly classified instances
                    per condition (class label) across `nbr_runs` evaluations for each classifier.
    Raises:
        Exception: If an error occurs while fitting a classifier in the pipeline.
    """
    conditions = np.unique(y_eval)
    vect_result = np.zeros((len(clf_dict.keys()), len(conditions)), dtype=np.float16)
    cond_result = np.zeros((1, len(conditions)), dtype=np.float16)
    cnt_clf = 0
    for i, (clf_name, clf_value) in enumerate(clf_dict.items()):
        print(f"=> Currently fitting Pipeline '{clf_name}'")
        for run in range(0, nbr_runs):
            X_, y_ = shuffle(X, y)
            try:
                clf_value.fit(X_, y_)
            except Exception as e:
                raise Exception(f"\nError when going through the pipeline '{clf_name}'. {e}")
            y_pred = clf_value.predict(X_eval)
            for j, cond in enumerate(list(conditions)):
                mask = np.where(y_eval == cond)[0]
                cond_result[:, j] = np.sum(y_pred[mask] == cond) / len(mask)
            if not i:
                vect_result = cond_result
            else:
                vect_result = np.vstack((vect_result, cond_result))
        cnt_clf += 1
    vect_result /= nbr_runs

    return vect_result
