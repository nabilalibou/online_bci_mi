"""
"""
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, RepeatedKFold, RepeatedStratifiedKFold
from ml_test.utils.math_utils import flatten3d, cartesian_product
from ml_test.utils.data_utils import (
    add_prefix_to_list,
    add_suffix_to_list,
    clean_list_of_str,
    shuffle_X_lists,
)
from ml_test.classification.NN_models import reset_weights


def avg_df(df_results, avg_col_names, l_levels_avg_by):
    """
    Cheesy way to compute and add columns of various averaging of cross-validation results taken from the input
    dataframe.
    See link below for Cheesy vs classic method:
    https://stackoverflow.com/questions/43223615/join-dataframes-one-with-multiindex-columns-and-the-other-without

    Parameters
    ----------
    df_results : DATAFRAME
        Results of classifications.
    avg_col_names : LIST of TUPLES
        Names of multiindex columns.
    l_levels_avg_by : LIST of INT
        Number of levels to get average BY !!

    Returns
    -------
    df_results_with_avg : DATAFRAME
        df_results with averages.

    """
    df_results_with_avg = df_results.copy()
    # Concatenating df_results columns level(s) averages (associating multiindex names (keys)) with df_results
    for n_avg in l_levels_avg_by:
        df_results_with_avg = pd.concat([df_results.groupby(axis=1, level=n_avg, sort=False).mean()],
                                        axis=1, keys=[avg_col_names[0]]).join(df_results_with_avg)
        avg_col_names.pop(0)
    if df_results.columns.nlevels > 1:
        df_results_with_avg = pd.concat([df_results.mean(axis=1)], axis=1,
                                        keys=[avg_col_names[0] + ('All',)]).join(df_results_with_avg)
    else:
        df_results_with_avg = pd.concat([df_results.mean(axis=1).to_frame(avg_col_names[0])],
                                        axis=1).join(df_results_with_avg)
    df_results_with_avg.columns.names = df_results.columns.names

    return df_results_with_avg


def avg_crossval_df(df_results, avg_col_names, l_levels_avg_by):
    """

    Parameters
    ----------
    df_results : DATAFRAME
        Results of classifications.
    avg_col_names : LIST of TUPLES
        Names of multiindex columns.
    l_levels_avg_by : LIST of INT
        Number of levels to get average BY !!

    Returns
    -------
    df_results_with_avg : DATAFRAME
        df_results with averages, rounded to third decimal.

    """
    new_avg_col_names = avg_col_names.copy()
    # Making tuples names for more than one column levels
    if df_results.columns.nlevels > 1:
        new_avg_col_names = [tuple([e]) for e in new_avg_col_names]
        for n in range(0, df_results.columns.nlevels - 2):
            new_avg_col_names = [tuple(['Avg'] + list(name)) for name in new_avg_col_names]
    df_results_with_avg = avg_df(df_results, new_avg_col_names, l_levels_avg_by)

    return df_results_with_avg.round(3)


def evaluate(X, y, X_eval, y_eval, clf_dict, score_dict, nbr_runs=1):

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
