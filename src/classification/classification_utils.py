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


def get_df_results_avg(df):
    """
    Cheesy way to compute and add columns of various averaging of cross-validation results taken from the input
    dataframe.
    See link below for Cheesy vs classic method:
    https://stackoverflow.com/questions/43223615/join-dataframes-one-with-multiindex-columns-and-the-other-without
    :param df_results:
    :param avg_col_names:
    :param level_names:
    :return:
    """
    # 1) if df_results is not a multiindex => just have to df['Avg total'] = df.mean(axis=1) then df.join(df2)
    # if is multiindex but check show that only 1 subj or 1 cond => do more or less like 1)
    # if df_result is multiindex with multiple subj and cond: need an average of every X col cond per different level 2
    # (subj) and an average or every col of each level 2 (avg by subj). join/concat all avg col each time before join
    # it to the principal dtf. Then do for the next avg. Then average all col of original dtf for average all (like 1)
    # more or less)
    # replace avg name by 'avg_{level_name}'. Make the fct as it can only have Subj, Cond => check that level name size
    # is not superior to 2
    # We only want a total average if there are not multiple subjects and conditions
    # if want to rename avg_total sub col: https://stackoverflow.com/questions/41221079/rename-multiindex-columns-in-pandas
    df_with_avg = df.copy()
    df_with_avg["Avg_total"] = df.mean(axis=1)  # axis = 1 or 0
    if isinstance(df.index, pd.MultiIndex):
        if all(len(df.columns.levels[i]) > 1 for i in range(len(df.columns.levels))):
            for level in range(len(df.columns.levels)):
                df_avg = df.T.groupby(level=level, sort=False).mean().T
                df_with_avg = pd.concat(
                    [df_avg], axis=1, keys=[f"Avg_{df.columns.names[level]}"]
                ).join(df_with_avg)  # concat so multiindex column are automaticaly put as first column
    df_with_avg.insert(0, "Avg_total", df_with_avg.pop("Avg_total"))  # make Avg_total the first column

    return df_with_avg.round(3)


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
