"""
"""
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
from utils.data_utils import add_prefix_to_list
from utils.result_utils import save_classif_report

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


def load_subject_data(data_dict, subject_name):
    """
    Loads data for a given subject from a dictionary.
    Args:
        data_dict (dict): A dictionary containing data for multiple subjects, where each key is a subject name and
            the value is a dictionary containing data for that subject, organized by condition names.
        subject_name (str): The name of the subject to load data for.
    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): The features for the subject.
            - Y (np.ndarray): The labels for the subject.
            - cond_names (list[str]): A list of condition names associated with the loaded data.
    Raises:
        ValueError: If the specified subject name is not found in the data dictionary.
    """
    subject_data = data_dict.get(subject_name)
    cond_names = []
    if subject_data is None:
        raise ValueError(f"Subject '{subject_name}' not found in data dictionary")
    X, Y = None, None
    for cond_num, (cond_name, data) in enumerate(subject_data.items()):
        if not cond_num:
            X = data["X"]
            Y = data["Y"]
        else:
            X = np.vstack((X, data["X"]))
            Y = np.hstack((Y, data["Y"]))
        cond_names.append(cond_name)
    return X, Y, cond_names


def evaluate_intra_subject(
    data_dict: dict,
    n_splits: int,
    clf_dict: dict,
    score_dict: dict,
    nbr_runs: int,
    report_path: str,
) -> None:
    """
    Performs intra-subject evaluation using Stratified K-Fold cross-validation.
    This function evaluates the performance of multiple classifiers (defined in `clf_dict`)
    on a single subject's data using K-Fold cross-validation with a specified number of splits (`n_splits`).
    It computes various scores (defined in `score_dict`) for each classifier and fold,
    and reports the mean score across folds. It also evaluates the performance on each condition
    within the subject's data (if present).
    Args:
        data_dict (dict): A dictionary containing data for multiple subjects. Each key is a subject name,
            and the corresponding value is a dictionary containing data for that subject, organized by condition names.
            The expected structure within the subject's data dictionary is:
            ```
            {
                'condition_name_1': {'X': features_array_1, 'Y': labels_array_1},
                'condition_name_2': {'X': features_array_2, 'Y': labels_array_2},
                ...
            }
            ```
        n_splits (int): The number of folds to use for K-Fold cross-validation.
        clf_dict (dict): A dictionary containing machine learning classifiers to evaluate.
            The key is the classifier name, and the value is the classifier object itself.
        score_dict (dict): A dictionary containing scoring functions to use.
            The key is the score name, and the value is the scoring function itself.
        nbr_runs (int): The number of times to repeat the evaluation for each fold (for potential stochasticity).
        report_path (str): The path to save the evaluation report (as two CSV files).
    Returns:
        None
    """
    index_names = [list(clf_dict.keys()), list(score_dict.keys())]
    level_names = ["Subjects", "Folds"]
    cond_level_names = ["Subjects", "Conditions"]
    col_names = [[], []]
    cond_col_names = [[], []]
    array_result_tot = []
    cond_array_result_tot = []
    for subj_num, (subj_name, subj_data) in enumerate(data_dict.items()):
        # kf = KFold(n_splits=n_splits, shuffle=True)
        kf = StratifiedShuffleSplit(n_splits=n_splits)
        col_names[1] = [f"Fold_{i + 1}" for i in range(n_splits)]
        array_result = []
        vect_result_cond_sum = None
        X, Y, cond_col_names[1] = load_subject_data(data_dict, subj_name)
        for train_index, eval_index in kf.split(
            X,
            Y,
        ):
            X_eval, y_eval = X[eval_index], Y[eval_index]
            X_train, y_train = X[train_index], Y[train_index]
            vect_result = evaluate(
                X_train, y_train, X_eval, y_eval, clf_dict, score_dict, nbr_runs=nbr_runs
            )
            vect_result_cond = evaluate_conditions(
                X_train, y_train, X_eval, y_eval, clf_dict, nbr_runs=nbr_runs
            )
            if vect_result_cond_sum is None:
                vect_result_cond_sum = vect_result_cond
            else:
                vect_result_cond_sum = np.add(vect_result_cond_sum, vect_result_cond)
            if len(array_result):
                array_result = np.hstack((array_result, vect_result))
            else:
                array_result = vect_result
        col_names[0].append(subj_name)
        vect_result_cond_sum /= n_splits
        if len(array_result_tot):
            array_result_tot = np.hstack((array_result_tot, array_result))
        else:
            array_result_tot = array_result
        if len(cond_array_result_tot):
            cond_array_result_tot = np.hstack((cond_array_result_tot, vect_result_cond_sum))
        else:
            cond_array_result_tot = vect_result_cond_sum
        print(f"test on subject '{subj_name}' done")
        print(f"Mean Result = {np.mean(array_result_tot).round(3)}")

    # Create the report DataFrames and save them
    col_names[1] = add_prefix_to_list(col_names[1], "Fold_")
    col_names[0] = add_prefix_to_list(col_names[0], "Subj_")
    cond_col_names[0] = col_names[0]

    col_multiindex = pd.MultiIndex.from_product(col_names, names=level_names)
    line_multiindex = pd.MultiIndex.from_product(index_names, names=["Classifiers", "Score_types"])
    df_results = pd.DataFrame(array_result_tot, columns=col_multiindex, index=line_multiindex)
    df_results_with_avg = get_df_results_avg(df_results)

    cond_col_multiindex = pd.MultiIndex.from_product(cond_col_names, names=cond_level_names)
    cond_df_results = (
        pd.DataFrame(cond_array_result_tot, columns=cond_col_multiindex, index=index_names[0])
        .astype(float)
        .round(3)
    )

    save_classif_report(df_results_with_avg, report_path)
    save_classif_report(cond_df_results, f"{report_path}_cond")


def evaluate_inter_subject(
    data_dict: dict,
    clf_dict: dict,
    score_dict: dict,
    nbr_runs: int,
    report_path: str,
) -> None:
    """
    Performs inter-subject evaluation using leave-one-out cross-validation.
    This function evaluates the performance of multiple classifiers (defined in `clf_dict`)
    on each subject's data using leave-one-out cross-validation. It iterates through each subject
    and uses all other subjects' data for training, then evaluates the classifiers on the left-out subject's data.
    It computes various scores (defined in `score_dict`) for each classifier and subject,
    and reports the mean score across subjects. It also evaluates the performance on each condition
    within the subject's data (if present).
    Args:
        data_dict (dict): A dictionary containing data for multiple subjects. Each key is a subject name,
            and the corresponding value is a dictionary containing data for that subject, organized by condition names.
            The expected structure within the subject's data dictionary is:
            ```
            {
                'condition_name_1': {'X': features_array_1, 'Y': labels_array_1},
                'condition_name_2': {'X': features_array_2, 'Y': labels_array_2},
                ...
            }
            ```
        clf_dict (dict): A dictionary containing machine learning classifiers to evaluate.
            The key is the classifier name, and the value is the classifier object itself.
        score_dict (dict): A dictionary containing scoring functions to use.
            The key is the score name, and the value is the scoring function itself.
        nbr_runs (int): The number of times to repeat the evaluation for each subject (for potential stochasticity).
        report_path (str): The path to save the evaluation report (as two CSV files).
    Returns:
        None
    """
    index_names = [list(clf_dict.keys()), list(score_dict.keys())]
    col_names = []
    cond_col_names = []
    array_result = []
    vect_result_cond = None
    for subj_num, (eval_subj_name, eval_subj_data) in enumerate(data_dict.items()):
        X_train, Y_train = None, None
        X_eval, Y_eval, cond_col_names[1] = load_subject_data(data_dict, eval_subj_name)
        for train_subj_name, train_subj_data in data_dict.items():
            if train_subj_name != eval_subj_name:
                for cond_num, (cond_name, data) in enumerate(train_subj_data.items()):
                    if not cond_num:
                        X_train = data["X"]
                        Y_train = data["Y"]
                    else:
                        X_train = np.vstack((X_train, data["X"]))
                        Y_train = np.hstack((Y_train, data["Y"]))

        vect_result = evaluate(
            X_train, Y_train, X_eval, Y_eval, clf_dict, score_dict, nbr_runs=nbr_runs
        )
        vect_result_cond = evaluate_conditions(
            X_train, Y_train, X_eval, Y_eval, clf_dict, nbr_runs=nbr_runs
        )
        col_names.append(eval_subj_name)
        if len(array_result):
            array_result = np.hstack((array_result, vect_result))
        else:
            array_result = vect_result
        if vect_result_cond is None:
            vect_result_cond = vect_result_cond
        else:
            vect_result_cond = np.add(vect_result_cond, vect_result_cond)
        print(f"test on subject '{eval_subj_name}' done")
        print(f"Mean Result = {np.mean(array_result).round(3)}")
    vect_result_cond /= len(data_dict.keys())

    # Create the report DataFrames and save them
    col_names = add_prefix_to_list(col_names, "Subj_")
    line_multiindex = pd.MultiIndex.from_product(index_names, names=["Classifiers", "Score_types"])
    df_results = pd.DataFrame(array_result, columns=col_names, index=line_multiindex)
    df_results_with_avg = get_df_results_avg(df_results)

    cond_df_results = (
        pd.DataFrame(vect_result_cond, columns=cond_col_names, index=index_names[0])
        .astype(float)
        .round(3)
    )
    save_classif_report(df_results_with_avg, report_path)
    save_classif_report(cond_df_results, f"{report_path}_cond")
