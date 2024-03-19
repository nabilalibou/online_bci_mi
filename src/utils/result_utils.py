import os
import datetime
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot the confusion matrix
    :param y_true:
    :param y_pred:
    :return:
    """
    cf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix")
    sns.heatmap(cf_matrix, annot=True, cmap="Blues")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)


def plot_fit_scores(history):
    """
    Plot training/validation accuracy/loss scores against the number of epochs.
    :param history:
    :return:
    """
    # Plot training and validation accuracy scores against the number of epochs.
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Model Accuracy")
    plt.legend(loc="upper left")

    # Plot training and validation loss scores against the number of epochs.
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Validation")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Model Loss")
    plt.legend(loc="upper right")


def get_optimal_col_len(df, col, extra_space):
    """
    :param df:
    :param col:
    :param extra_space:
    :return:
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    series = df[col]
    len_longest_col_name = max(len(str(i)) for i in series.name)
    col_name_len = []  # column level name length list
    nb_level_indexes = []
    tot_len_level = []  # total length of all column names + extra spaces in one level of indexes
    # Loop on each levels to get cell's maximum adapted length for column col names
    for level, level_name in enumerate(series.name):
        col_name_len.append(len(series.name[level]))
        if level:  # after the 1st level, check the multiindex levels
            df.sort_index(axis=1, inplace=True)  # sort the columns or else PerformanceWarning
            level_indexes = df.iloc[:level].columns.get_level_values(0)
            list_level_indexes = list(level_indexes.unique())
            nb_level_indexes.append(len(list_level_indexes))
            # Get theoretical (with extra spaces) length of all columns at this level num_lev
            tot_extra_space_len = extra_space * nb_level_indexes[level - 1]
            tot_len_level.append(len("".join(list_level_indexes)) + tot_extra_space_len)

            # Set cell's maximum adapted length for column col names
            if tot_len_level[-1] >= col_name_len[-2] + extra_space:
                len_longest_col_name = max(col_name_len)
            elif tot_len_level[-1] < col_name_len[-2] + extra_space:
                len_longest_col_name = (
                    col_name_len[-1] + col_name_len[-2] + extra_space - tot_len_level[-1]
                )

    return len_longest_col_name


def save_excel(df, filename, extra_space=3):
    """Converts the DataFrame to an Excel file and taking multi-indexes into account. Excel cell length set to the size
    of the largest DataFrame column name.

    Parameters:
    df (pandas.DataFrame): The DataFrame to convert.
    filename (str): The path to the Excel file to write to.
    extra_space (int): Extra space to add to the cell length for visual ergonomics.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    if not isinstance(extra_space, int) and extra_space < 1:
        raise TypeError("'extra_space' must be an integer > 0")
    writer = pd.ExcelWriter(filename, engine="xlsxwriter")
    df.to_excel(writer)
    worksheet = writer.sheets["Sheet1"]
    # Set index cell width to the widest index name and the column cell width to the widest column name
    try:
        for idx in range(0, df.index.nlevels):
            length = 0
            if df.index.nlevels > 1:
                length = len(df.index.names[idx])
                # Finds max name length in the level uniting the 'last' row names and the 'first' column names
            if idx == df.index.nlevels - 1:
                length = max(length, max(len(str(name)) for name in df.columns.names))
            max_length = max(
                length, max(len(str(row_label)) for row_label in df.index.get_level_values(idx))
            )
            worksheet.set_column(idx, idx, max_length + extra_space)

        max_length = 0
        for idx, col in enumerate(df):
            if isinstance(df.columns, pd.MultiIndex):
                for level in df.columns.levels:
                    length = max([len(str(el)) for el in level])
                    if length > max_length:
                        max_length = length
            else:
                length = len(str(col))
                if length > max_length:
                    max_length = length
            worksheet.set_column(
                idx + df.index.nlevels, idx + df.index.nlevels, max_length + extra_space
            )

        for idx, col in enumerate(df):  # loop through all columns
            series = df[col]
            widest_col = None
            if isinstance(series, pd.DataFrame):
                series = df.iloc[:, [idx]].squeeze()
            if isinstance(series.name, tuple):
                widest_col = get_optimal_col_len(df, col, extra_space)
            elif isinstance(series.name, str):
                widest_col = len(series.name)
            max_len = max(
                (
                    series.astype(str).map(len).max(),  # len of largest item
                    widest_col,  # cell's maximum adapted length
                )
            )
            worksheet.set_column(idx + df.index.nlevels, idx + df.index.nlevels, max_len + extra_space)
        writer.close()
    except Exception as e:
        writer.close()
        raise Exception(f"{e}. Automatic column adjustment could not be applied to the resulting spreadsheet")


def save_classif_report(df, report_path):
    """
    :param df:
    :param report_path:
    :return:
    """
    if not report_path.endswith(".xlsx"):
        report_path = f"{report_path}.xlsx"
    if os.path.exists(report_path):
        print(
            "Report file already exists, the file will be saved under a name format: {report_path}_{datetime}"
        )
        date_time = f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        report_path = date_time.join(os.path.splitext(report_path))
    directory = os.path.dirname(report_path)
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)  # recursively create the path
    save_excel(df, report_path)
    print(f"Results successfully written in '{report_path}'")
