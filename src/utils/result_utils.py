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
            # Get unique columns at this level num_lev

            # aaa = df.iloc
            # a = df[col[:level]]
            # level_indexes = df.iloc[:, df.columns.get_level_values(0) == level_name]
            # level_indexes = df[col[:level]].columns.get_level_values(0)
            level_indexes = df.iloc[:level].columns.get_level_values(0)

            list_level_indexes = list(level_indexes.unique())
            # list_level_indexes = list(level_indexes.nunique())
            # Get column number at this level num_lev
            nb_level_indexes.append(len(list_level_indexes))
            # Get theoretical (with extra spaces) length of all columns at this level num_lev
            tot_extra_space_len = extra_space * nb_level_indexes[level - 1]
            tot_len_level.append(len("".join(list_level_indexes)) + tot_extra_space_len)

            # Set cell's maximum adapted length for column col names
            if tot_len_level[-1] >= col_name_len[-2] + extra_space:
                len_longest_col_name = col_name_len[-1]
            elif tot_len_level[-1] < col_name_len[-2] + extra_space:
                len_longest_col_name = (
                    col_name_len[-1] + col_name_len[-2] + extra_space - tot_len_level[-1]
                )

    return len_longest_col_name


# def calculate_widest_cell(df, with_index=False):
#     """Calculate the widest cell of an Excel worksheet made from a pandas DataFrame with multi-indexes.
#
#     Parameters:
#     df (pandas.DataFrame): The DataFrame to calculate the widest cell for.
#     with_index (bool, optional): Whether to include the index in the width calculation. Default is False.
#
#     Returns:
#     int: The width of the widest cell in the Excel worksheet.
#     """
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("df must be a pandas.DataFrame")
#
#     max_width = 0
#     for col in df:
#         if with_index:
#             if isinstance(df.columns, pd.MultiIndex):
#                 for level in df.columns.levels:
#                     width = max([len(str(el)) for el in level])
#                     if width > max_width:
#                         max_width = width
#             else:
#                 width = len(str(col))
#                 if width > max_width:
#                     max_width = width
#         else:
#             values = df[col]
#             if isinstance(values, pd.Series):
#                 for el in values:
#                     width = len(str(el))
#                     if width > max_width:
#                         max_width = width
#
#     return max_width


# def write_excel(df, writer, index=False, columns=None, header=True):
#     """Converts the DataFrame to an Excel file.
#
#     Parameters:
#     df (pandas.DataFrame): The DataFrame to convert.
#     filename (str): The path to the Excel file to write to.
#     index (bool): Whether to include the index in the Excel file.
#     columns (list): A list of column names to include in the Excel file.
#     header (bool): Whether to include the header row in the Excel file.
#     """
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("df must be a pandas.DataFrame")
#
#     # Create a new Excel file
#     # writer = pd.ExcelWriter(filename)
#     df.to_excel(writer)
#
#     # Convert the DataFrame to an Excel sheet
#     df.to_excel(writer, sheet_name="Sheet1", columns=columns, header=header)
#
#     # Set the column widths
#     widest_cell_width = 0
#     if index:
#         for col in df.columns:
#             if isinstance(df.columns, pd.MultiIndex):
#                 for level in df.columns.levels:
#                     longest_name = max([len(str(el)) for el in level])
#                     widest_cell_width = max(widest_cell_width, longest_name)
#             else:
#                 longest_name = len(str(col))
#                 widest_cell_width = max(widest_cell_width, longest_name)
#
#     else:
#         for col in df:
#             longest_name = max([len(str(el)) for el in df[col]])
#             widest_cell_width = max(widest_cell_width, longest_name)
#
#     if widest_cell_width > 0:
#         for col in df:
#             wb = writer.book
#             ws = wb["Sheet1"]
#             ws.set_column(col, col, widest_cell_width)
#
#     # Save the Excel file
#     writer.save()


def auto_adjust_excel_col(df, writer, extra_space=3):
    """
    Parameters
    ----------
    df : DATAFRAME
        Results of classifications.
    writer : STR
        File path to writing.
    extra_space : INT
        Extra space to add on column/line length.

    Returns
    -------
    None.

    """
    df.to_excel(writer)  # send df to writer
    worksheet = writer.sheets["Sheet1"]  # pull worksheet object
    if isinstance(df.index, pd.MultiIndex):
        nlevels = df.index.nlevels
    else:
        nlevels = 1

    # Index cells length adaptation
    for idx in range(0, nlevels): # loop through all index levels
        len_names = len(df.index.names[idx])
        # Get max length between last index name and columns names
        if idx == nlevels-1:
            # len of the largest index or column name/header
            len_names = max(len_names, max(len(str(name)) for name in df.columns.names))
        max_len = max(
            len_names,  # len of the largest index name
            max(len(str(j)) for j in df.index.get_level_values(idx))  # len of the largest column name/header
            ) + extra_space  # adding a little extra space
        worksheet.set_column(idx, idx, max_len)  # set column width

    # Columns cells length adaptation
    for idx, col in enumerate(df):  # loop through all columns
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = df.iloc[:, [idx]].squeeze()
        # Get cell's maximum adapted length for column col names
        if isinstance(series.name, tuple):
            len_longest_col_name = get_optimal_col_len(df, col, extra_space)
        elif isinstance(series.name, str):
            len_longest_col_name = len(series.name)

        max_len = max((
            series.astype(str).map(len).max(),  # len of largest item
            len_longest_col_name  # cell's maximum adapted length
            )) + extra_space  # adding a little extra space
        worksheet.set_column(idx + nlevels, idx + nlevels, max_len)  # set column width
    writer.close()


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
    # try:
    for idx in range(0, df.index.nlevels):
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
    print("ee")


#   except Exception as e:
#       writer.close()
#       raise Exception(f"{e}. Automatic column adjustment could not be applied to the resulting spreadsheet")


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


def write_excel(df, report_path):
    """
    :param df:
    :param report_path:
    :return:
    """
    if not report_path.endswith('.xlsx'):
        report_path = f"{report_path}.xlsx"
    if os.path.exists(report_path):
        print("Report file already exists, the file will be saved under a name format: {report_path}_{datetime}")
        date_time = f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        report_path = date_time.join(os.path.splitext(report_path))
    directory = os.path.dirname(report_path)
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)  # recursively create the path
    writer = pd.ExcelWriter(report_path, engine='xlsxwriter')
    try:
        auto_adjust_excel_col(df, writer)
    except Exception as e:
        print(e)
        writer.close()
    print(f"Results successfully written in '{report_path}'")