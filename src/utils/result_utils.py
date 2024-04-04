import os
import datetime
import pathlib
import pandas as pd
import numpy as np
import seaborn as sns
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plots a confusion matrix for a classification task.
    This function takes the true labels (`y_true`) and predicted labels (`y_pred`) as NumPy arrays
    and creates a heatmap visualization of the confusion matrix using Seaborn. The confusion matrix
    shows the distribution of how the classifier mapped true labels to predicted labels.
    Args:
        y_true (np.ndarray): The ground truth labels, a NumPy array of integers.
        y_pred (np.ndarray): The predicted labels, a NumPy array of integers with the same length as `y_true`.
    Returns:
        None
    Raises:
        ValueError: May arise if `y_true` and `y_pred` have different lengths.
    """
    cf_matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix")
    sns.heatmap(cf_matrix, annot=True, cmap="Blues")
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)


def plot_fit_scores(history: keras.callbacks.History) -> None:
    """
    Plots training and validation accuracy/loss curves over epochs.
    This function generates two subplots to visualize the training and validation
    performance of a model trained using Keras' `fit` function. It extracts the
    accuracy and loss histories from the provided `history` object (typically a
    `keras.callbacks.History` instance) and plots them against the number of epochs.
    Args:
        history (keras.callbacks.History): The history object returned by the Keras
            `fit` function, containing training and validation performance metrics.
    Returns:
        None
    Raises:
        ValueError: May arise if `history` is not a valid `keras.callbacks.History` object.
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


def get_optimal_col_len(df: pd.DataFrame, col: str, extra_space: int) -> int:
    """
    Calculates the optimal column width for a DataFrame column in Excel,
    considering multi-index levels and extra space.
    Args:
        df (pd.DataFrame): The DataFrame containing the column to be analyzed.
        col (str): The name of the column for which to calculate the optimal width.
        extra_space (int): The desired extra space to be added to each column name,
            for better readability in Excel.
    Returns:
        int: The calculated optimal column width (in characters) for the given column.
    Raises:
        TypeError: If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas.DataFrame")
    series = df[col]
    suitable_cell_size = max(len(str(i)) for i in series.name)
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
                suitable_cell_size = max(col_name_len)
            elif tot_len_level[-1] < col_name_len[-2] + extra_space:
                suitable_cell_size = (
                    col_name_len[-1] + col_name_len[-2] + extra_space - tot_len_level[-1]
                )

    return suitable_cell_size


def save_excel(df: pd.DataFrame, filename: str, extra_space: int = 3) -> None:
    """
    Saves a pandas DataFrame to an Excel file, considering multi-index levels and adjusting column widths for
    readability.
    This function efficiently saves a pandas DataFrame to an Excel file using the `xlsxwriter` engine.
    It automatically calculates optimal column widths based on the following factors:

    - **Multi-index levels:** The function handles multi-index structures effectively,
      determining the maximum width needed for each level name and data element.
    - **Column names:** It considers the length of column names, including names from
      all levels of a multi-index, to ensure proper width allocation.
    - **Data values:** The function calculates the maximum string length of data values
      within each column to accommodate potential long entries.
    - **Extra space:** An optional `extra_space` parameter allows you to specify additional
      spacing for visual improvement in the Excel spreadsheet.

    Args:
        df (pd.DataFrame): The DataFrame to save as an Excel file.
        filename (str): The path to the output Excel file.
        extra_space (int, optional): The amount of extra space (in characters) to add
            to each column width for better readability. Defaults to 3.

    Raises:
        TypeError: If `df` is not a pandas DataFrame or `extra_space` is not a positive integer.
        Exception: If an error occurs during column width adjustment (non-critical, with informative message).
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
            worksheet.set_column(
                idx + df.index.nlevels, idx + df.index.nlevels, max_len + extra_space
            )
        writer.close()
    except Exception as e:
        writer.close()
        raise Exception(
            f"{e}. Automatic column adjustment could not be applied to the resulting spreadsheet"
        )


def save_classif_report(df: pd.DataFrame, report_path: str) -> None:
    """
    Saves a classification report DataFrame to an Excel file with human-readable formatting
    and automatic timestamp-based versioning (in case the file already exists).

    This function saves a DataFrame containing a classification report to an Excel file
    with the following key features:

    - **Column Width Adjustment:** The function ensures optimal column widths for readability
       by calling the `save_excel` function.
    - **Automatic Filename Versioning:** If a file with the same name already exists,
       a timestamp (YYYY-MM-DD_HH-MM-SS format) is appended to create a unique version.
    - **Directory Creation:** Recursively creates any necessary parent directories for the file.

    Args:
        df (pd.DataFrame): The DataFrame containing the classification report to save.
        report_path (str): The desired path to the output Excel file. If it doesn't end with
            ".xlsx", the extension is added automatically.
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
