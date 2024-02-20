"""
"""
import mne
import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
from utils.data_utils import add_prefix_to_list, add_suffix_to_list
from utils.preprocessing_utils import get_reported_bad_trials, offline_preprocess
from feature.extraction import get_features
from classification.pipelines import return_scorer_dict, return_clf_dict
from classification.classification_utils import evaluate, get_df_results_avg
from utils.result_utils import save_classif_report, write_excel

# matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
# plt.switch_backend("Qt5Agg")
mne.viz.set_browser_backend("qt")

# =============== Constants ===============
# Preprocessing parameters
doPrepro = False
data_repo = "../data"
save_prepro = f"../data_preprocessed/"  # None
subjects = ["na"]  # "na", "mp"
experiments = ["left_arm", "right_arm"]  # "left_arm", "right_arm", "right_leg"
paradigm = ["me"]  # 'me'  'mi'
l_freq, h_freq = 2, 50
sfreq = 512
doICA = "before_epoching"  # 'before_epoching'  'after_epoching'  None
work_on_sources = False  # if True, save ICs sources and their topomaps for each conditions
epoch_time = (-1.5, 5)  # (-1, 0)
bad_trials = get_reported_bad_trials()  # Trials reported as wrong by the subjects
data_path_suffix = (
    f"subj_({'_'.join(subjects)})_exp_({'_'.join(experiments)})_({'_'.join(paradigm)})"
)
prepro_path_suffix = f"filt({l_freq}_{h_freq})_basl{(epoch_time[0], 0)}_ICA{doICA}.pkl"

# Features parameters
freq_bands = np.array([0.5, 4.0, 8.0, 13.0, 30.0, 100.0])
# Check feature.extraction.get_features() documentation to see all available features
features_list = ["raw"]

# Classification parameters
eval_mode = "intra"  # 'inter'
nbr_runs = 1  # 7  => no need to be high for Keras NN models as there is 'nbr_epochs'
n_splits = 3  # 6
slide_windows_size = (
    1  # Will transform the feature into a 3D matrices to have batch of 2D frames  [10:80]
)
# kfold = "kfold"  # "stratified", "kfold" or "repstratified"
score_selection = ["accuracy"]
clf_selection = ["KNN", "KNNnostd", "CSP + KNN", "CSP4 + KNN", "CSP4 + KNNstd", "rbfSVC", "eegnet"]
clf_selection = ["Vect + stdScale + KNN"]
save_results = "../results/classif_report"
start_path = save_results.rfind("/")
folder = save_results[: start_path + 1]
file_name = save_results[start_path + 1 :]
file_name = "_".join((f"{eval_mode}_{n_splits}fold_{nbr_runs}runs_{data_path_suffix}", file_name))
path_results = "".join((folder, file_name))

# ===================== Check variables =====================
# Make a function
if not isinstance(slide_windows_size, int) and slide_windows_size < 1:
    raise ValueError(f"variable 'slide_windows_size' should be an integer > 1")
if eval_mode == "intra" and (len(experiments) + len(paradigm)) < 2:
    raise ValueError(f"Not enough conditions to evaluate them in an intra-subject analyse")
if eval_mode == "inter" and len(subjects) < 1:
    raise ValueError(f"Not enough subjects to evaluate them in an inter-subject analyse")
# =============== Preprocessing & Data loading ===============
y = []
data_dict = {}
subj_data_length = [0]
count = 0
for subj in subjects:
    data_length = 0
    data_dict[f"{subj}"] = {}
    for exp in experiments:
        if doPrepro:
            epochs_prepro = offline_preprocess(
                subj,
                exp,
                data_repo=data_repo,
                l_freq=l_freq,
                h_freq=h_freq,
                epoch_time=epoch_time,
                sfreq=sfreq,
                doICA=doICA,
                work_on_sources=work_on_sources,
                bad_trials=bad_trials,
                save_prepro_repo=save_prepro,
            )
        else:
            if work_on_sources:
                prepro_repo = f"{save_prepro}ics_sources/"
                prefix = "sources"
            else:
                prepro_repo = f"{save_prepro}channels/"
                prefix = "prepro"
            for task in paradigm:
                epoch_pickle_path = (
                    f"{prepro_repo}{prefix}_{subj}_{exp}_{task}_{prepro_path_suffix}"
                )
                try:
                    with open(epoch_pickle_path, "rb") as file1:
                        epochs = pickle.load(file1)
                except FileNotFoundError as e:
                    raise FileNotFoundError(e)
                epoch_events = epochs.events[:, 2]
                y.extend(epoch_events)
                if not count:
                    X = epochs.get_data(picks="eeg", tmin=epoch_time[0])
                else:
                    X = np.vstack((X, epochs.get_data(picks="eeg", tmin=epoch_time[0])))
                data_dict[f"{subj}"][f"{exp}_{task}"] = len(epoch_events)
                data_length += len(epoch_events)
                # sample size append paradigm first then for each cond then subj
            count += 1
    subj_data_length.append(data_length)

lb = LabelBinarizer()
y = lb.fit_transform(np.array(y)).flatten()

# feature parameters (see arguments from mne-feature functions at https://mne.tools/mne-features/api.html)
funcs_params = None
if "pow_freq_bands" in features_list:
    funcs_params = {
        "pow_freq_bands__freq_bands": freq_bands,
        "pow_freq_bands__psd_method": "welch",  # multitaper, fft
        "pow_freq_bands__log": False,
        "pow_freq_bands__psd_params": {"welch_n_overlap": 0.5},  # 0.5, 0.25, 0
    }
elif "energy_bands" in features_list:
    funcs_params = {"energy_bands__freq_bands": freq_bands}

X = get_features(X, features_list=features_list, sfreq=sfreq, funcs_params=funcs_params)  # n_jobs=4

# Cross-validation and Evaluate intra-subject, inter-subject

# get_classification_report, intrasubj_classification_report
# see if can do a function for both.

nn_defaut_params = dict(
    dropout_rate=0.5,
    learning_rate=0.005,
    nbr_epochs=40,
    batch_size=16,
    num_chans=X.shape[1],
    num_features=X.shape[2],
    sampling_rate=sfreq,
)
score_dict = return_scorer_dict(score_selection)
clf_dict = return_clf_dict(clf_selection, nn_defaut_params)
index_names = [clf_selection, score_selection]
level_names = ["Subjects", "Conditions"]
# avg_col_names = ["Avg_by_subj", "Avg_by_cond", "Avg_total"]
if eval_mode == "intra":
    # Evaluation intra-subject
    print("========")
    col_names = [[], []]
    array_result_tot = []
    subj_num = 0
    for subj_name, subj_cond in data_dict.items():
        # col_names[1] = []
        kf = KFold(n_splits=n_splits, shuffle=True)
        array_result = []
        for cond_name, sample_size in subj_cond.items():
            vect_result_sum = 0
            for train_index, eval_index in kf.split(
                X[subj_data_length[subj_num] : sample_size],
                y[subj_data_length[subj_num] : sample_size],
            ):
                X_eval, y_eval = X[eval_index], y[eval_index]
                X_train, y_train = X[train_index], y[train_index]
                vect_result = evaluate(
                    X_train, y_train, X_eval, y_eval, clf_dict, score_dict, nbr_runs=nbr_runs
                )
                vect_result_sum += vect_result
            vect_result_sum /= n_splits
            if len(array_result):
                array_result = np.hstack((array_result, vect_result_sum))
            else:
                array_result = vect_result_sum
            col_names[1].append(cond_name)
        col_names[0].append(subj_name)
        if len(array_result_tot):
            array_result_tot = np.hstack((array_result_tot, array_result))
        else:
            array_result_tot = array_result
        print(f"test on subject '{subj_name}' done")
        print(f"Mean Result = {np.mean(array_result_tot)}")
        subj_num += 1
    col_names[1] = add_suffix_to_list(add_prefix_to_list(col_names[1], "Cond_("), ")")
    col_names[0] = add_prefix_to_list(col_names[0], "Subj_")
    col_multiindex = pd.MultiIndex.from_product(col_names, names=level_names)
    line_multiindex = pd.MultiIndex.from_product(index_names, names=["Classifiers", "Score_types"])
    df_results = pd.DataFrame(array_result_tot, columns=col_multiindex, index=line_multiindex)
    df_results_with_avg = get_df_results_avg(df_results)
    write_excel(df_results_with_avg, path_results)
    save_classif_report(df_results_with_avg, path_results)
