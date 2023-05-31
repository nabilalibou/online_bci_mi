"""
"""
import mne
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from utils.data_utils import add_prefix_to_list, add_suffix_to_list
from utils.preprocessing_utils import get_reported_bad_trials, offline_preprocess
from classification.pipelines import return_scorer_dict, return_clf_dict

matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")
mne.viz.set_browser_backend("qt")

# =============== Constants ===============
# Preprocessing parameters
doPrepro = False
data_repo = "../data"
save_prepro = f"{data_repo}/preprocessed/"  # None
subjects = ["na"]  # "na", "mp"
experiments = ["left_arm", "right_arm"]  # "left_arm", "right_arm", "right_leg"
paradigm = ["me"]  # 'me'  'mi'
l_freq, h_freq = 2, 50
sfreq = 512
doICA = "before_epoching"  # 'before_epoching'  'after_epoching'  None
work_on_sources = False  # if True, save ICs sources and their topomaps for each conditions
epoch_time = (-1.5, 5)  # (-1, 0)
bad_trials = get_reported_bad_trials()  # Trials reported as wrong by the subjects
# Features parameters
# Classification parameters
eval_mode = 'intra'  # 'inter'

# =============== Preprocessing & Data loading ===============
y = []
samples_size = []
count = 0
for subj in subjects:
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
                prepro_repo = save_prepro
                prefix = "prepro"
            for task in paradigm:
                epoch_pickle_path = (
                    f"{prepro_repo}{prefix}_{subj}_{exp}_{task}_filt({l_freq}_{h_freq})_"
                    f"basl{(epoch_time[0], 0)}_ICA{doICA}.pkl"
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
                samples_size.append(X.shape[1])
            count += 1

# Extract features
# Cross-validation and Evaluate intra-subject, inter-subject

# get_classification_report, intrasubj_classification_report
# see if can do a function for both.
nbr_runs = 1  # 7  => no need to be high for Keras NN models
clf_parameter = {
    "keras_models": dict(dropout_rate=0.5, learning_rate=0.005, nbr_epochs=100, batch_size=16,
                         num_samples=X.shape[0], num_chans=X.shape[1],
                         num_features=X.shape[2], sampling_rate=sfreq)
}
clf_selection = ["KNN", "KNNnostd", "CSP + KNN", "CSP4 + KNN", "CSP4 + KNNstd", "rbfSVC", "eegnet"]
score_selection = ['accuracy']
score_dict = return_scorer_dict(score_selection)
clf_dict = return_clf_dict(clf_selection, clf_parameter)
if eval_mode == "intra_ev":
    index_names = [clf_selection, score_selection]
    stride = samples_size
    # Evaluation intra-subject
    n_splits = 6
    cnt_class = 0
    print("========")
    col_names = [[], []]
    array_result = []
    array_result_tot = []
    for subj_test in subjects:
        col_names[1] = []
        kf = KFold(n_splits=n_splits, shuffle=True)
        x1 = subj_test*stride
        for train_index_cond, eval_index_cond in kf.split(X[0:stride], y[0:stride]):
            eval_index = [int(x) + x1 for x in eval_index_cond]
            train_index = [int(x) + x1 for x in train_index_cond]
            X_eval, y_eval = X[eval_index], y[eval_index]
            X_train, y_train = X[train_index], y[train_index]
            report_path = "Results\\rerefnntest"
            start_path = report_path.rfind("\\")
            folder = report_path[:start_path + 1]
            file_name = report_path[start_path + 1:]
            file_name = '_'.join((f"{eval_mode}_{classes[cnt_class]}_{noOfSubjects}subj_{n_splits}fold_{nbr_runs}runs_"
                                  f"{pickle_path_X}", file_name))
            report_path = ''.join((folder, file_name))
            vect_result = evaluate(X_train, y_train, X_eval, y_eval, clf_dict, score_dict, nbr_runs=nbr_runs)
            col_names[1].append("+".join(map(str, eval_index_cond)))
            if len(array_result):
                array_result = np.hstack((array_result, vect_result))
            else:
                array_result = vect_result
        col_names[0].append(str(subj_test))
        if len(array_result_tot):
            array_result_tot = np.hstack((array_result_tot, array_result))
        else:
            array_result_tot = array_result
        print(f"test on subject {subj_test} done")
        print(f"Mean Result = {np.mean(array_result_tot)}")
    col_names[1] = add_suffix_to_list(add_prefix_to_list(col_names[1], "Cond_("), ")")
    col_names[0] = add_prefix_to_list(col_names[0], "Subj_")
    names = ["Subjects", "Conditions"]
    col_multiindex = pd.MultiIndex.from_product(col_names, names=names)
    line_multiindex = pd.MultiIndex.from_product(index_names, names=["Classifiers", "Score_types"])
    df_results = pd.DataFrame(array_result, columns=col_multiindex, index=line_multiindex)
    avg_col_names = ["Avg_by_subj", "Avg_by_cond", "Avg_total"]
    df_results_with_avg = avg_classif_df(df_results, avg_col_names, names)
    write_excel(df_results_with_avg, report_path)
    cnt_class += 1
