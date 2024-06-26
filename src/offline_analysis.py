"""
"""
import mne
import numpy as np
from utils.preprocessing_utils import get_reported_bad_trials, offline_preprocess
from feature.extraction import get_features
from classification.pipelines import return_scorer_dict, return_pipelines
from classification.classification_utils import evaluate_intra_subject, evaluate_inter_subject

# matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
# plt.switch_backend("Qt5Agg")
# mne.viz.set_browser_backend("qt")

# =============== Constants ===============
# Preprocessing parameters
doPrepro = False
data_repo = "../data"
save_prepro = f"../results/data_preprocessed/"
save_data = True
subjects = ["na", "mp"]  # "na", "mp"
experiments = ["left_arm", "right_arm"]  # "left_arm", "right_arm", "right_leg"
paradigm = ["me"]  # 'me'  'mi'
l_freq, h_freq = 0.1, 50
sfreq = 512
work_on_sources = False  # if True, save ICs sources and their topomaps for each conditions
interpolate = True
epoch_duration = 5
epoch_baseline = (-1.5, 0)  # (-1, 0)
bad_trials = get_reported_bad_trials()  # Trials reported as wrong by the subjects
data_path_suffix = (
    f"subj_({'_'.join(subjects)})_exp_({'_'.join(experiments)})_({'_'.join(paradigm)})"
)
prepro_path_suffix = f"filt({l_freq}_{h_freq})_basl{epoch_baseline}.fif"

# Classification parameters
eval_mode = "intra"  # 'inter'  'intra'
nbr_runs = 1  # 7  => no need to be high for Keras NN models as there is 'nbr_epochs'
n_splits = 10  # 6
slide_windows_size = (
    1  # Will transform the feature into a 3D matrices to have batch of 2D frames
)
score_selection = ["accuracy", "balanced_acc"]
pipeline_selection = ["KNN", "KNNnostd", "CSP + KNN", "CSP4 + KNN", "CSP4 + KNNstd", "rbfSVC", "eegnet"]
# "PCA3d + CSP + KNN", "PCA3d + CSP + LDA", "PCA3d + CSP +  LR", "PCA3d + CSP + LR",
pipeline_selection = ["Vect + KNN", "Vect + SVC", "Cov + TS + LR"] # "CSP + KNN", "CSP + stdScale + KNN" "Cov + TS + LR"
pipeline_selection = ["Vect + SVC", {"Vect": {}, "KNN": {"n_neighbors": 6}}]
# pipeline_selection = [{"Vect": {}, "KNN": {"n_neighbors": 6}}, {"Vect": {}, "SVC": {"kernel": "poly"}}]
report_path = f"../results/classif_report/{eval_mode}_{n_splits}fold_{nbr_runs}runs_{data_path_suffix}"


# Features parameters
# Check feature.extraction.get_features() documentation to see all available features
features_list = ["raw"]
freq_bands = np.array([0.5, 4.0, 8.0, 13.0, 30.0, 100.0])
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

# ===================== Check variables =====================
# Make a function
if not isinstance(slide_windows_size, int) and slide_windows_size < 1:
    raise ValueError(f"variable 'slide_windows_size' should be an integer > 1")
if eval_mode == "intra" and (len(experiments) + len(paradigm)) < 2:
    raise ValueError(f"Not enough conditions to evaluate them in an intra-subject analyse")
if eval_mode == "inter" and len(subjects) < 1:
    raise ValueError(f"Not enough subjects to evaluate them in an inter-subject analyse")

# =============== Preprocessing & Data loading ===============
num_chans = 0
num_features = 0
data_dict = {}
subj_data_length = [0]
for subj in subjects:
    data_length = 0
    count = 0
    data_dict[f"{subj}"] = {}
    for exp in experiments:
        if doPrepro:
            epochs_prepro = offline_preprocess(
                subj,
                exp,
                data_repo=data_repo,
                l_freq=l_freq,
                h_freq=h_freq,
                epoch_duration=epoch_duration,
                epoch_baseline=epoch_baseline,
                sfreq=sfreq,
                work_on_sources=work_on_sources,
                interpolate=interpolate,
                bad_trials=bad_trials,
                save_prepro_repo=save_prepro,
                Save_data=save_data
            )
        if work_on_sources:
            prepro_repo = f"{save_prepro}ic_sources/"
            prefix = "sources"
        else:
            prepro_repo = f"{save_prepro}/"
            prefix = "prepro"
        for task in paradigm:
            epoch_path = (
                f"{prepro_repo}{prefix}_{subj}_{exp}_{task}_{prepro_path_suffix}"
            )
            data_dict[f"{subj}"][f"{exp}_{task}"] = {}
            try:
                epochs = mne.read_epochs(epoch_path, proj=True, preload=True, verbose=None)
            except FileNotFoundError as e:
                raise FileNotFoundError(e)
            epoch_events = epochs.events[:, 2]
            X = epochs.get_data(picks="eeg")
            y = np.full(len(epoch_events), count)
            data_dict[f"{subj}"][f"{exp}_{task}"]["X"] = get_features(
                X, features_list=features_list, sfreq=sfreq, funcs_params=funcs_params
            )
            data_dict[f"{subj}"][f"{exp}_{task}"]["Y"] = y
            num_chans = X.shape[1]
            num_features = X.shape[2]
            data_length += len(epoch_events)
            count += 1
            # sample size append paradigm first then for each cond then subj
    subj_data_length.append(data_length)

# Cross-validation and Evaluate intra/inter-subject

nn_defaut_params = dict(
    dropout_rate=0.5,
    learning_rate=0.005,
    nbr_epochs=40,
    batch_size=16,
    num_chans=num_chans,
    num_features=num_features,
    sampling_rate=sfreq,
)
score_dict = return_scorer_dict(score_selection)
pipeline_dict = return_pipelines(pipeline_selection, nn_defaut_params)
# avg_col_names = ["Avg_by_subj", "Avg_by_cond", "Avg_total"]

match eval_mode:
    case "intra":
        evaluate_intra_subject(data_dict, n_splits, pipeline_dict, score_dict, nbr_runs, report_path)
    case "inter":
        evaluate_inter_subject(data_dict, pipeline_dict, score_dict, nbr_runs, report_path)
    case _:
        raise ValueError("eval_mode need to be either 'intra' or 'inter'")
