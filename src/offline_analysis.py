import mne
import matplotlib
import matplotlib.pyplot as plt
from utils.preprocessing_utils import get_reported_bad_trials, offline_preprocess

matplotlib.use("Qt5Agg")
# matplotlib.use('TkAgg')
# print(plt.get_backend())
plt.switch_backend("Qt5Agg")
mne.viz.set_browser_backend("qt")


doPrepro = True
save_prepro = True
data_repo = "../data"
subjects = ["na", "mp"]  # "na", "mp"
experiments = [
    "left_arm",
    "right_arm",
    "right_leg",
]  # "left_arm", "right_arm", "right_leg"
l_freq, h_freq = 2, 50
doICA = "after_epoching"  # 'before_epoching'  'after_epoching'  None
work_on_sources = False  # if True, return ICs sources for each conditions
baseline_time = (-1.5, 0)  # (-1, 0)
basl_mode = "mean"  # ‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
bad_trials = get_reported_bad_trials()  # Trials reported as meh by the subjects


for subj in subjects:
    for exp in experiments:
        if doPrepro:
            epochs_prepro = offline_preprocess(
                subj,
                exp,
                data_repo=data_repo,
                l_freq=l_freq,
                h_freq=h_freq,
                baseline_time=baseline_time,
                basl_mode=basl_mode,
                doICA="after_epoching",
                work_on_sources=work_on_sources,
                bad_trials=bad_trials,
                save_prepro=save_prepro,
            )
