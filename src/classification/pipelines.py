"""
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    SelectPercentile,
    GenericUnivariateSelect,
    SelectFromModel,
)
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from mne.decoding import CSP, Vectorizer, EMS, Scaler, SPoC, UnsupervisedSpatialFilter
from mne.preprocessing import Xdawn
from pyriemann.estimation import (
    Covariances,
    ERPCovariances,
    XdawnCovariances,
    CospCovariances,
    HankelCovariances,
)
from pyriemann.estimation import Coherences as CovCoherences
from pyriemann.tangentspace import TangentSpace
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.classification import MDM, FgMDM, MeanField
from pyriemann.classification import SVC as RSVC
from pyriemann.classification import KNearestNeighbor as RKNN
from pyriemann.spatialfilters import CSP as CovCSP
from pyriemann.spatialfilters import SPoC as CovSPoC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    roc_auc_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
    balanced_accuracy_score,
)
from sklearn.base import BaseEstimator, TransformerMixin
from src.classification.nn_models import *  # Precise the models used


class AddNewDim(TransformerMixin):
    def fit(self, X, y=None):  # native fit method take 3 positional arguments
        return self

    def transform(self, X, y=None):
        return np.expand_dims(X, X.ndim)


class StandardScaler3D(TransformerMixin):
    def fit(self, X, y=None):  # native fit method take 3 positional arguments
        return self

    def transform(self, X, y=None):
        return StandardScaler().fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)


# class EmotivCoverageMapper(TransformerMixin):
#
#     def fit(self, X, y=None):  # native fit method take 3 positional arguments
#         return self
#
#     def transform(self, X, y=None):
#         """
#         return shape: S*T*M*M
#         :param X:
#         :param y:
#         :return:
#         """
#         X_ = np.zeros([X.shape[0], X.shape[-1], 9, 9])
#         for epoch in range(X.shape[0]):
#             for trial in range(X.shape[-1]):
#                 X_[epoch, trial] = data_emotiv_1Dto2D(X[epoch, :, trial])
#         return X_


# Score dict
_all_score_dict = {
    "accuracy": accuracy_score,
    "f1": f1_score,
    "recall": recall_score,
    "precision": precision_score,
    "roc_auc": roc_auc_score,
    "kappa": cohen_kappa_score,
    "balanced_acc": balanced_accuracy_score,
}


def get_all_score_dict():
    return sorted(_all_score_dict.keys())


def return_scorer_dict(score_selection):
    score_dict = {}
    for score in score_selection:
        if score not in _all_score_dict:
            raise KeyError(f"Error: score metric {score} not found in 'all_score_dict")
        for score_name, scorer in _all_score_dict.items():
            if score == score_name:
                score_dict[score_name] = scorer
    return score_dict


def _return_clf_dict(clf_selection, all_clf_dict):
    clf_dict = {}
    for clf in clf_selection:
        if clf not in all_clf_dict:
            raise KeyError(f"Error: classifier {clf} not found in 'all_clf_dict")
        for pipeline_name, pipeline_value in all_clf_dict.items():
            if clf == pipeline_name:
                clf_dict[pipeline_name] = pipeline_value
    return clf_dict


def return_clf_dict(clf_selection, nn_default_params, clf_params={}):
    """
    Key insensitive dictionary of different Estimators and Transformers objects to construct sklearn pipelines.

    For clf in clf_selection: if in big dict{"name": callable} <= need defaut values for kerasclassifier to be
    instantiated no ? how to give params easily. agnostic to maj.

    access param vocab = https://scikit-learn.org/stable/modules/compose.html#nested-parameters

    Notes: Covariance estimator (Cov), CSP and others MNE Estimators/Transformers as well as Keras CNN take 3d data
    (covariance need time point per channel). Others classifiers that are sklearn Estimators and Transformers take 2d
    data (sample, feature).

    Notes: Some of them should be transferred to feature using {Transformer()}.Transform(X). Like CSP, Cov, ERPCov,
    XdawnCov, Coh that take (epoch, channel, time) in input and ouput (epoch, channel, channel).
    Or EMS taking (epoch, channel, time) and output (epoch, channel, time), Xdawn taking (epoch, channel, time) and
    output (epoch, n_components*n_event_types, time). This way it will take less computation time if we compute them
    before cross-validation and we can use combine feature with feature_extraction.py

    :param nn_defaut_params:
    :param clf_selection:
    :param clf_parameter:
    :param clf_params:
    :return:
    """

    dropout_rate = nn_default_params["dropout_rate"]
    learning_rate = nn_default_params["learning_rate"]
    nbr_epochs = nn_default_params["nbr_epochs"]
    batch_size = nn_default_params["batch_size"]
    num_chans = nn_default_params["num_chans"]
    num_features = nn_default_params["num_features"]
    sampling_rate = nn_default_params["sampling_rate"]

    all_estimators = {
        # Data reshape
        "Vect": Vectorizer(),
        "addnewdim": AddNewDim(),
        # "EmotivMap": EmotivCoverageMapper(),
        # Feature Scaler
        "stdScale": StandardScaler(),
        "stdScale3d": StandardScaler3D(),
        "minmaxScale": MinMaxScaler(feature_range=(0, 1)),
        "ChanStd": Scaler(scalings="mean", with_mean=True, with_std=True),
        # Spatial filter
        "CSP": CSP(n_components=4, reg=None),
        "Xdawn": Xdawn(n_components=2),  # improve SNR in ERP
        "SPoC": SPoC(n_components=4, reg=None),
        "EMS": EMS(),
        # Feature extraction & selection
        "PCA": PCA(n_components=None),
        "PCA3d": UnsupervisedSpatialFilter(PCA(n_components=None)),
        "VarTresh": VarianceThreshold(threshold=0),
        "SelectKbest": SelectKBest(k=10),
        "SelectPercent": SelectPercentile(percentile=10),
        "GenUnivSelect": GenericUnivariateSelect(mode="percentile", param=1e-05),
        "SelectFromModel": SelectFromModel(LinearSVC(penalty="l1"), threshold=None),
        # sklearn classifiers
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(C=1.0, kernel="rbf", gamma="scale"),
        "LinSVC": LinearSVC(penalty="l2", C=1.0, max_iter=1000),
        "LR": LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=100),
        "LDA": LinearDiscriminantAnalysis(solver="svd", shrinkage=None),
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.0),
        "GPC": GaussianProcessClassifier(
            kernel=None, max_iter_predict=100, multi_class="one_vs_rest"
        ),
        # == Classifiers on covariance matrices using a Riemannian-based kernel ==
        # Transformers
        "Cov": Covariances(estimator="scm"),
        "erpCov": ERPCovariances(estimator="scm"),
        "xdawnCov": XdawnCovariances(
            nfilter=4, applyfilters=True, estimator="scm", xdawn_estimator="scm"
        ),
        "TS": TangentSpace(metric="riemann"),
        "CovCSP": CovCSP(nfilter=4, metric="euclid", log=True),
        "CospCSP": CospCovariances(window=128, overlap=0.75, fmin=None, fmax=None, fs=None),
        "CovCoh": CovCoherences(
            window=128, overlap=0.75, fmin=None, fmax=None, fs=None, coh="ordinary"
        ),
        "CovSPoC": CovSPoC(nfilter=4, metric="euclid", log=True),
        "HankCov": HankelCovariances(delays=4, estimator="scm"),
        "ChanSelect": ElectrodeSelection(nelec=16, metric="riemann"),
        # Classifiers
        "MDM": MDM(metric=dict(mean="riemann", distance="riemann")),
        "FgMDM": FgMDM(metric=dict(mean="riemann", distance="riemann")),
        "CovSVC": RSVC(metric="riemann", C=1.0, shrinking=True),
        "CovKNN": RKNN(n_neighbors=5, metric=dict(mean="riemann", distance="riemann")),
        "MDMF": MeanField(
            method_label="sum_means", metric=dict(mean="riemann", distance="riemann")
        ),
        # == Decision Tree ==
        "DTC": DecisionTreeClassifier(max_depth=None),
        "RF": RandomForestClassifier(n_estimators=100, max_depth=5, max_features="sqrt"),
        "GBDT": GradientBoostingClassifier(
            learning_rate=0.1,
            max_depth=3,
            min_samples_leaf=1,
            min_samples_split=2,
            n_estimators=100,
        ),
        # == Naive Bayes ==
        "GaussNB": GaussianNB(),
        "CompNB": ComplementNB(alpha=1.0),
        "MultiNB": MultinomialNB(alpha=1.0),
        # == Neural Networks ==
        "Perceptron": Perceptron(
            penalty=None,
            alpha=0.0001,
            l1_ratio=0.15,
            max_iter=1000,
            n_iter_no_change=5,
            class_weight=None,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,), activation="relu", solver="adam", alpha=0.0001, max_iter=200
        ),
        "SGD": SGDClassifier(penalty="l2", alpha=0.0001, l1_ratio=0.15, max_iter=1000),
        "NN": KerasClassifier(
            model=shallow_NN,
            num_chans=num_chans,
            num_features=num_features,
            num_hidden=16,
            activation="relu",
            learning_rate=learning_rate,
            epochs=nbr_epochs,
            batch_size=batch_size,
            verbose=False,
        ),
        "NN2": KerasClassifier(
            model=shallow_NN2,
            num_chans=num_chans,
            num_features=num_features * 14,
            num_hidden=16,
            activation="relu",
            learning_rate=learning_rate,
            epochs=nbr_epochs,
            batch_size=batch_size,
            verbose=False,
        ),
        "DNN_2l": KerasClassifier(
            model=DNNa_2l,
            num_chans=num_chans,
            num_features=num_features,
            num_hidden=60,
            activation="relu",
            learning_rate=learning_rate,
            epochs=nbr_epochs,
            batch_size=batch_size,
            verbose=False,
        ),
        "SCNNa": KerasClassifier(
            model=SCNNa,
            num_chans=num_chans,
            num_features=num_features,
            learning_rate=learning_rate,
            filters=30,
            epochs=nbr_epochs,
            batch_size=batch_size,
            verbose=False,
        ),
        "SCNNb": KerasClassifier(
            model=SCNNb,
            num_chans=num_chans,
            num_features=num_features,
            learning_rate=learning_rate,
            filters=40,
            kernel_size=13,
            pool_size=75,
            strides=15,
            epochs=nbr_epochs,
            batch_size=batch_size,
            verbose=False,
        ),
        "EEGNET": KerasClassifier(
            model=eegnet,
            Chans=num_chans,
            Samples=num_features,
            dropoutRate=dropout_rate,
            learning_rate=learning_rate,
            epochs=nbr_epochs,
            batch_size=batch_size,
            verbose=False,
        ),
        "custEEGNET": KerasClassifier(
            model=custom_eegnet,
            num_chans=num_chans,
            num_features=num_features,
            sampling_rate=sampling_rate,
            dropout_rate=dropout_rate,
            F1=8,
            D=2,
            learning_rate=learning_rate,
            epochs=nbr_epochs,
            batch_size=batch_size,
            verbose=False,
        ),
        # == Ensemble methods ==
        "adaB": AdaBoostClassifier(
            estimator=None, n_estimators=50, learning_rate=1.0, algorithm="SAMME.R"
        ),
    }

    pipeline_dict = {}
    for pipeline_prompt in clf_selection:
        pipeline_name_list = [x.strip(" ") for x in pipeline_prompt.split("+")]
        steps = []
        for clf_name in pipeline_name_list:
            clf_found = False
            for i, (estim_name, estim_funct) in enumerate(all_estimators.items()):
                if clf_name.lower() == estim_name.lower():
                    steps.append((f"{estim_name}", estim_funct))
                    clf_found = True
                if i == len(all_estimators.keys()) - 1 and not clf_found:
                    raise ValueError(
                        f"'{clf_name}' is not a valid estimator name. Valid estimator are :"
                        f"{all_estimators.keys()}"
                    )
        constructed_pipeline = Pipeline(steps)
        constructed_pipeline.set_params(**clf_params)
        pipeline_dict[pipeline_prompt] = constructed_pipeline

    return pipeline_dict
