from setuptools import find_packages, setup

setup(
    name="eeg_clf_sandbox",
    version="0.0.1",
    url="https://github.com/Nabil-AL/eeg-clf_sandbox.git",
    packages=find_packages(),
    license="MIT",
    author="Nabil AL",
    description="Repository used to test several preprocessing/classification pipelines "
    "on brain signals (EEG/EMG/MEG datasets).",
    python_requires=">=3.10",
)
