from setuptools import find_packages, setup

setup(
    name="online_bci_mi",
    version="0.0.1",
    url="https://github.com/nabilalibou/online_bci_mi.git",
    packages=find_packages(),
    license="MIT",
    author="Nabil Alibou",
    description="This repository explores the development of a real-time Brain-Computer Interface (BCI) capable of "
                "classifying motor execution and motor imagery tasks using EEG signals. The ultimate goal is to create "
                "a rock-paper-scissors game controlled by BCI.",
    python_requires=">=3.10",
)
