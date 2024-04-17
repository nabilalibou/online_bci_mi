# Real-Time Brain-Computer Interface for Rock-Paper-Scissors

This repository explores the development of a real-time Brain-Computer Interface (BCI) capable of classifying motor 
execution and motor imagery tasks using EEG signals. The ultimate goal is to create a rock-paper-scissors game 
controlled by BCI using minimal training data.

The project arose from a desire to compete with a colleague in various games using BCI technology. We aimed to develop a system that 
accurately classifies different limb movements using only 50 training samples per condition.

## Approach:

This study is divided into two parts:

**Offline Analysis**: This phase involves preprocessing and classification of recorded EEG signals to identify optimal 
algorithms for real-time processing.

**Online Analysis**: This phase implements the chosen algorithms in a real-time BCI system using [MNE-lsl](https://github.com/mne-tools/mne-lsl) for online 
classification and control.

## Experimental Setup:

- Subjects: Two right-handed participants (me and my colleague).
- Tasks: Three motor execution (ME) and three motor imagery (MI) tasks, each focused on a different limb (left arm, right 
arm, right leg).
- Data Acquisition: 32-channel EEG system (TMSI) with 2064 Hz sampling rate using a standard 10/20 electrode montage.
- Protocol: Each session consisted of an initial resting state followed by 50 repetitions of a sequence combining ME and 
MI tasks as illustrated in the diagram below:

<p align="center">
<img src="docs\readme_img\protocol.png" width="547" height="913">
</p>

## Offline Analysis
### Preprocessing
The raw EEG data undergoes various steps to remove artifacts and noise, including:
- Band-pass filtering (0.1 Hz - 50 Hz) with zero-phase response.
- Automatically detect bad channels based on peak-to-peak amplitude, deviation from the mean amplitude, correlation with 
others channels and power spectral density distribution.
- Epoching (-0.15s to 5s around each event-task).
- Re-referencing to the 'Cz' electrode.
- Reconstruction of bad channels by spherical spline interpolation [[1]](#1).
- ICA for artifact rejection with automatic component labeling using [MNE-ICALabel](https://github.com/mne-tools/mne-icalabel).
- Baseline correction.
- Automatic bad epoch rejection based on the power spectral density and the Riemannian covariance (using the Potato 
algorithm [[2]](#2)).

### Feature extraction 
(Details to be added)  
### Classifiers
(Details on chosen algorithms to be added).  
### Classification Scores 
(Results of offline classification - intra-subject & inter-subject - to be added).  

## Online Analysis: 
### Preprocessing
(Details to be added) 
### Feature extraction 
(Details to be added)  
### Classifiers
(Details on chosen algorithms to be added).  
### Classification Scores 
(Results of offline classification - intra-subject & inter-subject - to be added).

## Getting Started:
Clone the repository and install the dependencies:
```
git clone https://github.com/nabilalibou/online_bci_mi.git
pip install -r requirements.txt
```

## Video/Gif: 
(To be added - demonstrate online preprocessing and classification)

# References

<a id="1">[1]</a>
F. Perrin, J. Pernier, O. Bertrand, and J. F. Echallier, “Erratum: Spherical splines for scalp 
potential and current density mapping (Electroenceph. Clin. Neurophysiol. 1989, 72: 184
187),” vol. 76, Jan. 1990.

<a id="2">[2]</a>
The Riemannian Potato: an automatic and adaptive artifact detection method for online experiments using Riemannian 
geometry A. Barachant, A Andreev, and M. Congedo. TOBI Workshop lV, Jan 2013, Sion, Switzerland. pp.19-20.