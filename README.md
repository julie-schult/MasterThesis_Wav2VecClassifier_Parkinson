# TFE4940 - Electronic Systems Design and Innovation - Master's Thesis

By Julie E. Schult & Laura F. Ven

## Description

This git repository contains all source code written By Julie E. Schult & Laura F. Ven for the course TFE4940 (Electronic Systems Design and Innovation - Master's Thesis) at the Norwegian University of Science and Technology (NTNU). This thesis project is about using transfer learning with the pretrained model Wav2Vec to detect Parkinson'd Desease from audio utterances. All code is written in Python.

## Files overview

* [dataset.py](datasey.py) is a file that contains 4 functions. Two of them are functions that split data into smaller segments og 10 seconds, and pad the utterances to be the same length. One function pads with zeros, while the other one repeats the utterance until wanted length. The last two functions are Custom Dataset files that returns the waveforms and labels in such s way it can be a input to a DataLoader.
* [help_functions.py](help_functions.py) contains help functions used in [dataset.py](datasey.py). One function splits utterances into segments of 10 seconds, and the other function is a train-val-test-splitter that only splits on ID of speaker.
* [plot_utils.py](plot_utils.py) is a file used during training of a model, where the weights are saved as a .pt file.
* [wav2vec_baseline](wav2vec_baseline.py) is the baseline file which contains the wav2vec classifier class. It is possible to modify the layers.
* [wav2vec_notebook.ipynb](wav2vec_notebook.ipynb) shows an example on how you can train and test a wav2vec classifier.
