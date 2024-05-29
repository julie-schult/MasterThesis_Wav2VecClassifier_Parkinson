# TFE4940 - Electronic Systems Design and Innovation - Master's Thesis

By Julie E. Schult & Laura F. Ven

## Description

This git repository contains all source code written By Julie E. Schult & Laura F. Ven for the course TFE4940 (Electronic Systems Design and Innovation - Master's Thesis) at the Norwegian University of Science and Technology (NTNU). This thesis project is about using transfer learning with the pretrained model Wav2Vec to detect Parkinson's Desease from audio utterances. All code is written in Python.

## Files overview
* [preprocessing](preprocessing/) is a folder containing all files needed for preprocessing.
  * [padding.py](preprocessing/padding.py) contains padding methods (zero, repeat, reflect, edge, mean).
  * [augmentation.py](preprocessing/augmentation.py) contains two functions where SpeedPerturbation is used in order to make the dataset either 2 or 3 times bigger.
  * [splitdataset.py](preprocessing/splitdataset.py) contains a function that splits longer audio files (ex. longer than 10s) into smaller segments.
  * [splitpadsave.py](preprocessing/splitpadsave.py) contains a bigger function that uses all the functions above. It takes traininhg, validation and test data, and do splitting, augmentation and padding. Eventually it saves the data either as files, or returns them.
* [helpfunctions](helpfunctions/) is a folder containing all help/support files for training and testing the model.
  * [saveweights.py](helpfunctions/saveweights.py) contains a function that saves weights from an epoch as a pt file.
  * [mergesplits.py](helpfunctions/mergesplits.py) contains a function used in the testing. It sorts the data that gathers the split/augmented utterances that becomes one whole file. If at least ONE of the utterances are classified as "PD", the whole file is classified as "PD"
* [models](models/) is a folder containing all wav2vec classifiers used. All models have the same architecture: Wav2Vec Feature extractor, convolution layer, batch norm, relu, maxpool, dropout, flatten, linear, batch norm, relu.
  * [wav2vecClassifier_commonvoice.py.py](models/wav2vecClassifier_commonvoice.py.py) contains a model class, where the Wav2Vec model is trained on the Common Voice dataset. 
  * [Wav2Vec2Classifier_librispeech.py](models/Wav2Vec2Classifier_librispeech.py) contains a model class, where the Wav2Vec model is trained on Librispeech.
