import pandas as pd
import torchaudio
import torch
import numpy as np

from padding import padding
from splitdataset import split_audios
from augmentation import speedPert, speedPert2

"""
INPUT
csv_path: a path to a csv containing IDs, file path and label
TRAIN_ID: a path to a csv containing IDs used for training set
VAL_ID: a path to a csv containing IDs used for validation set
TEST_ID: path to a csv containing IDs used for test set
padding_type: type of padding as explained in padding.py (zero, repeat, reflect, edge, mean)
augment: if data augmentation is wanted or not. 0=no augmentation, 1=double dataset, 2: triple dataset
length: desired length for clipping
save: saving type. Save as pth files (requires a lot of storage), or as lists.

Loads all 4 csvs.
Check if augmentation is wanted.
Split audios if needed.
Pad audios if needed.

OUTPUT
Either saved pth files, or lists of data

X_train: list of tensors, containing training data
y_train: list of labels to corresponding training tensors
X_val: list of tensors, containing validation data
y_val: list of labels to corresponding validation tensors
X_test: list of tensors, containing test data
y_test: list of labels to corresponding test tensors
n_test: list of names of corresponding test tensors, used later to add clips together again
"""

def splitpad(csv_path, TRAIN_ID, VAL_ID, TEST_ID, padding_type, augment, length, save):
    df = pd.read_csv(csv_path)
    df_train = pd.read_csv(TRAIN_ID)
    df_val = pd.read_csv(VAL_ID)
    df_test = pd.read_csv(TEST_ID)

    train_id = df_train['id'].tolist()
    val_id = df_val['id'].tolist()
    test_id = df_test['id'].tolist()

    # X=tensor, y=label, n=name
    X_train = []
    X_val = []
    X_test = []
    
    y_train = []
    y_val = []
    y_test = []
    
    n_test = []

    for _, row in df.iterrows():
        id = row['id']
        path = row['path']
        label = row['label']

        waveform, _ = torchaudio.load(path)
        
        if augment == 0:
            waveforms = [waveform]
            extra_names = ['originalClipped']
        if augment == 1:
            waveforms = speedPert(waveform, 0.9, 1.1)
            extra_names = ['originalClipped', 'augmentedClipped']
        if augment == 2:
            waveforms = speedPert2(waveform, 0.9, 1.1)
            extra_names = ['originalClipped', 'augmentedClipped', 'augmentedClipped2']

        for idx, w in enumerate(waveforms):
            if (len(np.array(w.squeeze(0)))) > (int(length*16000)):
                all_clips, all_names = split_audios(file_path=path, extra_name=extra_names[idx], limit_s=10)
                for idx0, val in enumerate(all_clips):
                    val = padding(val, length_seconds=length, padding_type=padding_type)
                    if id in train_id:
                        X_train.append(val)
                        y_train.append(label)
                    elif id in val_id:
                        X_val.append(val)
                        y_val.append(label)
                    elif id in test_id:
                        X_test.append(val)
                        y_test.append(label)
                        n_test.append(all_names[idx0])
            else:
                waveform_padded = padding(w, length_seconds=length, padding_type=padding_type)
                if idx in (1, 2):
                    path = path[:-4] + '_augmented_0'
                    
                if id in train_id:
                    X_train.append(waveform_padded)
                    y_train.append(label)
                elif id in val_id:
                    X_val.append(waveform_padded)
                    y_val.append(label)
                elif id in test_id:
                    X_test.append(waveform_padded)
                    y_test.append(label)
                    n_test.append(path)
    
    if save == 'pth':
        torch.save(X_train, 'X_train.pth')
        torch.save(y_train, 'y_train.pth')
        torch.save(X_val, 'X_val.pth')
        torch.save(y_val, 'y_val.pth')
        torch.save(X_test, 'X_test.pth')
        torch.save(y_test, 'y_test.pth')
        torch.save(n_test, 'n_test.pth')
        return
    else:
        return X_train, y_train, X_val, y_val, X_test, y_test, n_test
    
    
# FOR ONLY PREPROCESS ONE CSV FILE (ex. by testing of another dataset)

def splitpad_simple(csv_path, ID, padding_type, augment, length, save):
    df = pd.read_csv(csv_path)
    df_id = pd.read_csv(ID)

    ids = df_id['id'].tolist()

    # X=tensor, y=label, n=name
    X = []
    y = []
    n = []

    for _, row in df.iterrows():
        id = row['id']
        path = row['path']
        label = row['label']

        waveform, _ = torchaudio.load(path)
        
        if augment == 0:
            waveforms = [waveform]
            extra_names = ['originalClipped']
        if augment == 1:
            waveforms = speedPert(waveform, 0.9, 1.1)
            extra_names = ['originalClipped', 'augmentedClipped']
        if augment == 2:
            waveforms = speedPert2(waveform, 0.9, 1.1)
            extra_names = ['originalClipped', 'augmentedClipped', 'augmentedClipped2']

        for idx, w in enumerate(waveforms):
            if (len(np.array(w.squeeze(0)))) > (int(length*16000)):
                all_clips, all_names = split_audios(file_path=path, extra_name=extra_names[idx], limit_s=10)
                for idx0, val in enumerate(all_clips):
                    val = padding(val, length_seconds=length, padding_type=padding_type)
                    if id in ids:
                        X.append(val)
                        y.append(label)
                        n.append(all_names[idx0])
            else:
                waveform_padded = padding(w, length_seconds=length, padding_type=padding_type)
                if idx in (1, 2):
                    path = path[:-4] + '_augmented_0'
                    
                if id in ids:
                        X.append(waveform_padded)
                        y.append(label)
                        n.append(path)
    
    if save == 'pth':
        torch.save(X, 'X.pth')
        torch.save(y, 'y.pth')
        torch.save(n, 'n.pth')
        return
    else:
        return X, y, n