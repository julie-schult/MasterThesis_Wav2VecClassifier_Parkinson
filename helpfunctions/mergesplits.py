import numpy as np
import pandas as pd

# Merge clipped utterances
def mergesplits(all_names, all_predicted_labels, all_labels, test_id):

    id_df = pd.read_csv(test_id)
    #id = id_df['id'].tolist()
    #id_label = id_df['label'].tolist()

    # LISTS TO BE RETURNED
    paths = []
    path_preds = []
    path_labels = []
    ids = []
    id_preds = []
    id_labels = []

    # NESTED LISTS SORTED BY ORIGINAL FILE
    temp_path_preds = []
    temp_path_labels = []

    # FILLING NESTED LISTS
    for idx, name in enumerate(all_names):
        temp_temp_path_preds = []
        temp_temp_path_labels = []

        # GET CORRECT PATH AND ID FORMAT
        # if utterance is not clipped
        if name[-4:] == '.wav':
            if 'PC-GITA' in name:
                id = name.split('/')[-1].split('_')[0]
            elif 'S0489' in name:
                id = name.split('/')[-3]
            path = name
        # if utterance is clipped/augmented
        else: 
            if 'PC-GITA' in name:
                id = name.split('/')[-1].split('_')[0]
            elif 'S0489' in name:
                id = name.split('/')[-3]
            listed_name = name.split('_')[:-2]
            path = '_'.join(listed_name) + '.wav'

        # Adding id if not already in ids
        if id not in ids: ids.append(id)
        # Checking if path is already in paths, and get index
        already_in_temp = False
        already_idx1 = 0
        for idx1, sublist in enumerate(paths):
            if sublist == path:
                already_in_temp = True
                already_idx1 = idx1
        # If path in paths
        if already_in_temp:
            temp_path_preds[already_idx1].append(all_predicted_labels[idx])
            temp_path_labels[already_idx1].append(all_labels[idx])
        # If path not in paths
        else:
            paths.append(path)
            temp_temp_path_preds.append(all_predicted_labels[idx])
            temp_temp_path_labels.append(all_labels[idx])
            temp_path_preds.append(temp_temp_path_preds)
            temp_path_labels.append(temp_temp_path_labels)

    
    # PATHS AND LABELS
    # if one utterance is PD, all utterances for the same ID are PD
    for idx1, sublist in enumerate(temp_path_preds):
        if any(value == 1 for value in sublist):
            path_preds.append(1)
        else: 
            path_preds.append(0)
            
        path_labels.append(temp_path_labels[idx1][0])
    # ID LABELS
    for idx1, id in enumerate(ids):
        id_labels.append(id_df.loc[id_df['id'] == id, 'label'].values[0])
        id_preds.append(0)
        for idx2, label in enumerate(path_preds):
            if id in paths[idx2] and label==1:
                id_preds[idx1] = 1
    return paths, path_preds, path_labels, ids, id_preds, id_labels