import torch
import numpy as np

"""
INPUTS:
waveform: digital signal of a recording
length_seconds: wanted length of the padded output
padding_type: type of padding style (zero, repeat, reflect, edge, mean)

If the waveform is equal to length_seconds: return waveform
If the waveform is larger than length_seconds: return
If the wrong padding type is defined: return

OUTPUT:
A padded waveform
"""

def padding(waveform, length_seconds, padding_type):
    length = int(length_seconds * 16000)
    
    waveform = waveform.squeeze(0)
    waveform_array = np.array(waveform)
    waveform_length = len(waveform_array)
    padding_size = int((length - waveform_length)/2)
    # Return waveform if it is wanted length (length_seconds)
    if waveform_length == length:
        return waveform
    # Return waveform if it is longer than wanted length (length_seconds)
    if waveform_length > length:
        print('ERROR! THE WAVEFORM IS LONGER THAN YOUR WANTED LENGTH')
        return
    
    ### ZERO ###
    if padding_type == 'zero':
        waveform_padded = np.pad(waveform_array, padding_size, mode='constant', constant_values=0)
        
    ### REPEAT ###
    elif padding_type == 'repeat':
        num_repetitions = length // waveform_length
        remainder = length % waveform_length
        waveform_padded = np.tile(waveform_array, num_repetitions)
        waveform_padded = np.append(waveform_padded, waveform_array[:remainder])
            
    ### REFLECTION ###
    elif padding_type == 'reflect':
        waveform_padded = np.pad(waveform_array, padding_size, mode='reflect')
        
    ### EDGE ###
    elif padding_type == 'edge':
        waveform_padded = np.pad(waveform_array, padding_size, mode='edge')
    ### MEAN ###
    elif padding_type == 'mean':
        waveform_padded = np.pad(waveform_array, padding_size, mode='mean')

    else:
        print('ERROR! YOU HAVE WRONG padding_type INPUT')
        return

    if len(waveform_padded)!=length: waveform_padded = np.append(waveform_padded, 0)
    waveform_padded = torch.from_numpy(waveform_padded)
    return waveform_padded