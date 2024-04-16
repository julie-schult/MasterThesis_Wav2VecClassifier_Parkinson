import torchaudio

"""
INPUTS
file_path: path to a wav-file
extra_name: an extra name to be saved, to find the split utterances later (ex. clipped)
limit_s: the split length

Loads the file to waveform, splits the waveform into smaller segments of limit_s seconds. Creates names for each segment. Returns all segments and names.

OUTPUTS
all_clips: segments
all_names: names of segments
"""
# For long audio files (> 10s), we want to split them into smaller segments of 10s
def split_audios(file_path, extra_name, limit_s=10):
    fs = 16000
    
    name = file_path[:-4]
    
    all_clips = []
    all_names = []

    waveform,_ = torchaudio.load(file_path)
    waveform = waveform.squeeze(0)
    len_w = len(waveform)
    cutting_limit = limit_s * fs

    for idx in range(0, len_w, cutting_limit):
        segment = waveform[idx:(idx+cutting_limit)]
        all_clips.append(segment)
    for n in range(len(all_clips)):
        new_name = name + '_' + extra_name + '_' + str(n+1)
        all_names.append(new_name)
        
    return all_clips, all_names