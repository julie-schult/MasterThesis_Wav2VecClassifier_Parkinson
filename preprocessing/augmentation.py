from torchaudio.transforms import SpeedPerturbation

"""
speedPert
INPUT
waveform: digital signal of a recording
min: minimum length of signal
max: maximum langth of signal

OUTPUT
augmented waveform: a random value between min and max. Doubles the dataset.
"""

def speedPert(waveform, min, max):
    speed = SpeedPerturbation(16000, [min, max])
    augmented_waveform, _ = speed(waveform)

    waveform_list = [waveform, augmented_waveform]
    return waveform_list

"""
speedPert2
INPUT
waveform: digital signal of a recording
min: minimum length of signal
max: maximum langth of signal

OUTPUT
2 augmented waveform: one from min and one from max. Triples the dataset
"""

def speedPert2(waveform, min, max):
    speed_min = SpeedPerturbation(16000, [min])
    speed_max = SpeedPerturbation(16000, [max])
    
    min_waveform, _ = speed_min(waveform)
    max_waveform, _ = speed_max(waveform)

    waveform_list = [waveform, min_waveform, max_waveform]
    return waveform_list
