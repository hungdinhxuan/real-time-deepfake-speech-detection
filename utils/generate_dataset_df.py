from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pyaudio
import matplotlib.pylab as plt
import matplotlib
import torchaudio
import io
import numpy as np
import torch
from utils import set_seed


#torchaudio.set_audio_backend("soundfile")
set_seed(1)
torch.set_num_threads(1)

vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                  model='silero_vad',
                                  force_reload=True)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


# Define constants
DF21_DATASET_PATH = "/home/hungdx/Datasets/ASVspoof2021_DF_eval"
DF21_TRAIN_PROTOCOL_PATH = "/datad/hungdx/KDW2V-AASISTL/protocols/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt"

SAMPLE_RATE = 16000
CUT_SIZE = 16000
DURATION_CUT = 1
DESTINATION_PATH = f"/datad/hungdx/Rawformer-implementation-anti-spoofing/datasets/My_ASVspoof2021_DF_eval_{DURATION_CUT}s"
DESTINATION_DF21_TRAIN_PROTOCOL_PATH = f"/datad/hungdx/Rawformer-implementation-anti-spoofing/datasets/protocols/ASVspoof2021.DF.cm.eval.trl_{DURATION_CUT}s.txt"


THRESHOLD = 0.6  # Threshold for VAD to determine speech


def process_line(line):
    line = line.strip().split()
    file = line[0]
    
    file_path = os.path.join(DF21_DATASET_PATH, file + ".flac")
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.squeeze(0)

    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"File {file}.flac sample rate is not {SAMPLE_RATE}")

    duration = len(waveform) / SAMPLE_RATE
    number_of_cuts = len(waveform) // CUT_SIZE
    residual = len(waveform) % CUT_SIZE

    results = []
    for i in range(number_of_cuts):
        cut_waveform = waveform[i * CUT_SIZE: (i + 1) * CUT_SIZE]
        new_confidence = vad_model(cut_waveform, SAMPLE_RATE).item()
        suffix = "" if new_confidence > THRESHOLD else "_no_speech"
        file_name = f"{file}_{i}{suffix}.flac"    
        
        file_path = os.path.join(DESTINATION_PATH, file_name)
        
        try:
            torchaudio.load(file_path)
        except:
            torchaudio.save(os.path.join(DESTINATION_PATH, file_name),
                            cut_waveform.unsqueeze(0), SAMPLE_RATE)

    # Process residual part if necessary
    if residual > 0 and len(waveform[-residual:]) >= 512:
        process_residual_part(waveform, residual,
                              number_of_cuts, line, results)

    return results


def process_residual_part(waveform, residual, count, line, results):
    cut_waveform = waveform[-residual:]
    new_confidence = vad_model(cut_waveform, SAMPLE_RATE).item()
    suffix = "" if new_confidence > THRESHOLD else "_no_speech"
    file_name = f"{line[0]}_{count}_residual{suffix}.flac"
    file_path = os.path.join(DESTINATION_PATH, file_name)
    try:
        torchaudio.load(file_path)
    except:
        torchaudio.save(file_path,
                        cut_waveform.unsqueeze(0), SAMPLE_RATE)

# Reading the lines in advance to prevent file access issues in threads
with open(DF21_TRAIN_PROTOCOL_PATH) as file:
    lines = file.readlines()
 
# or ThreadPoolExecutor(max_workers=number_of_workers)
executor = ProcessPoolExecutor(max_workers=40)
futures = [executor.submit(process_line, line) for line in lines]


for future in tqdm(as_completed(futures), total=len(futures)):
    result = future.result()
    
