from typing import List

def run_asr(
    input_file: str,
    prep_file: str,
    output_vad_file: str,
    vad_folder: str = "VAD_wavs"
):
    import json
    import os
    import librosa
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import nemo.collections.asr as nemo_asr
    import soundfile as sf
    
    with open(output_vad_file, "r") as user_file:
        vad_file = json.load(user_file)

    manifest = []
    with open(vad_file['rttm_filepath']) as f:
        for line in f:
            manifest.append(line)

    os.makedirs(vad_folder, exist_ok=True)

    data, sr = librosa.load(prep_file, sr=16000)

    labels = np.zeros(len(data))

    filelist = []
    for idx, interval in enumerate(manifest):
        file = f'{vad_folder}/wav_{idx}.wav'
        start = float(interval.split(" ")[3])
        end = float(interval.split(" ")[3]) + float(interval.split(" ")[4])
        labels[int(sr * start) : int(sr * end)] = [1] * (int(sr * end) - int(sr * start))
        sf.write(file, data[int(sr * start) : int(sr * end)], sr, 'PCM_24')
        filelist.append(file)

    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name="stt_enes_contextnet_large")
    transcripts = asr_model.transcribe(paths2audio_files=filelist)[0]

    plt.figure(figsize=(10, 10))
    plt.subplots(1, 1, figsize=(12, 6))
    plt.plot(data / max(data), label="Signal", color="b")
    plt.plot(labels, label="LABELS", color='r')
    plt.legend()
    plt.grid()

    # print(transcripts)

    return input_file, transcripts