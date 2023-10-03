from typing import List

def run_asr(
    rttm_file: str,
    input_file: str,
    prep_file: str,
    vad_folder: str = "VAD_wavs",
    modelname: str = "stt_enes_contextnet_large"
):
    import os
    import librosa
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import nemo.collections.asr as nemo_asr
    import soundfile as sf

    manifest = []
    with open(rttm_file) as f:
        for line in f:
            manifest.append(line)

    inputfile_stem = Path(input_file).stem
    target_vad_folder = f"{vad_folder}/{inputfile_stem}"
    os.makedirs(target_vad_folder, exist_ok=True)

    data, sr = librosa.load(prep_file, sr=16000)

    labels = np.zeros(len(data))

    filelist = []
    for idx, interval in enumerate(manifest):
        file = f'{target_vad_folder}/wav_{idx}.wav'
        start = float(interval.split(" ")[3])
        end = float(interval.split(" ")[3]) + float(interval.split(" ")[4])
        labels[int(sr * start) : int(sr * end)] = [1] * (int(sr * end) - int(sr * start))
        sf.write(file, data[int(sr * start) : int(sr * end)], sr, 'PCM_24')
        filelist.append(file)

    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name=modelname)
    transcripts = asr_model.transcribe(paths2audio_files=filelist)[0]

    plt.figure(figsize=(10, 10))
    plt.subplots(1, 1, figsize=(12, 6))
    plt.plot(data / max(data), label="Signal", color="b")
    plt.plot(labels, label="LABELS", color='r')
    plt.legend()
    plt.grid()

    plt.savefig(inputfile_stem)

    return input_file, transcripts