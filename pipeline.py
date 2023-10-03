import click
from pathlib import Path
import nemo.collections.asr as nemo_asr
import ffmpeg
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import os
import shutil

from src.nemo.frame_vad_infer import main as vad
from hydra import compose, initialize

@click.command()
@click.option(
    '--filename',
    required=True,
    type=click.STRING,
    help="File name"
)
def pipeline(filename: str):

    output_vad_file = "frame_vad_outputs/manifest_vad_output.json"
    manifest_name = 'manifest.json'
    vad_folder = "VAD_wavs"

    input_file_path = Path(filename)
    prep_file = str(f"{input_file_path.stem}_preprocessed{input_file_path.suffix}")

    audio_input = ffmpeg.input(filename)
    audio_output = ffmpeg.output(audio_input, prep_file, format='wav', ar=16000, ac=1)
    ffmpeg.run(audio_output)

    manifest = {
        "audio_filepath": prep_file,
        "offset": 0,
        "duration": None
    }

    with open(manifest_name, "w+") as user_file:
        json.dump(manifest, user_file)
    
    initialize(config_path=".", job_name="test_app")
    cfg = compose(
        config_name="frame_vad_infer_postprocess.yaml",
        return_hydra_config=True,
        overrides=[f"input_manifest={manifest_name}", f"dataset={manifest_name}"]
    )
    vad(cfg)

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

    print(transcripts)

    shutil.rmtree(Path(output_vad_file).parent)
    shutil.rmtree(vad_folder)
    os.remove(manifest_name)
    os.remove(prep_file)

    return None

if __name__ == "__main__":
    pipeline()
