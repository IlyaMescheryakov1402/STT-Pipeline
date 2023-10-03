import click
from pathlib import Path
import os
import shutil

from typing import List

from src.utils import create_manifest
from src.VAD import run_vad
from src.ASR import run_asr

@click.command()
@click.option(
    '--filenames',
    required=True,
    type=click.STRING,
    help="File name"
)
@click.option(
    '--config_name',
    default="frame_vad_infer_postprocess.yaml",
    type=click.STRING,
    help="Config name for frame VAD"
)
def pipeline(
    filenames: List[str],
    config_name: str,
):

    output_vad_file = "frame_vad_outputs/manifest_vad_output.json"
    manifest_name = 'manifest.json'
    vad_folder = "VAD_wavs"

    create_manifest(filenames, manifest_name)

    prep_files = run_vad(manifest_name, config_name)

    for filename, prep_file in zip(filenames, prep_files):
        run_asr(filename, prep_file, output_vad_file)

    shutil.rmtree(Path(output_vad_file).parent)
    shutil.rmtree(vad_folder)
    os.remove(manifest_name)
    # os.remove(prep_file)

    return None

if __name__ == "__main__":
    pipeline()
