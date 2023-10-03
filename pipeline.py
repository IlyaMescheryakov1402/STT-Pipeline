import click
from pathlib import Path
import os
import shutil
import json
from typing import List

from src.utils import create_manifest
from src.VAD import run_vad
from src.ASR import run_asr
from nemo.utils import logging

@click.command()
@click.option(
    '--filenames',
    required=True,
    multiple=True,
    help="File names"
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

    logging.info("############################# Step 1 - preprocess files ############################")
    prep_files = create_manifest(filenames, manifest_name)

    logging.info("############################# Step 2 - run VAD #############################")
    run_vad(manifest_name, config_name)

    logging.info("############################# Step 3 - run ASR #############################")

    rttm_filelist = []
    with open(output_vad_file, 'r') as manifest:
        for line in manifest.readlines():
            rttm_filepath = json.loads(line.strip())['rttm_filepath']
            rttm_filelist.append(rttm_filepath)

    result = []
    for rttm_file, filename, prep_file in zip(rttm_filelist, filenames, prep_files):
        input_file, transcripts = run_asr(rttm_file, filename, prep_file)
        result.append((input_file, transcripts))
        os.remove(prep_file)

    shutil.rmtree(Path(output_vad_file).parent)
    shutil.rmtree(vad_folder)
    os.remove(manifest_name)

    logging.info("RESULT:\n\n")
    for input_file, transcripts in result:
        logging.info(f"{input_file}: {' '.join(transcripts)}\n")

    return None

if __name__ == "__main__":
    pipeline()
