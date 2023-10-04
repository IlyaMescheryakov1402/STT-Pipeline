from typing import List, Optional

def create_manifest_clearml(
    dataset_id: str,
    manifest_name: str,
):
    from src.utils import create_manifest
    from clearml.automation.controller import PipelineController
    import clearml
    import glob
    import os

    dataset_folder = clearml.Dataset.get(dataset_id=dataset_id).get_local_copy()
    print(os.system(f"ls {dataset_folder}"))
    filenames = glob.glob(f"{dataset_folder}/*.wav")
    prep_files = create_manifest(filenames, manifest_name)
    PipelineController.upload_artifact(name="vad_input_manifest", artifact_object=manifest_name)

    ds = clearml.Dataset.create(
        dataset_name="WAV files preprocessed",
        dataset_project="Ilya",
        parent_dateset=dataset_id
    )
    for file in prep_files:
        ds.add_files(path=file)
    ds.upload()
    ds.finalize()

    return ds.id


def run_vad_clearml(
    prep_dataset_id: str,
    config_name: str,
    output_vad_file: str,
):
    from src.VAD import run_vad
    from clearml.automation.controller import PipelineController
    manifest_name = PipelineController._get_pipeline_task().artifacts['vad_input_manifest'].get_local_copy()

    dataset_folder = clearml.Dataset.get(dataset_id=prep_dataset_id).get_local_copy()
    print(os.system(f"ls {dataset_folder}"))
    filenames = glob.glob(f"{dataset_folder}/*.wav") # надо в манифесте заменить пути на эти пути, и сделать это до подачи в ВАД

    run_vad(manifest_name, config_name)

    # output_vad_file = PipelineController._get_pipeline_task().artifacts['vad_output_manifest'].get_local_copy()
    rttm_filelist = []
    with open(output_vad_file, 'r') as manifest:
        for line in manifest.readlines():
            audio_filepath = json.loads(line.strip())['audio_filepath']
            rttm_filepath = json.loads(line.strip())['rttm_filepath']
            # rttm_filelist.append(rttm_filepath)
            print(f"rttm_filepath: {rttm_filepath}")
            PipelineController.upload_artifact(name=f"rttm_{audio_filepath}", artifact_object=rttm_filepath)
    return None


def run_asr_clearml(
    dataset_id: str,
    prep_files: List[str],
    model_id: Optional[str]
):
    from src.ASR import run_asr
    from clearml.automation.controller import PipelineController
    import os
    import shutil
    import glob
    from pathlib import Path
    import pandas as pd
    import clearml

    dataset_folder = clearml.Dataset.get(dataset_id=dataset_id).get_local_copy()
    print(os.system(f"ls {dataset_folder}"))
    filenames = glob.glob(f"{dataset_folder}/*.wav")

    if model_id is not None:
        out_model = clearml.InputModel(model_id=model_id)
        modelname = out_model.get_weights()
        print(modelname)
    else:
        modelname = "stt_enes_contextnet_large"

    # output_vad_file = PipelineController._get_pipeline_task().artifacts['vad_output_manifest'].get_local_copy()
    # rttm_filelist = []
    # with open(output_vad_file, 'r') as manifest:
    #     for line in manifest.readlines():
    #         rttm_filepath = json.loads(line.strip())['rttm_filepath']
    #         rttm_filelist.append(rttm_filepath)

    result_input_file = []
    result_transcripts = []
    # for rttm_file, filename, prep_file in zip(rttm_filelist, filenames, prep_files):
    for filename in filenames:
        filestem = Path(filename).stem
        rttm_file = PipelineController._get_pipeline_task().artifacts[f'rttm_{filestem}.wav'].get_local_copy()

        input_file, transcripts = run_asr(rttm_file, filename, prep_file, modelname=modelname)
        result_input_file.append(input_file)
        result_transcripts.append(' '.join(transcripts))
        os.remove(prep_file)

    dictdf = {'filename': result_input_file, 'transcript': result_transcripts}
        
    df = pd.DataFrame(dictdf)
    df.to_csv('transcript.csv')

    print("RESULT:\n\n")
    for input_file, transcripts in result:
        print(f"{input_file}: {' '.join(transcripts)}\n")

    PipelineController.upload_artifact(name="transcript", artifact_object='transcript.csv')

    return None

# def pipeline(
#     filenames: List[str],
#     config_name: str,
# ):

#     output_vad_file = "frame_vad_outputs/manifest_vad_output.json"
#     manifest_name = 'manifest.json'
#     vad_folder = "VAD_wavs"

#     logging.info("############################# Step 1 - preprocess files ############################")
#     prep_files = create_manifest(filenames, manifest_name)

#     logging.info("############################# Step 2 - run VAD #############################")
#     run_vad(manifest_name, config_name)

#     logging.info("############################# Step 3 - run ASR #############################")

#     rttm_filelist = []
#     with open(output_vad_file, 'r') as manifest:
#         for line in manifest.readlines():
#             rttm_filepath = json.loads(line.strip())['rttm_filepath']
#             rttm_filelist.append(rttm_filepath)

#     result = []
#     for rttm_file, filename, prep_file in zip(rttm_filelist, filenames, prep_files):
#         input_file, transcripts = run_asr(rttm_file, filename, prep_file)
#         result.append((input_file, transcripts))
#         os.remove(prep_file)

#     shutil.rmtree(Path(output_vad_file).parent)
#     shutil.rmtree(vad_folder)
#     os.remove(manifest_name)

#     logging.info("RESULT:\n\n")
#     for input_file, transcripts in result:
#         logging.info(f"{input_file}: {' '.join(transcripts)}\n")

#     return None

if __name__ == "__main__":
    import os
    os.environ["CLEARML_CONFIG_FILE"] = "/home/imeshcheryakov/clearml_public.conf"

    # import click
    # from pathlib import Path
    # import shutil
    # import json
    # from typing import List

    # from src.utils import create_manifest
    # from src.VAD import run_vad
    # from src.ASR import run_asr
    # from nemo.utils import logging

    from clearml.automation.controller import PipelineController
    # import os
    # from copy import copy
    # from dotenv import load_dotenv

    n_workers = 1

    dockerfile = "python:3.10-slim-buster"
    queue="service"

    pipe = PipelineController(
        name='STT pipeline',
        project='Test project',
        target_project=True,
        version="1.0.0",
        add_pipeline_tags=True,
        docker=dockerfile,
        packages="./requirements.txt"
    )

    pipe.add_parameter(
        name='dataset_id',
        description='Id of clearml dataset',
        default="fbfbaa134e0842e69e95c86af313007c",
    )

    pipe.add_parameter(
        name='model_id',
        description='Id of clearml model',
        default="d0c6c876c1514ef691d1687207e10b29"
    )

    pipe.add_parameter(
        name='manifest_name',
        description='manifest_name',
        default="manifest.json"
    )

    pipe.add_parameter(
        name='config_name',
        description='config_name',
        default="frame_vad_infer_postprocess.yaml"
    )

    pipe.add_parameter(
        name='output_vad_file',
        description='output_vad_file',
        default="frame_vad_outputs/manifest_vad_output.json"
    )

    print('launch step one')
    pipe.add_function_step(
        name='create_manifest_clearml',
        function=create_manifest_clearml,
        function_kwargs=dict(
            dataset_id='${pipeline.dataset_id}',
            manifest_name='${pipeline.manifest_name}',
        ),
        function_return=['prep_dataset_id'],
        project_name='', 
        repo="https://github.com/IlyaMescheryakov1402/STT-Pipeline.git", 
        repo_branch="master",
        docker=dockerfile,
        execution_queue=queue,
        packages="./requirements.txt",
        parents=[]
    )

    print('launch step two')
    pipe.add_function_step(
        name='run_vad_clearml',
        function=run_vad_clearml,
        function_kwargs=dict(
            prep_dataset_id='${create_manifest_clearml.prep_dataset_id}',
            config_name='${pipeline.config_name}',
            output_vad_file='${pipeline.output_vad_file}',
        ),
        project_name='', 
        repo="https://github.com/IlyaMescheryakov1402/STT-Pipeline.git", 
        repo_branch="master",
        docker=dockerfile,
        execution_queue=queue,
        packages="./requirements.txt",
        parents=["create_manifest_clearml"]
    )


    print('launch step three')
    pipe.add_function_step(
        name='run_asr_clearml',
        function=run_asr_clearml,
        function_kwargs=dict(
            dataset_id='${pipeline.dataset_id}',
            prep_dataset_id='${create_manifest_clearml.prep_dataset_id}',
            model_id='${pipeline.model_id}',
        ),
        project_name='', 
        repo="https://github.com/IlyaMescheryakov1402/STT-Pipeline.git", 
        repo_branch="master",
        docker=dockerfile,
        execution_queue=queue,
        packages="./requirements.txt",
        parents=["create_manifest_clearml", "run_vad_clearml"]
    )

    pipe.start(queue=queue)
    # pipe.start_locally(run_pipeline_steps_locally=True)
