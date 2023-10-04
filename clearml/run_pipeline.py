from typing import Optional


def create_manifest_clearml(
    dataset_id: str,
    manifest_name: str,
):
    from src.utils import create_manifest
    from clearml.automation.controller import PipelineController
    from clearml import Dataset
    import clearml
    import glob
    import os

    # Скачиваем датасет, проверяем что скачался
    dataset_folder = Dataset.get(dataset_id=dataset_id).get_local_copy()
    print(os.system(f"ls {dataset_folder}"))
    filenames = glob.glob(f"{dataset_folder}/*.wav")

    # Препроцессим файлы, создаем манифест (но не используем его)
    prep_files = create_manifest(filenames, manifest_name)

    # Создаем новый датасет, куда кладем предобработанные вавки
    pipe_id = PipelineController._get_pipeline_task().id
    ds = clearml.Dataset.create(
        dataset_name="WAV files preprocessed",
        dataset_project="Ilya",
        dataset_tags=[f"pipeline: {pipe_id}", f"parent: {dataset_id}"]
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
    manifest_name: str,
):
    from src.VAD import run_vad
    from clearml.automation.controller import PipelineController
    from clearml import Dataset
    import glob
    import json
    import os
    from pathlib import Path

    # Скачиваем датасет, проверяем что скачался
    dataset_folder = Dataset.get(dataset_id=prep_dataset_id).get_local_copy()
    print(os.system(f"ls {dataset_folder}"))
    prep_files = glob.glob(f"{dataset_folder}/*.wav")

    # создаем манифест заново, потому что пути до данных изменены
    with open(manifest_name, "w") as user_file:
        for prep_file in prep_files:
            manifest = {
                "audio_filepath": prep_file,
                "offset": 0,
                "duration": None
            }
            print(manifest)
            json.dump(obj=manifest, fp=user_file)
            user_file.write('\n')

    # Запускаем ВАД
    run_vad(manifest_name, config_name)

    # Из манифеста ВАДа достаем RTTM-разметку и как артифакты
    # подгружаем к пайплайну
    with open(output_vad_file, 'r') as manifest:
        for line in manifest.readlines():
            audio_filepath = json.loads(line.strip())['audio_filepath']
            rttm_filepath = json.loads(line.strip())['rttm_filepath']
            rttm_new_name = f"rttm_{Path(audio_filepath).stem}.rttm"
            print(f"rttm_filepath: {rttm_filepath}")
            print(f"rttm_new_name: {rttm_new_name}")

            PipelineController.upload_artifact(
                name=rttm_new_name,
                artifact_object=rttm_filepath
            )

    return None


def run_asr_clearml(
    prep_dataset_id: str,
    model_id: Optional[str]
):
    from src.ASR import run_asr
    from clearml.automation.controller import PipelineController
    from clearml import Logger, Dataset, InputModel, Task
    import os
    import glob
    from pathlib import Path
    import pandas as pd

    logger = Task.current_task().get_logger()
    dataset_folder = Dataset.get(dataset_id=prep_dataset_id).get_local_copy()
    print(os.system(f"ls {dataset_folder}"))
    prep_files = glob.glob(f"{dataset_folder}/*.wav")

    # Если в пайплайн передавалась конкретная модель, используем её
    if model_id is not None:
        out_model = InputModel(model_id=model_id)
        modelname = out_model.get_weights()
        from_clearml = True
        print(modelname)
    else:
        modelname = "stt_enes_contextnet_large"
        from_clearml = False

    result_input_file = []
    result_transcripts = []
    for prep_file in prep_files:

        # достаем название изначального файла исходя из того, как было
        # сформировано название предобработанного файла
        filestem = Path(prep_file).stem

        # Cкачиваем соответствующий артифакт
        rttm_file = PipelineController._get_pipeline_task().artifacts[
            f'rttm_{filestem}.rttm'
        ].get_local_copy()

        # Запускаем ASR
        input_file, transcripts = run_asr(
            rttm_file,
            f"{filestem.split('_')[0]}.wav",
            prep_file,
            modelname=modelname,
            from_clearml=from_clearml
        )

        result_input_file.append(input_file)
        result_transcripts.append(' '.join(transcripts))

        # Логгируем изображения
        logger.report_media(
            title=filestem.split('_')[0],
            series="VAD results",
            iteration=0,
            image=f"{filestem.split('_')[0]}.png",
            delete_after_upload=True
        )

    # Сохраняем наши транскрипты в файл и подгружаем как артифакт
    dictdf = {'filename': result_input_file, 'transcript': result_transcripts}
    df = pd.DataFrame(dictdf)
    df.to_csv('transcript.csv')
    PipelineController.upload_artifact(
        name="transcript",
        artifact_object='transcript.csv'
    )

    # Логгируем результаты в вывод
    logger.report_text("RESULT:\n\n")
    assert len(result_input_file) == len(result_transcripts)
    for idx in range(len(result_transcripts)):
        logger.report_text(
            f"{result_input_file[idx]}: {result_transcripts[idx]}\n"
        )

    return None


if __name__ == "__main__":
    import os
    os.environ["CLEARML_CONFIG_FILE"] = "/home/imeshcheryakov/clearml_public.conf"

    from clearml.automation.controller import PipelineController

    n_workers = 1

    dockerfile = "python:3.10-slim-buster"
    queue = "service"

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
        project_name='Test project',
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
            manifest_name='${pipeline.manifest_name}',
        ),
        project_name='Test project',
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
            prep_dataset_id='${create_manifest_clearml.prep_dataset_id}',
            model_id='${pipeline.model_id}',
        ),
        project_name='Test project',
        repo="https://github.com/IlyaMescheryakov1402/STT-Pipeline.git",
        repo_branch="master",
        docker=dockerfile,
        execution_queue=queue,
        packages="./requirements.txt",
        parents=["create_manifest_clearml", "run_vad_clearml"]
    )

    pipe.start(queue=queue)
