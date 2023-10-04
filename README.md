# STT-Pipeline

## Environment
```
conda create -n kvint python=3.10
conda activate kvint
pip install -r requirements.txt
```

## Preprocess
Беру файл, его делаю моноканальным и меня семпл рейт на 16 кГц. Для нового файла делаю манифест, который используется далее

## Voice Activity Detector:
файл взял отсюда: https://github.com/NVIDIA/NeMo/blob/main/examples/asr/speech_classification/vad_infer.py
с конфигом https://github.com/NVIDIA/NeMo/blob/f477e051ec68aaa909ca891c1605383f87e11fbb/examples/asr/conf/vad/vad_inference_postprocessing.yaml

Поисследовал

Увидел, что этот VAD работает хуже, чем VAD на фреймах

Поэтому взял пример отсюда https://github.com/NVIDIA/NeMo/blob/f477e051ec68aaa909ca891c1605383f87e11fbb/examples/asr/speech_classification/frame_vad_infer.py
с конфигом https://github.com/NVIDIA/NeMo/blob/f477e051ec68aaa909ca891c1605383f87e11fbb/examples/asr/conf/vad/frame_vad_infer_postprocess.yaml

## Промежуточная стадия
Получаю разметку RTTM после VAD, из неё извлекаю временные метки для голоса, а затем нарезаю вавку согласно этой разметке на несколько маленьких вавок.

## Voice Activity Detector:
Передаю список маленьких вавок для разметки. Пример работы взял отсюда: https://github.com/NVIDIA/NeMo/blob/stable/tutorials/asr/Multilang_ASR.ipynb
Получаю разметку для каждого отрывка

## Cleanup
Удаляю все побочные файлы - папку с разметкой, папку с вавками, файл манифеста и преобразованные вавки

# ENTRYPOINT

```
python pipeline.py --filename Recording.wav --filename Recording1.wav
```

В результате в конце выполнения пайплайна в выводе появятся расшифровки файлов, а в корне репозитория появятся изображения работы VAD для каждого из файлов. Пример изображения представлен ниже:
![Alt text](images/Recording.png)

# CLEARML

# Доступы
чтобы видеть ссылки ниже, нужно, чтобы я добавил Вас в воркспейс.

## ClearML Agent
Хостим двух агентов на Google Collab / Kaggle
https://clear.ml/docs/latest/docs/guides/ide/google_colab/
https://colab.research.google.com/github/allegroai/clearml/blob/master/examples/clearml_agent/clearml_colab_agent.ipynb

## ClearML Data
Был создан дадасет https://app.clear.ml/datasets/simple/0af2bc8e207042a4939ff209eb3699bc/experiments/fbfbaa134e0842e69e95c86af313007c?columns=selected&columns=name&columns=hyperparams.properties.version&columns=tags&columns=status&columns=project.name&columns=users&columns=started&columns=last_update&order=-last_update&filter=, содержащий 2 одинаковые вавки (те же самые что и в корне репы) - с помощью файла `clearml/push_dataset.py`

## ClearML Model
Была добавлена в модел реджистри модель https://app.clear.ml/projects/ad9c338261d043399aa080b4741df1f5/models/d0c6c876c1514ef691d1687207e10b29/general?columns=selected&columns=framework&columns=name&columns=tags&columns=ready&columns=project.name&columns=user.name&columns=task.name&columns=last_update&order=-created&filter= - с помощью команды 
```
wget https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_enes_contextnet_large/versions/1.0.0/files/stt_enes_contextnet_large.nemo
```
и файла `clearml/push_model.py`

## ClearML Pipeline
https://app.clear.ml/pipelines/d8fe55fdd9684987ba0bae076b59ad8c/experiments/

энтрипоинт - `clearml/run_pipeline.py`