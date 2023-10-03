import os
os.environ["CLEARML_CONFIG_FILE"] = "/home/imeshcheryakov/clearml_public.conf"

import clearml
task = clearml.Task.init(
    project_name='Ilya',
    task_name='Upload dataset',
    task_type='custom'
)

ds = clearml.Dataset.create(dataset_name="WAV files", dataset_project="Ilya")

ds.add_files(path="Recording.wav")
ds.add_files(path="Recording1.wav")

ds.upload()

ds.finalize()

task.close()