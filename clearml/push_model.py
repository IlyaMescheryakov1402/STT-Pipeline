import os
import clearml
os.environ["CLEARML_CONFIG_FILE"] = "/home/imeshcheryakov/clearml_public.conf"

task = clearml.Task.init(
    project_name='Ilya',
    task_name='Upload model',
    task_type='custom'
)
out_model = clearml.OutputModel(
    task=task,
    name='stt_enes_contextnet_large.nemo',
    framework='NEMO'
)
out_model.set_upload_destination("https://files.clear.ml")
out_model.update_weights(
    weights_filename='models/stt_enes_contextnet_large.nemo'
)
task.close()
