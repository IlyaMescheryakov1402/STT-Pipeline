from src.nemo.frame_vad_infer import main as vad
from hydra import compose, initialize


def run_vad(
    manifest_name: str,
    config_name: str,
):
    initialize(config_path="../conf", job_name="test_app")
    cfg = compose(
        config_name=config_name,
        return_hydra_config=True,
        overrides=[f"input_manifest={manifest_name}", f"dataset={manifest_name}"]
    )
    vad(cfg)
    return True