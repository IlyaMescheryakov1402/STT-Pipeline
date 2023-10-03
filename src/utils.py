from typing import List

def create_manifest(
    filenames: List[str],
    manifest_name: str,
):
    import ffmpeg
    from pathlib import Path
    import json

    user_file = open(manifest_name, "w")

    prep_files = []
    for filename in filenames:

        input_file_path = Path(filename)
        prep_file = str(f"{input_file_path.stem}_preprocessed{input_file_path.suffix}")

        prep_files.append(prep_file)

        audio_input = ffmpeg.input(filename)
        audio_output = ffmpeg.output(audio_input, prep_file, format='wav', ar=16000, ac=1)
        ffmpeg.run(audio_output)

        manifest = {
            "audio_filepath": prep_file,
            "offset": 0,
            "duration": None
        }

        json.dump(obj=manifest, fp=user_file)
        user_file.write('\n')
    
    return prep_files