import json

from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from soundfile import SoundFile


def _parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
    )
    parser.add_argument(
        "--json_path",
        type=str,
    )
    args = parser.parse_args()
    return args


def _get_audio_paths(
    data_root: str, audio_exts: list = [".wav", ".flac", ".mp3", ".aac"]
):
    audio_paths = []

    for audio_ext in audio_exts:
        audio_paths.extend(list(Path(data_root).rglob(f"*{audio_ext}")))

    return list(set(audio_paths))


def _main():
    args = _parse_args()

    audio_paths = _get_audio_paths(args.data_root)

    json_length = {}

    for audio_path in tqdm(audio_paths):
        sn = SoundFile(str(audio_path)).frames
        audio_path_cut = Path(
            *Path(audio_path).parts[len(Path(args.data_root).parts) :]
        )
        audio_path_cut = audio_path_cut.parent / audio_path_cut.stem

        json_length[str(audio_path_cut).replace("\\", "/")] = sn

    Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.json_path, "w") as f:
        json.dump(json_length, f)


if __name__ == "__main__":
    _main()