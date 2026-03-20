from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from model import DiffusionSchedule, TinyDiffusionModel, sample_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a video from a trained tiny diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/tinyvid.pt"), help="Checkpoint saved by train.py.")
    parser.add_argument("--digit", type=int, default=2, help="Digit to animate.")
    parser.add_argument("--device", type=str, default="cpu", help="Device string, for example cpu or cuda.")
    parser.add_argument("--out", type=Path, default=Path("outputs/output.gif"), help="Output animation path.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output animation.")
    return parser.parse_args()


def _to_uint8_frames(video: torch.Tensor) -> list[torch.Tensor]:
    video = ((video + 1.0) * 0.5).clamp(0.0, 1.0)
    video = (video[:, 0] * 255.0).round().to(torch.uint8).cpu()
    return [frame.contiguous() for frame in video]


def _frame_to_bytes(frame: torch.Tensor) -> bytes:
    return bytes(frame.view(-1).tolist())


def _save_with_pillow(frames: list[torch.Tensor], output_path: Path, fps: int) -> bool:
    try:
        from PIL import Image
    except ImportError:
        return False

    pil_frames = [Image.frombytes("L", (frame.shape[1], frame.shape[0]), _frame_to_bytes(frame)) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return True


def _write_pgm(frame: torch.Tensor, output_path: Path) -> None:
    header = f"P5\n{frame.shape[1]} {frame.shape[0]}\n255\n".encode("ascii")
    output_path.write_bytes(header + _frame_to_bytes(frame))


def _save_with_ffmpeg(frames: list[torch.Tensor], output_path: Path, fps: int) -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        return False

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for index, frame in enumerate(frames):
            _write_pgm(frame, temp_dir_path / f"frame_{index:03d}.pgm")

        command = [
            ffmpeg_path,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(temp_dir_path / "frame_%03d.pgm"),
            str(output_path),
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return True


def save_animation(video: torch.Tensor, output_path: Path, fps: int) -> None:
    frames = _to_uint8_frames(video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _save_with_pillow(frames, output_path, fps):
        return

    if _save_with_ffmpeg(frames, output_path, fps):
        return

    raise RuntimeError(
        "Saving animations requires Pillow or ffmpeg. Install one of them and run test.py again."
    )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    model = TinyDiffusionModel(
        frame_count=checkpoint["frame_count"],
        latent_size=checkpoint["latent_size"],
        hidden_dim=checkpoint["hidden_dim"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model_state"])

    schedule = DiffusionSchedule(num_steps=checkpoint["diffusion_steps"], device=args.device)
    video = sample_video(model, schedule, digit=args.digit, device=args.device)[0]
    save_animation(video, args.out, args.fps)
    print(f"saved sample for digit {args.digit} to {args.out}")


if __name__ == "__main__":
    main()
