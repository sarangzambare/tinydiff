from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset


FRAME_COUNT = 20
FRAME_SIZE = 32
FPS = 10


SEGMENTS = {
    "top": ("h", 6, 6, 26),
    "upper_left": ("v", 6, 6, 16),
    "upper_right": ("v", 26, 6, 16),
    "middle": ("h", 16, 6, 26),
    "lower_left": ("v", 6, 16, 26),
    "lower_right": ("v", 26, 16, 26),
    "bottom": ("h", 26, 6, 26),
}


DIGIT_TO_SEGMENTS = {
    0: ("top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"),
    1: ("upper_right", "lower_right"),
    2: ("top", "upper_right", "middle", "lower_left", "bottom"),
}


@dataclass(frozen=True)
class DatasetConfig:
    digits: Sequence[int] = (0, 1, 2)
    num_samples: int = 2048
    frame_count: int = FRAME_COUNT
    frame_size: int = FRAME_SIZE


def _clamp_int(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _draw_horizontal(frame: torch.Tensor, y: int, x0: int, x1: int, thickness: int, progress: float) -> None:
    if progress <= 0.0:
        return
    x_end = x0 + int(round((x1 - x0) * min(progress, 1.0)))
    y0 = _clamp_int(y - thickness // 2, 0, frame.shape[0] - 1)
    y1 = _clamp_int(y + (thickness + 1) // 2, 0, frame.shape[0])
    x0 = _clamp_int(x0, 0, frame.shape[1] - 1)
    x_end = _clamp_int(x_end, x0 + 1, frame.shape[1])
    frame[y0:y1, x0:x_end] = 1.0


def _draw_vertical(frame: torch.Tensor, x: int, y0: int, y1: int, thickness: int, progress: float) -> None:
    if progress <= 0.0:
        return
    y_end = y0 + int(round((y1 - y0) * min(progress, 1.0)))
    x0 = _clamp_int(x - thickness // 2, 0, frame.shape[1] - 1)
    x1 = _clamp_int(x + (thickness + 1) // 2, 0, frame.shape[1])
    y0 = _clamp_int(y0, 0, frame.shape[0] - 1)
    y_end = _clamp_int(y_end, y0 + 1, frame.shape[0])
    frame[y0:y_end, x0:x1] = 1.0


def render_digit_video(
    digit: int,
    frame_count: int = FRAME_COUNT,
    frame_size: int = FRAME_SIZE,
    variant: int = 0,
) -> torch.Tensor:
    if digit not in DIGIT_TO_SEGMENTS:
        raise ValueError(f"Unsupported digit {digit}. Use 0, 1, or 2.")

    active_segments = DIGIT_TO_SEGMENTS[digit]
    video = torch.zeros(frame_count, 1, frame_size, frame_size, dtype=torch.float32)

    thickness = 2 + (variant % 2)
    wobble_phase = (variant % 8) / 8.0
    wobble_strength = 1 + (variant % 3)

    for frame_idx in range(frame_count):
        t = frame_idx / max(frame_count - 1, 1)
        segment_cursor = t * len(active_segments)
        shift_x = int(round(math.sin((t + wobble_phase) * 2.0 * math.pi) * wobble_strength))
        shift_y = int(round(math.cos((t + wobble_phase) * 2.0 * math.pi) * 1.0))

        canvas = torch.zeros(frame_size, frame_size, dtype=torch.float32)
        for segment_idx, segment_name in enumerate(active_segments):
            local_progress = max(0.0, min(1.0, segment_cursor - segment_idx))
            orientation, a, b, c = SEGMENTS[segment_name]
            if orientation == "h":
                _draw_horizontal(canvas, a + shift_y, b + shift_x, c + shift_x, thickness, local_progress)
            else:
                _draw_vertical(canvas, a + shift_x, b + shift_y, c + shift_y, thickness, local_progress)

        video[frame_idx, 0] = canvas

    return video


class TinyVideoDataset(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.digits = tuple(int(d) for d in config.digits)
        if not self.digits:
            raise ValueError("DatasetConfig.digits must contain at least one digit.")
        invalid_digits = sorted(set(self.digits) - set(DIGIT_TO_SEGMENTS))
        if invalid_digits:
            raise ValueError(f"Unsupported digits in dataset config: {invalid_digits}. Use only 0, 1, or 2.")

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        digit = self.digits[index % len(self.digits)]
        variant = index // len(self.digits)
        video = render_digit_video(
            digit=digit,
            frame_count=self.config.frame_count,
            frame_size=self.config.frame_size,
            variant=variant,
        )
        video = video * 2.0 - 1.0
        return torch.tensor(digit, dtype=torch.long), video


class TensorVideoDataset(Dataset):
    def __init__(self, digits: torch.Tensor, videos: torch.Tensor):
        if digits.ndim != 1:
            raise ValueError("Expected digits tensor with shape [num_samples].")
        if videos.ndim != 5:
            raise ValueError("Expected videos tensor with shape [num_samples, frame_count, 1, height, width].")
        if digits.shape[0] != videos.shape[0]:
            raise ValueError("Digits and videos must contain the same number of samples.")
        self.digits = digits.to(torch.long).contiguous()
        self.videos = videos.to(torch.float32).contiguous()

    def __len__(self) -> int:
        return self.digits.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.digits[index], self.videos[index]


def generate_dataset_tensors(
    digits: Iterable[int] = (0, 1, 2),
    num_samples: int = 2048,
    frame_count: int = FRAME_COUNT,
    frame_size: int = FRAME_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = TinyVideoDataset(
        DatasetConfig(
            digits=tuple(digits),
            num_samples=num_samples,
            frame_count=frame_count,
            frame_size=frame_size,
        )
    )
    all_digits = []
    all_videos = []
    for index in range(len(dataset)):
        digit, video = dataset[index]
        all_digits.append(digit)
        all_videos.append(video)
    return torch.stack(all_digits, dim=0), torch.stack(all_videos, dim=0)


def save_dataset_cache(
    output_path: Path,
    digits: Iterable[int] = (0, 1, 2),
    num_samples: int = 2048,
    frame_count: int = FRAME_COUNT,
    frame_size: int = FRAME_SIZE,
) -> Path:
    digits = tuple(int(d) for d in digits)
    digits_tensor, videos_tensor = generate_dataset_tensors(
        digits=digits,
        num_samples=num_samples,
        frame_count=frame_count,
        frame_size=frame_size,
    )
    payload = {
        "digits": digits_tensor,
        "videos": videos_tensor,
        "config": {
            "digits": digits,
            "num_samples": num_samples,
            "frame_count": frame_count,
            "frame_size": frame_size,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return output_path


def load_dataset_cache(path: Path) -> TensorVideoDataset:
    payload = torch.load(path, map_location="cpu")
    return TensorVideoDataset(payload["digits"], payload["videos"])


def build_dataset(
    digits: Iterable[int] = (0, 1, 2),
    num_samples: int = 2048,
    cache_path: Path | None = None,
    rebuild: bool = False,
) -> TensorVideoDataset:
    digits = tuple(int(d) for d in digits)
    if cache_path is None:
        digits_tensor, videos_tensor = generate_dataset_tensors(digits=digits, num_samples=num_samples)
        return TensorVideoDataset(digits_tensor, videos_tensor)

    if rebuild or not cache_path.exists():
        save_dataset_cache(cache_path, digits=digits, num_samples=num_samples)

    payload = torch.load(cache_path, map_location="cpu")
    config = payload.get("config", {})
    expected_config = {
        "digits": digits,
        "num_samples": num_samples,
        "frame_count": FRAME_COUNT,
        "frame_size": FRAME_SIZE,
    }
    if config != expected_config:
        raise ValueError(
            f"Dataset cache at {cache_path} was built with config {config}, expected {expected_config}. "
            "Use a different cache path or rebuild the dataset."
        )
    return TensorVideoDataset(payload["digits"], payload["videos"])


def _frame_to_bytes(frame: torch.Tensor) -> bytes:
    return bytes(frame.contiguous().view(-1).tolist())


def _video_to_uint8_frames(video: torch.Tensor) -> list[torch.Tensor]:
    video = (video.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu()
    return [frame[0] for frame in video]


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


def save_preview_gif(video: torch.Tensor, output_path: Path, fps: int = FPS) -> None:
    frames = _video_to_uint8_frames(video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _save_with_pillow(frames, output_path, fps):
        return

    if _save_with_ffmpeg(frames, output_path, fps):
        return

    raise RuntimeError("Saving GIF previews requires Pillow or ffmpeg.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate preview GIFs for the procedural digit videos.")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset_previews"), help="Directory to save digit GIFs.")
    parser.add_argument("--variant", type=int, default=0, help="Which procedural style variant to preview.")
    parser.add_argument("--fps", type=int, default=FPS, help="Frames per second for the GIF.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for digit in range(3):
        video = render_digit_video(digit=digit, variant=args.variant)
        output_path = args.out_dir / f"digit_{digit}.gif"
        save_preview_gif(video, output_path, fps=args.fps)
        print(f"saved {output_path}")


if __name__ == "__main__":
    main()
