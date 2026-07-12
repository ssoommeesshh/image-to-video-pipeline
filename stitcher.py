"""Stitch generated video clips into one final output.

This module deliberately stays focused on file-level concatenation so it can be
used independently of prompt generation and clip creation.
"""

from __future__ import annotations

import subprocess
try:
	from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
	get_ffmpeg_exe = None
from dataclasses import dataclass
from pathlib import Path


class StitchingError(RuntimeError):
	"""Raised when video clips cannot be stitched together."""


@dataclass(slots=True)
class VideoStitcher:
	"""Concatenates clip files into a single output video using ffmpeg."""

	def stitch_clips(self, clip_paths: list[str | Path], output_path: str | Path) -> Path:
		"""Stitch ordered clip files into one final video."""

		if not clip_paths:
			raise StitchingError("At least one clip is required for stitching")

		output_file = Path(output_path)
		output_file.parent.mkdir(parents=True, exist_ok=True)

		normalized_clip_paths = [Path(path) for path in clip_paths]
		for clip_path in normalized_clip_paths:
			if not clip_path.exists():
				raise StitchingError(f"Clip file does not exist: {clip_path}")

		list_file = output_file.parent / f"{output_file.stem}_concat_list.txt"
		list_contents = "\n".join(f"file '{clip_path.resolve().as_posix()}'" for clip_path in normalized_clip_paths)
		list_file.write_text(list_contents + "\n", encoding="utf-8")

		ffmpeg_exe = get_ffmpeg_exe() if get_ffmpeg_exe is not None else "ffmpeg"
		command = [
			ffmpeg_exe,
			"-y",
			"-f",
			"concat",
			"-safe",
			"0",
			"-i",
			str(list_file),
			"-c",
			"copy",
			str(output_file),
		]

		try:
			subprocess.run(command, check=True, capture_output=True, text=True, errors="replace")
		except FileNotFoundError as error:
			ffmpeg_exe = get_ffmpeg_exe() if get_ffmpeg_exe is not None else "ffmpeg"
			raise StitchingError(f"{ffmpeg_exe} is not available on this system") from error
		except subprocess.CalledProcessError as error:
			raise StitchingError(f"Failed to stitch clips: {error.stderr.strip()}") from error
		finally:
			if list_file.exists():
				list_file.unlink()

		if not output_file.exists():
			raise StitchingError(f"Stitching completed but no file was written: {output_file}")

		return output_file


def stitch_clips(clip_paths: list[str | Path], output_path: str | Path) -> Path:
	"""Convenience wrapper for stitching clips with the default stitcher."""

	return VideoStitcher().stitch_clips(clip_paths, output_path)
