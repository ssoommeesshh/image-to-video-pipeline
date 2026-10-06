# Recovering the ammonium-chloride Kaggle run clip by clip

Use this when the full resident experiment cannot finish its startup, but the
single-clip pilot completed. The [original full fixture](../knowledge/ammonium_chloride_resident_pilot.json)
has five generated clips and four CPU-rendered cards. The chunked runner makes
one child experiment per generated clip, so each Wan worker encodes only that
clip's motion prompt, negative prompt, and Wan's default negative prompt.
Completed clip experiments are validated and reused on rerun. An interrupted
clip starts in a fresh attempt folder rather than silently reusing a partial
MP4. Continuation clips use the preceding completed clip's extracted final
frame; the four cards remain in their original sequence.

This trades extra T5/VAE startup time for smaller, independently restartable
jobs. The converted INT8 experts still use the same `/kaggle/tmp` cache within
the current session. A Kaggle session reset can erase that temporary cache.
The code has passed a local CPU end-to-end and resume test; this exact
multi-process orchestration still needs a Kaggle GPU run.

After the [normal two-T4 setup](kaggle_resident_handoff.md), run:

```python
from pathlib import Path
import subprocess, sys

ROOT = Path('/kaggle/working')
PIPELINE = ROOT / 'image-to-video-pipeline'
WAN = ROOT / 'Wan2.2'
subprocess.run(
    ['git', '-C', str(PIPELINE), 'pull', '--ff-only', 'origin', 'feat/cpu-notebook-module'],
    check=True,
)
assert (PIPELINE / 'run_ammonium_chunked.py').is_file()
assert (WAN / 'quantization/kaggle_generate_video.py').is_file()

pilot = ROOT / 'ammonium_chloride_clip1_run_01/ammonium_chloride'
command = [
    sys.executable, str(PIPELINE / 'run_ammonium_chunked.py'),
    '--wan-repo-dir', str(WAN),
    '--cache-dir', '/kaggle/tmp/wan22-int8-cache',
    '--output-dir', str(ROOT / 'ammonium_chloride_chunked_01'),
]
if (pilot / 'final_video.mp4').is_file() and (pilot / 'clip_1/last_frame.png').is_file():
    command += ['--reuse-first-pilot', str(pilot)]
    print('Reusing the successful first clip')
else:
    print('First pilot files unavailable; clip 1 will be generated again')
subprocess.run(command, cwd=PIPELINE, check=True)
final_video = ROOT / 'ammonium_chloride_chunked_01/final_video.mp4'
print(final_video)
```

If the job is interrupted after a completed clip, rerun the same cell with
the **same** `--output-dir`. It checks each completed segment and creates a new
`attempt_02` only for an incomplete or changed step. Do not run it concurrently
with an existing Wan job. The final result is the 18-second stitched MP4 at
`/kaggle/working/ammonium_chloride_chunked_01/final_video.mp4`.

```python
from IPython.display import Video, display
display(Video(str(final_video), embed=True))
```
