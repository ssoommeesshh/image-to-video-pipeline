# Ammonium-chloride Kaggle pilot

Use the two-T4 setup and installs in [the resident Wan handoff](kaggle_resident_handoff.md), then run this cell. The three starting images are already tracked in the `feat/cpu-notebook-module` checkout. No image upload is needed for this experiment.

```python
from pathlib import Path
import subprocess, sys, torch

ROOT = Path('/kaggle/working')
PIPELINE = ROOT / 'image-to-video-pipeline'
WAN = ROOT / 'Wan2.2'
fixture = PIPELINE / 'knowledge/ammonium_chloride_resident_pilot.json'
assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, 'Select Kaggle two T4s'
assert (WAN / 'quantization/kaggle_generate_video.py').is_file()
assert fixture.is_file()
for number in (1, 3, 4):
    image = PIPELINE / f'outputs/ammonium_chloride_test/clip_{number}/input_image.jpeg'
    assert image.is_file(), f'Missing starting image: {image}'

# Increment this name for each changed prompt, image, or model setting.
output = ROOT / 'ammonium_chloride_resident_run_01'
assert not output.exists(), f'Choose a fresh output directory: {output}'
command = [
    sys.executable, str(PIPELINE / 'main.py'),
    '--knowledge-file', str(fixture),
    '--output-dir', str(output),
    '--experiment-name', 'ammonium_chloride',
    '--generator-script', str(PIPELINE / 'kaggle_wan_resident.py'),
    '--generator-executable', sys.executable,
    '--generator-working-dir', str(PIPELINE),
    '--wan-repo-dir', str(WAN),
    '--wan-model-preset', 'i2v-a14b',
    '--continuity-mode', 'chain',
    '--require-resident-daemon',
    '--generator-arg=--lightning',
    f'--generator-arg=--knowledge-file={fixture}',
    '--generator-arg=--gpu0-memory-gib=6',
    '--generator-arg=--gpu1-memory-gib=10',
    '--generator-arg=--memory-telemetry',
]
subprocess.run(command, cwd=PIPELINE, check=True)
print(output / 'ammonium_chloride/final_video.mp4')
```

The five generated clips are `clip_1` through `clip_5`. Clips 2 and 5 use the preceding generated frame; clips 1, 3 and 4 start from the three supplied images. Each Wan call requests two seconds, and four explanation cards are rendered on CPU. This is roughly 18 seconds before any stitch timing adjustment. The card before clip 5 explains that ammonia solution is added between shots; the existing silver-nitrate dropper does not have to turn into a different reagent on camera. It is a full sequence test, so run it after the shorter two-clip pilot if the resident worker has not yet passed a Kaggle GPU run. The CPU test verifies input routing and stitching, but cannot predict Wan's visual fidelity or memory use.

For the white-fume observation, [NCERT's Class XI manual](https://www.ncert.nic.in/pdf/publication/sciencelaboratorymanuals/classXI/chemistry/kelm207.pdf) specifies NaOH, warming, and an HCl-dipped rod near the tube mouth. The fume belongs at the rod, not rising spontaneously from the salt. The manual's Nessler step passes ammonia through the reagent; this fixture instead uses the [ATF direct Nessler spot-test method](https://www.atf.gov/file/178136/download) to fit the existing dropper image. It is labeled as a spot test in the video. The chloride sequence uses silver nitrate on a water extract and ammonium hydroxide to dissolve the white precipitate, as described by NCERT. Have a chemistry reviewer inspect the resulting frames before treating them as an accurate demonstration.
