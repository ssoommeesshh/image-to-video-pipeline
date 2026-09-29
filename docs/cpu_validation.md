# CPU milestone validation

Validated on Windows with CPython 3.12.13. No GPU, model weights, hosted model, or cloud API was used.

- Dataset validator: all 100 records structurally valid.
- Dataset planner plus video integration tests: 18 passed (11 runner tests and 7 dataset tests).
- Both copies of the v1 clip-plan schema match.
- Evidence fingerprints survive LF/CRLF checkout differences and still reject changed source content.
- `notebooks/cpu_workflow.ipynb`: all cells executed successfully with a real Jupyter kernel.
- Selected experiment: `chem_12_101`; three ordered, evidence-linked clips.
- Notebook output: 320 x 240, 12 FPS, 36 decoded frames, 3 seconds.
- Notebook confirmed rejection of `chem_9_001` and reuse of unchanged cached clips.
- CLI dummy execution produced a stitched MP4 and manifest.
- The existing `local_dummy_generator.py` also completed through the new local adapter.
- Source archive and wheel built successfully.
- Wheel installed into a separate environment containing only core dependencies. Its three-clip fixture ran from outside the repository with no PyTorch, SciPy, catalog checkout, or GPU: 18 frames / 1.5 seconds.

The tests cover scene image changes, continuation from the actual final frame, first-image preservation, invalid/missing images, blocked plans, unique IDs, cache invalidation after prompt/image changes, corrupt output regeneration, and failure recovery.

These are integration results. Dummy output consists of static test images. They do not validate scientific apparatus, physical motion, image fidelity, video prompt quality, cloud execution, or the correctness of existing human-review records. Dataset evidence status is carried through rather than newly approved by this work.
