# Pilot reliability check

CPU-only local validation on CPython 3.12.13. No GPU, hosted generation, or image collection was used. The notebook passed in a fresh local kernel with an initially empty output directory.

| Requested check | Result |
| --- | --- |
| One missing scene image | Passed: missing map entry and deleted image file reject the complete run before generator calls or output-directory creation |
| Non-image input | Passed: text file rejected with the scene ID before generation |
| Scene continuity | Passed: continuing clip uses the previous decoded last frame; a new scene requires its own image |
| Resume | Passed: unchanged clips reused; prompt/image changes and corrupted outputs regenerated |
| Blocked plan | Passed: unapproved catalog experiment produces no executable clips; runner rejects it |
| Fresh hosted runtime | Not run here: Colab/Kaggle restart requires the user's session. Fresh local kernel with a new output directory is the local equivalent tested |

Automated checks: 25 passed (14 video-module tests and 11 dataset/source tests). Source regressions cover edited-claim invalidation, exact quote validation, per-page approval isolation, low-overlap reviewed support, and portable evidence-query fingerprints.

The notebook now includes repeatable edge-case cells using a separate CPU fixture. In Colab/Kaggle, restart the runtime, reinstall dependencies if necessary, and run every cell from the top. Set `EDU_DATASET_DIR` and `EDU_RUN_DIR` explicitly. A successful local run is not a claim of hosted-runtime verification.

The pilot's source review is in the dataset repository at `reports/boiling_point_source_review.md`. Its agent review and pending human review now travel in `review_provenance` on new plans and manifests. Real scene images and prompt/video quality work remain pending.
