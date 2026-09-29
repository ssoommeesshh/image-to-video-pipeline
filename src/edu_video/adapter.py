"""Map a validated clip plan into the existing Experiment/Clip models."""
from .models import Clip, Experiment, PromptBundle, Scene


def to_experiment(plan):
    return Experiment(name=plan["title"], subject=plan["subject"],
        knowledge_source=", ".join(str(s.get("source_id", "")) for s in plan["source_references"]),
        metadata={"experiment_id": plan["experiment_id"], "dataset_revision": plan["dataset_revision"]},
        clips=[Clip(name=c["clip_id"], image_state=Scene(scene=c["image_prompt"], camera="", lighting=""),
                    motion_prompt=PromptBundle(image_prompt=c["image_prompt"], motion_prompt=c["motion_prompt"],
                        negative_prompt=c["negative_prompt"], stop_condition=c["stop_condition"],
                        clip_duration_seconds=c["duration_seconds"]),
                    metadata={"scene_id": c["scene_id"], "new_scene": c["new_scene"],
                              "step_ids": c["step_ids"], "source_passage_ids": c["source_passage_ids"]})
               for c in plan["clips"]])
