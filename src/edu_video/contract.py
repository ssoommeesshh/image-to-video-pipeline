"""Versioned clip-plan boundary validation, including sequence invariants."""
import copy
import json
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator


def validate_plan(plan):
    schema = json.loads(files("edu_video").joinpath("schemas/clip_plan.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(plan)
    ids = [clip["clip_id"] for clip in plan["clips"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Clip IDs must be unique")
    previous = None
    for clip in plan["clips"]:
        if clip["scene_id"] != previous and not clip["new_scene"]:
            raise ValueError(f"{clip['clip_id']}: scene transition requires new_scene=true")
        if not clip["new_scene"] and clip["starting_image"] is not None:
            raise ValueError("Continuing clips use the previous frame, not a starting image")
        if plan["review_status"] == "evidence_gate_passed":
            if clip["review_status"] != "evidence_gate_passed" or not clip["source_passage_ids"]:
                raise ValueError("Evidence-cleared clips require passage IDs and matching review status")
        previous = clip["scene_id"]
    return plan


def load_plan(plan):
    if isinstance(plan, (str, Path)):
        path = Path(plan).expanduser().resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent
    else:
        data, base = copy.deepcopy(plan), Path.cwd()
    return validate_plan(data), base
