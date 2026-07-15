"""
RAG interface — takes a natural language query and returns
the right experiment procedure + clips, generating new ones
via LLM if the knowledge base doesn't have it yet.
"""
from __future__ import annotations
import json
from pathlib import Path
from difflib import get_close_matches

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def _load_index() -> dict[str, Path]:
    index = {}
    for f in KNOWLEDGE_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            key = data.get("experiment_name", data.get("name", f.stem)).lower()
            index[key] = f
            # also index by subject keywords
            for clip in data.get("clips", []):
                step = clip.get("step", "").lower()
                if step:
                    index[step] = f
        except Exception:
            pass
    return index


def lookup(query: str) -> dict | None:
    index = _load_index()
    q = query.lower().strip()
    if q in index:
        return json.loads(index[q].read_text())
    close = get_close_matches(q, index.keys(), n=1, cutoff=0.5)
    if close:
        print(f"[rag] fuzzy match: '{q}' → '{close[0]}'")
        return json.loads(index[close[0]].read_text())
    return None


def generate_procedure_with_llm(query: str) -> dict:
    """
    On a RAG miss, call a small LLM to generate a structured
    experiment procedure in the same JSON format as the knowledge files.
    """
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    system = """You generate physics/chemistry experiment procedures for a video pipeline.
Output ONLY valid JSON matching this schema exactly:
{
  "experiment_name": "string",
  "subject": "string",
  "base_scene": "detailed description of the lab setting, camera angle, lighting",
  "clips": [
    {
      "clip_id": "clip_1",
      "step": "short step name",
      "image_prompt": "detailed static image description for clip 1 only, null for others",
      "motion_prompt": "what moves/happens in this 5-second clip",
      "negative_prompt": "what should NOT appear"
    }
  ]
}
Use 3-6 clips. Only clip_1 gets an image_prompt, all others use null."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Generate a procedure for: {query}"}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=1000, temperature=0.3, do_sample=True)
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # free model memory immediately
    del model
    torch.cuda.empty_cache()

    try:
        # strip markdown fences if present
        clean = response.strip().strip("```json").strip("```").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"[rag] LLM returned invalid JSON:\n{response[:200]}")
        raise


def store(experiment: dict) -> Path:
    name = experiment.get("experiment_name", experiment.get("name", "unknown"))
    slug = name.lower().replace(" ", "_").replace("/", "_")
    path = KNOWLEDGE_DIR / f"{slug}.json"
    path.write_text(json.dumps(experiment, indent=2))
    print(f"[rag] stored new procedure: {path}")
    return path


def lookup_or_generate(query: str) -> dict:
    result = lookup(query)
    if result:
        print(f"[rag] cache hit: {result.get('experiment_name', result.get('name', 'unknown'))}")
        return result
    print(f"[rag] miss for '{query}' — generating with LLM")
    generated = generate_procedure_with_llm(query)
    store(generated)
    return generated


if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "newton's cradle"
    result = lookup_or_generate(query)
    print(json.dumps(result, indent=2))
