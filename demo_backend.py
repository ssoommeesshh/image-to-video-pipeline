"""Read-only demo search and explicit prepared-video lookup."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class DemoBackend:
    def __init__(self, dataset_dir: Path, manifest_path: Path | None = None):
        self.dataset_dir = Path(dataset_dir).resolve()
        self.manifest_path = Path(manifest_path or ROOT / "demo_video_cache.json").resolve()
        self.video_root = self.manifest_path.parent.resolve()
        self._rag_module = None
        self._search_cache: dict[tuple[str, str | None], dict] = {}
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        videos = manifest.get("videos")
        if not isinstance(videos, list):
            raise ValueError("Video cache manifest needs a videos list")
        demo_ids: set[str] = set()
        experiment_ids: set[str] = set()
        for item in videos:
            demo_id = item.get("demo_id")
            if not isinstance(demo_id, str) or not demo_id or demo_id in demo_ids:
                raise ValueError("Each video needs a unique demo_id")
            demo_ids.add(demo_id)
            experiment_id = item.get("experiment_id")
            if experiment_id:
                if experiment_id in experiment_ids:
                    raise ValueError(f"Duplicate experiment_id in video cache: {experiment_id}")
                experiment_ids.add(experiment_id)
            path = Path(item.get("path", ""))
            if path.is_absolute() or ".." in path.parts or path.suffix.lower() != ".mp4":
                raise ValueError(f"Unsafe video path for {demo_id}")
            resolved = (self.video_root / path).resolve()
            if not resolved.is_relative_to(self.video_root):
                raise ValueError(f"Video escapes cache root: {demo_id}")
        self._videos = videos

    def videos(self) -> list[dict]:
        result = []
        for item in self._videos:
            path = (self.video_root / item["path"]).resolve()
            result.append({
                **item,
                "available": path.is_file() and path.stat().st_size > 0,
                "url": f"/api/video/{item['demo_id']}" if path.is_file() and path.stat().st_size > 0 else None,
            })
        return result

    def video_for_experiment(self, experiment_id: str) -> dict | None:
        return next((item for item in self.videos() if item["experiment_id"] == experiment_id), None)

    def video_path(self, demo_id: str) -> Path | None:
        item = next((v for v in self._videos if v["demo_id"] == demo_id), None)
        if item is None:
            return None
        path = (self.video_root / item["path"]).resolve()
        return path if path.is_file() and path.stat().st_size > 0 else None

    def _rag(self):
        if self._rag_module is None:
            script = self.dataset_dir / "scripts" / "query_rag.py"
            if not script.is_file():
                raise FileNotFoundError(f"Evidence query script missing: {script}")
            import sys

            scripts = str(script.parent)
            if scripts not in sys.path:
                sys.path.insert(0, scripts)
            spec = importlib.util.spec_from_file_location("demo_rag_query", script)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            self._rag_module = module
        return self._rag_module

    def _evidence(self, experiment_id: str, graph: dict, passages: dict) -> dict:
        nodes = {node["id"]: node for node in graph["nodes"]}
        edges = graph["edges"]
        source_passages = {}
        sections = []
        for link in edges:
            if link["from"] != experiment_id or link["type"] not in {"HAS_CANDIDATE_SECTION", "HAS_SECTION"}:
                continue
            section = nodes[link["to"]]
            for member in edges:
                if member["from"] == section["id"] and member["type"] == "CONTAINS_PASSAGE":
                    passage = passages[member["to"]]
                    source_passages[passage["passage_id"]] = {
                        "passage_id": passage["passage_id"],
                        "manual_filename": passage["manual_filename"],
                        "pdf_page": passage["pdf_page"],
                        "printed_page": passage.get("printed_page"),
                        "review_status": member["status"],
                        "exact_quote": passage["text"],
                    }
            sections.append(section)
        reviews = json.loads((self.dataset_dir / "evidence/reviews.json").read_text(encoding="utf-8"))
        claims = []
        for node in graph["nodes"]:
            if node.get("type") != "claim" or node.get("experiment_id") != experiment_id:
                continue
            support = [link for link in edges if link["from"] == node["id"] and link["type"] in {"SUPPORTED_BY", "CANDIDATE_SUPPORT"}]
            review = reviews.get("claims", {}).get(node["claim_id"], {})
            claims.append({
                "claim_id": node["claim_id"], "text": node["text"], "kind": node["kind"],
                "support_status": "verified" if any(link["status"] == "verified" for link in support) else "needs_review",
                "human_review_status": review.get("human_review_status", "pending"),
            })
        verified = bool(claims) and all(claim["support_status"] == "verified" for claim in claims)
        return {
            "source_status": "has_candidate_sections" if sections else "no_link_in_supplied_pdfs",
            "confidence_status": "source_verified" if verified else "candidate_needs_human_review" if sections else "no_link_in_supplied_pdfs",
            "source_passages": list(source_passages.values()), "claims": claims,
        }

    def _load_graph(self) -> tuple[dict, dict]:
        graph = json.loads((self.dataset_dir / "evidence/evidence_graph.json").read_text(encoding="utf-8"))
        names = ["data/chemistry_experiments.json", "evidence/manuals.json", "evidence/sections.json",
                 "evidence/passages.jsonl", "evidence/reviews.json"]
        hashes = "".join(hashlib.sha256((self.dataset_dir / name).read_text(encoding="utf-8").encode("utf-8")).hexdigest() for name in names)
        fingerprint = hashlib.sha256((str(graph["schema_version"]) + hashes).encode("utf-8")).hexdigest()
        if graph.get("build_fingerprint") != fingerprint:
            raise ValueError("Evidence graph is stale; rebuild it before using the demo")
        passages = {item["passage_id"]: item for line in (self.dataset_dir / "evidence/passages.jsonl").read_text(encoding="utf-8").splitlines() if line.strip() for item in [json.loads(line)]}
        return graph, passages

    def search(self, query: str, experiment_id: str | None = None) -> dict:
        query = query.strip()
        if not query or len(query) > 300:
            raise ValueError("Enter a query between 1 and 300 characters")
        if experiment_id and (len(experiment_id) > 80 or not experiment_id.replace("_", "").isalnum()):
            raise ValueError("Invalid experiment ID")
        key = (query, experiment_id)
        if key not in self._search_cache:
            from scipy import sparse
            with (self.dataset_dir / "rag_vectorizer.pkl").open("rb") as handle:
                vectorizer = pickle.load(handle)
            matrix = sparse.load_npz(str(self.dataset_dir / "rag_index_matrix.npz"))
            documents = json.loads((self.dataset_dir / "rag_index_meta.json").read_text(encoding="utf-8"))
            retrieval = self._rag().retrieve_with_status(query, vectorizer, matrix, documents, top_n=5)
            candidates = retrieval["results"]
            if experiment_id:
                candidate = next((item for item in documents if item["id"] == experiment_id), None)
                if candidate is None:
                    raise ValueError(f"Unknown experiment ID: {experiment_id}")
                meta = candidate["metadata"]
                candidates = [{"id": experiment_id, "title": meta["title"], "subject": meta["subject"],
                               "class_level": meta["class_level"], "score": None,
                               "source_verification": meta.get("source_verification"),
                               "video_readiness": meta.get("video_readiness")}]
            graph, passages = self._load_graph()
            matches = []
            for candidate in candidates:
                match = {"experiment_id": candidate["id"], "title": candidate["title"],
                         "subject": candidate["subject"], "class_level": candidate["class_level"],
                         "retrieval_score": candidate["score"],
                         "source_verification": candidate.get("source_verification"),
                         "video_readiness": candidate.get("video_readiness"),
                         **self._evidence(candidate["id"], graph, passages)}
                match["prepared_video"] = self.video_for_experiment(match["experiment_id"])
                matches.append(match)
            response = {"query": query, "retrieval_status": "explicit_selection" if experiment_id else retrieval["status"],
                        "retrieval_message": f"Selected {experiment_id}; evidence status shown below." if experiment_id else retrieval["message"],
                        "matches": matches,
                        "nearby_experiments": [{"experiment_id": item["id"], "title": item["title"]} for item in retrieval["results"] if item["id"] not in {m["experiment_id"] for m in matches}]}
            self._search_cache[key] = response
        # MP4s can be added while the page stays open; refresh availability even
        # when the text/evidence search itself came from the in-process cache.
        for match in self._search_cache[key]["matches"]:
            match["prepared_video"] = self.video_for_experiment(match["experiment_id"])
        return self._search_cache[key]
