"""Paper index stage "separate": split the cleaned chunks into text, table and image elements.

Each element gets the id {chunk file stem}__{type}__{index:04d}, which later keys both its raw
content in the docstore and its summary in the vector store. Images that are too small or too
long are usually icons and rules: they stay in the manifest with the reason, but get no summary.

Output folder paper_index/:
  texts/*.pkl    CompositeElement
  tables/*.pkl   Table element
  images/*.pkl   dict with doc_id, image_base64, size_info and metadata
  manifest.json  every element, including the filtered images

The stage rebuilds the folder's element files and manifest from all cleaned chunks every time.
"""

from __future__ import annotations

import json
import logging
import pickle
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from bmrag.build.profiles import ImageFilterSettings
from bmrag.build.stages import BuildContext

log = logging.getLogger(__name__)


@dataclass
class ElementRecord:
    doc_id: str
    element_type: str  # text | table | image
    source_file: str  # chunk file the element came from
    pickle_path: str  # relative to the output folder; empty for a filtered image
    text_preview: str = ""
    has_summary: bool = False
    filtered: bool = False
    filter_reason: str = ""
    image_width: float | None = None
    image_height: float | None = None
    image_page_ratio: float | None = None
    image_aspect_ratio: float | None = None


@dataclass
class Manifest:
    records: list[ElementRecord] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"records": [asdict(r) for r in self.records]}, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls(records=[ElementRecord(**r) for r in json.loads(path.read_text(encoding="utf-8"))["records"]])


def image_size(image: Any) -> dict[str, float] | None:
    """Width, height, area, page ratio and aspect ratio; None if the coordinates are unusable."""
    try:
        coordinates = getattr(image.metadata, "coordinates", None)
        if coordinates is None:
            return None
        data = coordinates.to_dict()
        points = data.get("points")
        if not points or len(points) < 3:
            return None
        (x1, y1), (x3, y3) = points[0], points[2]
        width, height = abs(float(x3 - x1)), abs(float(y3 - y1))
        if height == 0:
            return None
        page_area = float(data.get("layout_width", 1)) * float(data.get("layout_height", 1)) or 1.0
        return {
            "width": width,
            "height": height,
            "area": width * height,
            "page_ratio": (width * height) / page_area,
            "aspect_ratio": width / height,
        }
    except Exception:
        return None


def filter_reason(size: dict[str, float] | None, limits: ImageFilterSettings) -> str:
    """Why an image should be dropped; an empty string keeps it. The text goes into the manifest."""
    if size is None:
        return "no_coordinates"
    max_aspect = limits.max_aspect_ratio
    checks = [
        (
            size["page_ratio"] < limits.min_page_ratio,
            f"page_ratio={size['page_ratio']:.4f} < {limits.min_page_ratio:.3f}",
        ),
        (size["width"] < limits.min_width, f"width={size['width']:.1f} < {limits.min_width}"),
        (size["height"] < limits.min_height, f"height={size['height']:.1f} < {limits.min_height}"),
        (size["aspect_ratio"] > max_aspect, f"aspect_ratio={size['aspect_ratio']:.2f} > {max_aspect} (too wide)"),
        (
            size["aspect_ratio"] < 1.0 / max_aspect,
            f"aspect_ratio={size['aspect_ratio']:.2f} < {1 / max_aspect:.2f} (too tall)",
        ),
    ]
    return next((reason for failed, reason in checks if failed), "")


def _write_pickle(root: Path, relative: Path, obj: Any) -> None:
    with (root / relative).open("wb") as f:
        pickle.dump(obj, f)


def separate_elements(chunk_paths: list[Path], output_root: Path, limits: ImageFilterSettings) -> Manifest:
    from unstructured.documents.elements import CompositeElement, Image, Table

    for folder in ("texts", "tables", "images"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)
    manifest = Manifest()
    counters: Counter[tuple[str, str]] = Counter()

    def next_id(stem: str, element_type: str) -> str:
        index = counters[(stem, element_type)]
        counters[(stem, element_type)] += 1
        return f"{stem}__{element_type}__{index:04d}"

    for chunk_path in sorted(chunk_paths):
        stem = chunk_path.stem
        with chunk_path.open("rb") as f:
            chunks = pickle.load(f)
        if not isinstance(chunks, list):
            log.warning("%s: expected a list of chunks, got %s; skipped", chunk_path.name, type(chunks).__name__)
            continue

        for chunk in chunks:
            if not isinstance(chunk, CompositeElement):
                continue
            doc_id = next_id(stem, "text")
            relative = Path("texts") / f"{doc_id}.pkl"
            _write_pickle(output_root, relative, chunk)
            manifest.records.append(ElementRecord(doc_id, "text", chunk_path.name, str(relative), str(chunk)[:200]))

            for original in getattr(chunk.metadata, "orig_elements", None) or []:
                if isinstance(original, Table):
                    doc_id = next_id(stem, "table")
                    relative = Path("tables") / f"{doc_id}.pkl"
                    _write_pickle(output_root, relative, original)
                    manifest.records.append(
                        ElementRecord(doc_id, "table", chunk_path.name, str(relative), str(original)[:200])
                    )
                elif isinstance(original, Image):
                    encoded = getattr(original.metadata, "image_base64", None)
                    if encoded:
                        manifest.records.append(
                            _store_image(
                                original, encoded, next_id(stem, "image"), chunk_path.name, output_root, limits
                            )
                        )
    return manifest


def _store_image(
    image: Any, encoded: str, doc_id: str, source_file: str, output_root: Path, limits: ImageFilterSettings
) -> ElementRecord:
    size = image_size(image)
    reason = filter_reason(size, limits)
    dimensions = {
        "image_width": size["width"] if size else None,
        "image_height": size["height"] if size else None,
        "image_page_ratio": size["page_ratio"] if size else None,
        "image_aspect_ratio": size["aspect_ratio"] if size else None,
    }
    if reason:
        return ElementRecord(
            doc_id,
            "image",
            source_file,
            "",
            f"[filtered] base64 len={len(encoded)}",
            filtered=True,
            filter_reason=reason,
            **dimensions,
        )
    relative = Path("images") / f"{doc_id}.pkl"
    _write_pickle(
        output_root,
        relative,
        {
            "doc_id": doc_id,
            "image_base64": encoded,
            "size_info": size,
            "metadata": {k: v for k, v in vars(image.metadata).items() if k != "image_base64"},
        },
    )
    return ElementRecord(
        doc_id, "image", source_file, str(relative), f"[image] base64 len={len(encoded)}", **dimensions
    )


def run(ctx: BuildContext) -> dict[str, Any]:
    source, target = ctx.layout.paper_chunks_cleaned, ctx.layout.paper_index
    chunk_files = sorted(source.glob("*.pkl"))
    if not chunk_files:
        raise SystemExit(f"No cleaned chunks in {source}; run the clean stage first")
    manifest = separate_elements(chunk_files, target, ctx.profile.image_filter)
    manifest.save(ctx.layout.paper_manifest)
    kept = Counter(r.element_type for r in manifest.records if not r.filtered)
    filtered = Counter(r.filter_reason.split("=")[0] for r in manifest.records if r.filtered)
    log.info("Kept %s; filtered images by reason %s", dict(kept), dict(filtered))
    return {"kept": dict(kept), "filtered": dict(filtered)}
