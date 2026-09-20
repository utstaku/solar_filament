"""MAGFiLO COCO-style annotations.

One physical image (`file_name`) can carry up to three independent annotation sets from different
annotators. Each set is its own entry in `images`, with an id like `050102-20111116063134Lh`, and the
hosts score predictions against every set separately. Anything that splits data must therefore group
by `file_name`, never by entry id.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pycocotools import mask as mask_util

from common import IMAGE_SIZE


@dataclass
class Annotations:
    file_to_entries: dict[str, list[str]]  # file_name -> annotator-image ids
    entry_to_rles: dict[str, list[dict]]  # annotator-image id -> one pycocotools RLE per filament
    n_annotations: int
    n_multi_polygon: int  # filaments drawn as more than one polygon (0 in the 2026 training set)

    def rles_for_file(self, file_name: str) -> list[list[dict]]:
        """Ground truth of one physical image: a list of filament RLEs per annotator."""
        return [self.entry_to_rles.get(entry, []) for entry in self.file_to_entries.get(file_name, [])]


def load_annotations(json_path: str | Path) -> Annotations:
    with open(json_path) as f:
        coco = json.load(f)
    file_to_entries = defaultdict(list)
    for image in coco["images"]:
        file_to_entries[image["file_name"]].append(image["id"])
    entry_to_rles = defaultdict(list)
    n_multi_polygon = 0
    for ann in coco["annotations"]:
        polygons = [p for p in ann["segmentation"] if len(p) >= 6]  # a polygon needs three points
        if not polygons:
            continue
        n_multi_polygon += len(polygons) > 1
        rle = mask_util.merge(mask_util.frPyObjects(polygons, IMAGE_SIZE, IMAGE_SIZE))
        entry_to_rles[ann["image_id"]].append(rle)
    return Annotations(dict(file_to_entries), dict(entry_to_rles), len(coco["annotations"]), n_multi_polygon)
