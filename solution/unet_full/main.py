# kaggle: gpu=true accelerator=NvidiaTeslaT4 internet=false kernel_sources={user}/filament-unet-cv
# %% [markdown]
# # Plain U-Net trained on all training frames
# Fixed 16 epochs; component filter fitted only on previously saved out-of-fold candidates.
# No validation score is claimed for this final full-data model.
# %%
import hashlib
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from cnn_baseline.data import FilamentFrames, channels, collate_tiles, worker_init
from cnn_baseline.training import fit, pick_device, predict
from dataset.annotations import load_annotations
from dataset.paths import find_data_root, find_input, output_dir
from models.unet import UNet
from preprocess.images import read_gray
from pretrained_unet.filtering import ComponentFilter, component_table, frame_context
from pretrained_unet.postprocess import instances, select
from scoring.submission import rows_from_label_map, write

SEED = 0
PARAMS = {"threshold": 0.5, "low": 0.2, "min_area": 150}
CUTOFF = 0.45
start = time.time()
out = output_dir()
root = find_data_root()
train_dir, test_dir = root / "train/train_images", root / "test/test_images"
files = sorted(p.name for p in train_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
test_files = sorted(p.name for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
annotations = load_annotations(next((root / "train").glob("*.json")))
source = find_input("oof_components.csv")
table = pd.read_csv(source)
splits = json.loads((source.parent / "splits.json").read_text())
held_out = [f for split in splits for f in split["validation"]]
assert len(splits) == 5 and len(held_out) == len(set(held_out)) and set(held_out) == set(files)
for split in splits:
    assert not set(split["train"]) & set(split["validation"])
    assert set(table.loc[table.fold == split["fold"], "frame"]) <= set(split["validation"])
assert set(table.fold.unique()) == set(range(5))
# Explicit conversion: never treat string 'False' as truthy.
target = table.target.astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
assert target.notna().all() and target.nunique() == 2
filt = ComponentFilter(CUTOFF).fit(table, target.to_numpy())
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = pick_device()


def log(*args):
    print(f"[{time.time() - start:.0f}s]", *args, flush=True)


log("device", device, "training frames", len(files), "OOF candidates", len(table))
loader = DataLoader(
    FilamentFrames(files, train_dir, annotations, tile=512, tiles_per_frame=4),
    batch_size=4,
    shuffle=True,
    num_workers=3,
    collate_fn=collate_tiles,
    worker_init_fn=worker_init,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True,
    generator=torch.Generator().manual_seed(SEED),
)
model = UNet(in_channels=2, base=32, depth=4)
history = fit(model, loader, epochs=16, lr=1e-3, device=device, log=log)
torch.save(model.state_dict(), out / "unet_full.pt")
report = {
    "train_frames": len(files),
    "test_frames": len(test_files),
    "epochs": 16,
    "seed": SEED,
    "params": PARAMS,
    "cutoff": CUTOFF,
    "loss_history": history,
    "oof_candidates": len(table),
    "oof_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "validation": None,
    "note": "Full-data final model; no independent validation score.",
}
(out / "full_report.json").write_text(json.dumps(report, indent=2))
rows = []
prob_dir = out / "test_probs"
prob_dir.mkdir(exist_ok=True)
for name in test_files:
    image = read_gray(test_dir / name)
    prob = next(predict(model, [torch.from_numpy(channels(image))], device))
    np.save(prob_dir / (name + ".npy"), prob)
    raw = instances(prob, **PARAMS)
    candidates = component_table(frame_context(name, image, prob), raw[0], raw[2])
    labels, _, boxes = select(*raw, filt.keep(candidates))
    rows.extend(rows_from_label_map(name.rsplit(".", 1)[0], labels, boxes))
frame = write(rows, out / "submission.csv")
report.update(submission_rows=len(frame), elapsed_sec=time.time() - start)
(out / "full_report.json").write_text(json.dumps(report, indent=2))
log("complete", report)
