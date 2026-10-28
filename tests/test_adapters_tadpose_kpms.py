"""TadPose SLEAP and keypoint-MoSeq adapters on a small synthetic bundle."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from boutscope.adapters.keypoint_moseq import KpmsImport, import_keypoint_moseq
from boutscope.adapters.tadpose_sleap import TadPoseImport, import_tadpose_sleap
from boutscope.cli import main
from boutscope.pose import PoseStore
from boutscope.project import ProjectError, load_project

STEMS = ("cropped_plateA_001", "cropped_plateB_001")
WELLS = ("A1", "A2")
N_SLOTS = 40
DROPPED_SLOT = 20  # camera dropped a frame: slot 20 has no raw frame
SKIP = 10  # raw_frame = frame + 10 (start trim)
OFFSETS = {"plateA_001": (200, 60), "plateB_001": (190, 57)}


@pytest.fixture
def bundle(tmp_path):
    rows, omitted = [], []
    for stem in STEMS:
        for well in WELLS:
            for slot in range(N_SLOTS):
                frame = slot if slot < DROPPED_SLOT else slot - 1
                entry = {
                    "video_stem": stem,
                    "well": well,
                    "slot": slot,
                    "frame": frame,
                    "raw_frame": frame + SKIP,
                }
                if slot == DROPPED_SLOT:
                    omitted.append({**entry, "frame": None, "raw_frame": None})
                elif slot % 13 == 5:
                    omitted.append(entry)  # usable frame, but not a bundle row
                else:
                    rows.append(entry)
    folder = tmp_path / "bundle"
    folder.mkdir()
    rows = pd.DataFrame(rows)
    rows.to_parquet(folder / "rows.parquet")
    pd.DataFrame(omitted).astype({"frame": "Int64", "raw_frame": "Int64"}).to_parquet(
        folder / "omitted_rows.parquet"
    )
    labels = (rows["slot"].to_numpy() // 7) % 3
    np.save(tmp_path / "pdv.npy", labels.astype(np.int32))
    np.save(tmp_path / "vel.npy", (labels + 1) % 2)
    catalogue = {
        str(k): {"label": f"pdv.{k + 1:02d}", "colour": "#009E73", "kind": "forward"}
        for k in range(3)
    }
    catalogue["_global"] = {}
    (tmp_path / "pdv.json").write_text(json.dumps(catalogue), encoding="utf-8")
    pd.DataFrame(
        [
            {"video": f"/somewhere/{k}.mp4", "x": x, "y": y, "w": 866, "h": 574}
            for k, (x, y) in OFFSETS.items()
        ]
    ).to_csv(tmp_path / "crop.csv", index=False)
    pd.DataFrame(
        [{"recording": f"{s}__{w}", "animal": f"{s[8:14]}_{w}"} for s in STEMS for w in WELLS]
    ).to_csv(tmp_path / "animals.csv", index=False)
    pd.DataFrame(
        [{"video": s, "condition": c, "plate": s[8:14]} for s, c in zip(STEMS, ("control", "drug"))]
    ).to_csv(tmp_path / "strata.csv", index=False)
    export = tmp_path / "export"
    export.mkdir()
    for stem in STEMS:
        coords = np.zeros((len(WELLS), N_SLOTS, 7, 2), dtype=np.float32)
        coords[..., 0] = 100.0 + np.arange(len(WELLS))[:, None, None] * 50
        coords[..., 1] = 80.0
        conf = np.ones((len(WELLS), N_SLOTS, 7), dtype=np.float32)
        conf[:, 3, :] = 0.0  # filled in by stage 1
        np.savez(export / f"{stem}.npz", coords=coords, conf=conf, wells=np.array(WELLS))
    return tmp_path


def _import(bundle, **kwargs):
    return import_tadpose_sleap(
        TadPoseImport(
            bundle_dir=bundle / "bundle",
            labellings={"pdv": bundle / "pdv.npy", "vel": bundle / "vel.npy"},
            crop_csv=bundle / "crop.csv",
            project_root=bundle / "project",
            catalogues={"pdv": bundle / "pdv.json"},
            animals_csv=bundle / "animals.csv",
            video_strata_csv=bundle / "strata.csv",
            pose_dir=bundle / "export",
            **kwargs,
        )
    )


def test_tadpose_project(bundle):
    project = _import(bundle)
    assert set(project.labellings) == {"pdv", "vel"}
    assert project.strata == ("condition", "plate")
    video = project.videos[STEMS[0]]
    assert video.path == "plateA_001.mp4" and video.strata["condition"] == "control"
    frame_map = project.load_frame_map(STEMS[0])
    assert frame_map[DROPPED_SLOT] == -1
    assert frame_map[0] == SKIP and frame_map[DROPPED_SLOT + 1] == DROPPED_SLOT + SKIP
    assert frame_map[5] == 5 + SKIP  # from the omitted rows
    animal = "plateA_A2"
    box = video.boxes[animal]
    assert (box.x + box.w // 2, box.y + box.h // 2) == (150 + 200, 80 + 60)  # crop offset added
    pose = PoseStore(project).load_slice(STEMS[0], animal, 0, 5)
    np.testing.assert_allclose(pose.coords[0, 2], [350, 140])
    assert not pose.visible(project.skeleton.min_confidence)[3].any()  # filled points hidden
    meta = project.load_label_meta("pdv")
    assert meta["0"].name == "pdv.01" and meta["0"].description == "forward"
    bouts = project.load_bouts("pdv")
    assert set(bouts["animal"].astype(str)) == {"plateA_A1", "plateA_A2", "plateB_A1", "plateB_A2"}
    assert set(bouts["condition"].astype(str)) == {"control", "drug"}
    across = bouts[(bouts.start < DROPPED_SLOT) & (bouts.stop > DROPPED_SLOT)]
    assert across.empty  # a dropped slot is missing from the table, so it ends a bout


def test_tadpose_errors(bundle):
    np.save(bundle / "short.npy", np.zeros(3, dtype=np.int32))
    with pytest.raises(ProjectError, match="labels for"):
        import_tadpose_sleap(
            TadPoseImport(
                bundle_dir=bundle / "bundle",
                labellings={"x": bundle / "short.npy"},
                crop_csv=bundle / "crop.csv",
                project_root=bundle / "p2",
            )
        )
    pd.DataFrame({"video": ["/x/other.mp4"], "x": [0], "y": [0], "w": [1], "h": [1]}).to_csv(
        bundle / "crop2.csv", index=False
    )
    with pytest.raises(ProjectError, match="no crop box"):
        import_tadpose_sleap(
            TadPoseImport(
                bundle_dir=bundle / "bundle",
                labellings={"x": bundle / "pdv.npy"},
                crop_csv=bundle / "crop2.csv",
                project_root=bundle / "p3",
            )
        )


def test_kpms_adds_a_labelling(bundle):
    h5py = pytest.importorskip("h5py")
    _import(bundle)
    with h5py.File(bundle / "results.h5", "w") as results:
        for stem in STEMS:
            for well in WELLS:
                results.create_group(f"{stem}__{well}").create_dataset(
                    "syllable", data=np.repeat([4, 1, 4], 5)
                )
    pd.DataFrame(
        [
            {"recording": f"{s}__{w}", "video": s, "animal": f"{s[8:14]}_{w}", "first_frame": 10}
            for s in STEMS
            for w in WELLS
        ]
    ).to_csv(bundle / "map.csv", index=False)
    project = import_keypoint_moseq(
        KpmsImport(bundle / "results.h5", bundle / "map.csv", bundle / "project")
    )
    bouts = project.load_bouts("moseq")
    assert len(bouts) == 3 * len(STEMS) * len(WELLS)
    first = bouts.iloc[0]
    assert (first.start, first.stop, str(first.label)) == (10, 15, "4")
    assert (first.video_frame_start, first.video_frame_stop) == (20, 25)  # via the frame map
    assert set(bouts["condition"].astype(str)) == {"control", "drug"}  # from videos.toml
    assert load_project(bundle / "project").load_label_meta("moseq")["1"].name == "syllable 1"
    pd.DataFrame(
        {"recording": ["x"], "video": [STEMS[0]], "animal": ["a"], "first_frame": [0]}
    ).to_csv(bundle / "bad.csv", index=False)
    with pytest.raises(ProjectError, match="no recording"):
        import_keypoint_moseq(
            KpmsImport(
                bundle / "results.h5", bundle / "bad.csv", bundle / "project", labelling="m2"
            )
        )


def test_cli_tadpose(bundle, capsys):
    code = main(
        [
            "import",
            "tadpose-sleap",
            str(bundle / "bundle"),
            str(bundle / "cli_project"),
            "--labels",
            f"pdv={bundle / 'pdv.npy'}",
            "--crop-csv",
            str(bundle / "crop.csv"),
            "--catalogue",
            f"pdv={bundle / 'pdv.json'}",
        ]
    )
    assert code == 0 and "pdv:" in capsys.readouterr().out
    assert (
        main(
            [
                "import",
                "tadpose-sleap",
                str(bundle / "bundle"),
                str(bundle / "p9"),
                "--labels",
                "broken",
                "--crop-csv",
                str(bundle / "crop.csv"),
            ]
        )
        == 1
    )
