# ╔══════════════════════════════════════════════════════════════════╗
# ║  boutscope — cli                                                 ║
# ║  « import, inspect, sample »                                     ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  The boutscope command: boutscope import <adapter>,              ║
# ║  boutscope info, boutscope sample.                               ║
# ╚══════════════════════════════════════════════════════════════════╝
"""``boutscope`` command line: import label tables, inspect a project, draw bouts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from boutscope import __version__
from boutscope import constants as bc
from boutscope.adapters.generic import ColumnMap, GenericImport, import_generic
from boutscope.bouts import summarise_labels
from boutscope.project import ProjectError, load_project
from boutscope.sampling import (
    SampleRequest,
    SamplingError,
    count_by_stratum,
    select_candidates,
    start_session,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "run"):
        return run_gui(argparse.Namespace(project=None, video_root=None))
    try:
        args.run(args)
    except (ProjectError, SamplingError) as error:
        print(f"boutscope: {error}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="boutscope", description=__doc__)
    parser.add_argument("--version", action="version", version=f"boutscope {__version__}")
    commands = parser.add_subparsers(title="commands")

    importer = commands.add_parser("import", help="add a label table to a project")
    adapters = importer.add_subparsers(title="adapters", required=True)
    generic = adapters.add_parser("generic", help="CSV / Parquet, one row per labelled frame")
    generic.add_argument("table", type=Path)
    generic.add_argument("project", type=Path, help="project folder (created if missing)")
    generic.add_argument("--labelling", required=True, help="name of this labelling, e.g. pdv_k34")
    generic.add_argument("--fps", type=float, help="true frame rate (needed for a new project)")
    for field_name in ("video", "animal", "frame", "label"):
        generic.add_argument(
            f"--{field_name}-col", default=field_name, help=f"column holding {field_name}"
        )
    generic.add_argument("--videos", type=Path, help="videos.toml with paths, fps, offsets, boxes")
    generic.add_argument("--meta", type=Path, help="CSV with label, name, colour[, description]")
    generic.add_argument(
        "--strata", nargs="*", default=[], help="stratum columns (new project only)"
    )
    generic.add_argument("--name", help="project name (default: folder name)")
    generic.add_argument("--overwrite", action="store_true", help="replace an existing labelling")
    generic.set_defaults(run=run_import_generic)

    tadpose = adapters.add_parser("tadpose-sleap", help="TadPose SLEAP feature bundle")
    tadpose.add_argument("bundle", type=Path, help="folder with rows.parquet")
    tadpose.add_argument("project", type=Path, help="new project folder")
    tadpose.add_argument(
        "--labels",
        action="append",
        required=True,
        metavar="NAME=NPY",
        help="a labelling: name and label array (repeatable)",
    )
    tadpose.add_argument(
        "--catalogue",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="PM catalogue with names and colours for a labelling",
    )
    tadpose.add_argument(
        "--crop-csv",
        type=Path,
        required=True,
        help="video,x,y,w,h of the SLEAP crop in the raw video",
    )
    tadpose.add_argument("--animals", type=Path, help="animals.csv (recording, animal)")
    tadpose.add_argument("--video-strata", type=Path, help="CSV: video + stratum columns")
    tadpose.add_argument("--pose-dir", type=Path, help="MoSeq export npz per video")
    tadpose.add_argument("--fps", type=float, default=50.0)
    tadpose.add_argument("--box-px", type=int, default=140)
    tadpose.add_argument("--name", help="project name (default: folder name)")
    tadpose.set_defaults(run=run_import_tadpose)

    kpms = adapters.add_parser("keypoint-moseq", help="kpms results.h5 into an existing project")
    kpms.add_argument("results", type=Path, help="results.h5")
    kpms.add_argument("project", type=Path)
    kpms.add_argument(
        "--recording-map",
        type=Path,
        required=True,
        help="CSV: recording, video, animal, first_frame",
    )
    kpms.add_argument("--labelling", default="moseq")
    kpms.add_argument("--overwrite", action="store_true")
    kpms.set_defaults(run=run_import_kpms)

    info = commands.add_parser("info", help="labellings, labels and bout counts of a project")
    info.add_argument("project", type=Path)
    info.add_argument("--labelling", help="show the label summary of one labelling")
    info.set_defaults(run=run_info)

    sample = commands.add_parser("sample", help="draw bouts as the GUI would, print them")
    sample.add_argument("project", type=Path)
    sample.add_argument("--labelling", required=True)
    sample.add_argument("--label", required=True)
    sample.add_argument("--min-duration", type=float, help="seconds")
    sample.add_argument("--max-duration", type=float, help="seconds")
    sample.add_argument("--strata", nargs="*", default=list(bc.DEFAULT_STRATA))
    sample.add_argument("--mode", choices=bc.SAMPLING_MODES, default="stratified")
    sample.add_argument("--seed", type=int, default=bc.DEFAULT_SEED)
    sample.add_argument("--max-per-animal", type=int)
    sample.add_argument("-n", type=int, default=20, help="bouts to print")
    sample.set_defaults(run=run_sample)

    check = commands.add_parser("check", help="exact seeking and pose alignment on real videos")
    check.add_argument("project", type=Path)
    check.add_argument("--labelling", help="labelling whose bouts test the alignment")
    check.add_argument("--video-root", type=Path)
    check.add_argument("--videos", type=int, default=1, help="videos to test seeking on")
    check.set_defaults(run=run_check)

    gui = commands.add_parser("gui", help="open the viewer (the default without a command)")
    gui.add_argument("project", type=Path, nargs="?")
    gui.add_argument("--video-root", type=Path, help="overrides video_root from local_paths.json")
    gui.set_defaults(run=run_gui)

    demo = commands.add_parser("demo", help="build a small synthetic project with videos")
    demo.add_argument("folder", type=Path)
    demo.add_argument("--open", action="store_true", help="open it in the viewer")
    demo.set_defaults(run=run_demo)
    return parser


def run_gui(args: argparse.Namespace) -> int:
    from boutscope.gui.app import main as gui_main

    return gui_main(args.project, args.video_root)


def run_demo(args: argparse.Namespace) -> None:
    from boutscope.synthetic import make_demo_project

    project = make_demo_project(args.folder)
    print(f"demo project -> {project.root}")
    if args.open:
        run_gui(argparse.Namespace(project=project.root, video_root=None))


def run_import_generic(args: argparse.Namespace) -> None:
    columns = ColumnMap(
        video=args.video_col, animal=args.animal_col, frame=args.frame_col, label=args.label_col
    )
    project = import_generic(
        GenericImport(
            table_path=args.table,
            project_root=args.project,
            labelling=args.labelling,
            fps=args.fps,
            columns=columns,
            videos_toml=args.videos,
            meta_csv=args.meta,
            strata=tuple(args.strata),
            project_name=args.name,
            overwrite=args.overwrite,
        )
    )
    info = project.labellings[args.labelling]
    print(
        f"{args.labelling}: {info.n_frames} frames, {info.n_labels} labels, {info.n_bouts} bouts "
        f"-> {project.root}"
    )


def run_import_tadpose(args: argparse.Namespace) -> None:
    from boutscope.adapters.tadpose_sleap import TadPoseImport, import_tadpose_sleap

    project = import_tadpose_sleap(
        TadPoseImport(
            bundle_dir=args.bundle,
            labellings=_pairs(args.labels),
            crop_csv=args.crop_csv,
            project_root=args.project,
            catalogues=_pairs(args.catalogue),
            animals_csv=args.animals,
            video_strata_csv=args.video_strata,
            pose_dir=args.pose_dir,
            fps=args.fps,
            box_px=args.box_px,
            name=args.name,
        )
    )
    for name, info in project.labellings.items():
        print(f"{name}: {info.n_frames} frames, {info.n_labels} labels, {info.n_bouts} bouts")
    print(f"-> {project.root}")


def run_import_kpms(args: argparse.Namespace) -> None:
    from boutscope.adapters.keypoint_moseq import KpmsImport, import_keypoint_moseq

    project = import_keypoint_moseq(
        KpmsImport(
            results_h5=args.results,
            recording_map=args.recording_map,
            project_root=args.project,
            labelling=args.labelling,
            overwrite=args.overwrite,
        )
    )
    info = project.labellings[args.labelling]
    print(
        f"{args.labelling}: {info.n_frames} frames, {info.n_labels} syllables, "
        f"{info.n_bouts} bouts -> {project.root}"
    )


def _pairs(items: list[str]) -> dict[str, Path]:
    pairs = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            raise ProjectError(f"expected NAME=PATH, got {item!r}")
        pairs[name] = Path(path)
    return pairs


def run_check(args: argparse.Namespace) -> None:
    from boutscope.checks import check_alignment, check_seeking, format_alignment
    from boutscope.video import resolve_video_path

    project = load_project(args.project)
    for video in list(project.videos)[: args.videos]:
        report = check_seeking(resolve_video_path(project, video, args.video_root))
        state = "exact" if not report.wrong else f"WRONG at {report.wrong[:5]}"
        print(f"seeking {report.video}: {report.n_seeks} seeks, {state}")
    if args.labelling:
        print(
            format_alignment(check_alignment(project, args.labelling, video_root=args.video_root))
        )


def run_info(args: argparse.Namespace) -> None:
    project = load_project(args.project)
    print(
        f"{project.name}  ({project.adapter}, {project.fps:g} fps, {len(project.videos)} videos, "
        f"strata: {', '.join(project.strata) or 'none'})"
    )
    for name, info in project.labellings.items():
        print(
            f"  {name}: {info.n_labels} labels, {info.n_bouts} bouts, {info.n_frames} frames"
            + (", typicality" if info.has_typicality else "")
        )
    if args.labelling:
        meta = project.load_label_meta(args.labelling)
        summary = summarise_labels(project.load_bouts(args.labelling))
        summary.insert(0, "name", [meta[str(label)].name for label in summary.index])
        with pd.option_context("display.width", 120, "display.max_rows", 500):
            print(summary.round(3).to_string())


def run_sample(args: argparse.Namespace) -> None:
    project = load_project(args.project)
    bouts = project.load_bouts(args.labelling)
    request = SampleRequest(
        labelling=args.labelling,
        label=args.label,
        min_duration_s=args.min_duration,
        max_duration_s=args.max_duration,
        strata=tuple(args.strata),
        mode=args.mode,
        seed=args.seed,
        max_per_animal=args.max_per_animal,
    )
    counts = count_by_stratum(select_candidates(bouts, request), request.strata)
    session = start_session(bouts, request)
    print(f"{len(session.order)} bouts available in {len(counts)} strata (seed {request.seed})")
    shown = bouts.set_index("bout_id").loc[session.next_batch(args.n)]
    columns = ["video", "animal", "start", "stop", "duration_s", *request.strata]
    print(shown[list(dict.fromkeys(columns))].to_string())


if __name__ == "__main__":
    sys.exit(main())
