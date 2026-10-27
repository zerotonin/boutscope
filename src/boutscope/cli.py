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
        parser.print_help()
        return 0
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
    return parser


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
