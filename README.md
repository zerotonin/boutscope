# boutscope

[![Tests](https://github.com/zerotonin/boutscope/actions/workflows/tests.yml/badge.svg)](https://github.com/zerotonin/boutscope/actions/workflows/tests.yml)
[![Docs](https://github.com/zerotonin/boutscope/actions/workflows/docs.yml/badge.svg)](https://zerotonin.github.io/boutscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Ground truth for behavioural categorisers.**

Clustering and segmentation tools (k-means prototype movements,
keypoint-MoSeq, B-SOiD, VAME, SimBA, ...) label every frame of a video.
boutscope shows you what those labels actually contain: it indexes the
labels into bouts, draws seeded and stratified samples of bouts in a
duration range you choose, plays them in the original video, and records
your verdict on each bout. From the verdicts it estimates how often each
label is right, with a confidence interval.

> **Status: 0.1, core only.** The project format, the generic adapter, the
> bout index and the sampler work from the command line. The viewer, the
> annotation GUI and the adapters for keypoint-MoSeq and TadPose follow.

## Install

1. **Set up your paths first.** Copy the template and fill in `video_root`
   and `project_root`:

   ```bash
   cp local_paths.template.json local_paths.json
   ```

   (For an installed package: `~/.config/boutscope/local_paths.json`, or set
   `BOUTSCOPE_PATHS`.)

2. Install:

   ```bash
   conda env create -f environment.yml
   conda activate boutscope
   ```

## Quick start

```bash
# one row per labelled frame: video, animal, frame, label (+ optional columns)
boutscope import generic labels.csv myproject --labelling km36 --fps 50
boutscope info myproject --labelling km36
boutscope sample myproject --labelling km36 --label 7 --min-duration 0.1 -n 20
```

The project format is documented in [docs/project_format.rst](docs/project_format.rst).

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT
