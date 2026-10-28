Adapters
========

An adapter turns a categoriser's output into a project (see
:doc:`project_format`).  After any import, run::

    boutscope check <project> --labelling <name>

It checks that direct seeks land on the frames sequential decoding gives,
and that the keypoints sit on the animal: the video is shifted against the
pose by −15…+15 frames, and the darkest shift under the keypoints must be 0.

Generic table
-------------

Any categoriser that can write one row per labelled frame::

    boutscope import generic labels.csv myproject --labelling km36 --fps 50

TadPose SLEAP
-------------

Reads a TadPose SLEAP feature bundle directly (boutscope does not import
TadPose)::

    boutscope import tadpose-sleap <bundle> <project> \
        --labels pdv_k34=labels_pdv.npy --labels vel_k23=labels_vel.npy \
        --catalogue pdv_k34=catalogue_pdv.json --catalogue vel_k23=catalogue_vel.json \
        --crop-csv crop_positions.csv --animals animals.csv \
        --video-strata video_strata.csv --pose-dir moseq/export

* ``slot`` (the uniform time grid) becomes the project frame; slot → raw
  video frame, from ``rows.parquet`` and ``omitted_rows.parquet``, becomes
  each video's frame map, so camera-dropped frames are handled exactly.
* SLEAP tracked a cropped plate; keypoints and animal boxes are shifted by
  the crop offset so the viewer can read the raw videos.
* Keypoints come from the MoSeq export (confidence 0 wherever stage 1
  filled a point in), so interpolated points are not drawn.
* ``--video-strata`` is a CSV with a ``video`` column and one column per
  stratum (e.g. condition, plate); the values are stored per video.

keypoint-MoSeq
--------------

Adds the syllables of a fitted kpms model to an existing project::

    pip install h5py   # or: pip install "boutscope[kpms]"
    boutscope import keypoint-moseq results.h5 <project> \
        --recording-map recordings.csv --labelling moseq

The recording map places each kpms recording on the project grid:
``recording, video, animal, first_frame``.  Animal ids must be the ones the
project already uses, so the strips line up across labellings.
