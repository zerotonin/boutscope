Project format
==============

A project is a folder.  Adapters write it; the GUI reads nothing else.
This page is the contract.

::

    project/
    ├── project.toml                    name, fps, adapter, labellings, strata, skeleton
    ├── videos.toml                     per video: path, fps, frame offset, boxes
    ├── labels/<labelling>.parquet      one row per labelled frame
    ├── labels/<labelling>.meta.toml    name and colour per label
    ├── bouts/<labelling>.parquet       bout index, written at import
    ├── framemap/<video>.npy            optional: video frame of every project frame
    └── pose/<video>/                   optional keypoints (coords.npy, conf.npy, meta.toml)

A project holds one or more *labellings* of the same frames, e.g. two
k-means runs and a keypoint-MoSeq model.

Label table
-----------

One row per labelled frame.

==================  ========  =============================================
Column              Required  Meaning
==================  ========  =============================================
``video``           yes       Key into ``videos.toml``.
``animal``          yes       Animal identity.  The same id in two videos
                              means the same individual.
``frame``           yes       Integer time index on a uniform grid at the
                              project fps.  Bouts need consecutive values.
``label``           yes       The categoriser's output (stored as text).
``video_frame``     no        Frame index in the video file, when neither a
                              frame map nor ``frame + frame_offset`` gives it.
``typicality``      no        Higher = more typical of its label (e.g. minus
                              the distance to a k-means centre).  Enables the
                              typical / atypical sampling modes.
stratum columns     no        Any grouping (``condition``, ``genotype``, ...)
                              named in ``project.toml`` ``strata``.
==================  ========  =============================================

Rules:

* ``(video, animal, frame)`` is unique.
* Labels are taken exactly as given.  Merge short bouts or drop frames
  *before* import if you want that.
* A frame missing from the table ends a bout.

Bouts
-----

A bout is the longest unbroken run of one label for one animal in one
video over consecutive ``frame`` values.  The bout table holds
``bout_id``, ``video``, ``animal``, ``label``, ``start``, ``stop``
(exclusive), ``n_frames``, ``duration_s``, ``video_frame_start``,
``video_frame_stop``, the stratum columns (value at the first frame) and
the mean ``typicality``.  Bout ids depend only on the table's content, not
on its row order.

videos.toml
-----------

.. code-block:: toml

    [videos.plate1]
    path = "2024/plate1.mp4"      # relative to video_root
    fps = 50.0                    # true rate; overrides the file header
    frame_offset = 10             # video_frame = frame + frame_offset
    width = 1280
    height = 720

    [videos.plate1.boxes.A1]      # region showing animal A1, video pixels
    x = 64
    y = 40
    w = 110
    h = 110

Frame maps
----------

``framemap/<video>.npy`` (int64) holds the video frame of every project
frame 0…N−1, −1 where the video has none (a camera-dropped frame).  It
belongs to the video, so every labelling uses it; a clip reads the exact
video frames of its project frames and repeats the previous frame where
one was dropped.  Write it with :func:`boutscope.project.write_frame_map`.

The video frame of a project frame comes from, in this order: the label
table's ``video_frame`` column, the frame map, ``frame + frame_offset``.

Per-video strata
----------------

A video entry may carry stratum values::

    [videos.plate1.strata]
    condition = "PTZ"
    plate = "ptz2"

A label table without a stratum column takes the value of its video, so
labellings from other tools (e.g. keypoint-MoSeq) need not repeat them.

Label meta
----------

.. code-block:: toml

    [labels.3]
    name = "forward swim"
    colour = "#009E73"
    description = "optional"

Labels without an entry get their own text as name and a Wong colour.

project.toml
------------

.. code-block:: toml

    format_version = 1
    name = "sept2024"
    fps = 50.0
    adapter = "generic"
    created = "2026-10-28T00:00:00+00:00"
    strata = ["condition"]
    context_s = 0.5

    [labellings.pdv_k34]
    n_frames = 51483468
    n_bouts = 30000000
    n_labels = 34
    content_hash = "..."          # SHA-256 of the canonical label table
    has_typicality = false
    has_video_frame = true

    [skeleton]
    keypoints = ["left_eye", "right_eye", "centre", "tail1"]
    edges = [["centre", "tail1"]]
    min_confidence = 0.2

Pose
----

Optional, per video, for the skeleton overlay::

    pose/<video>/coords.npy    float32 (animals, frames, keypoints, 2), video pixels
    pose/<video>/conf.npy      float32 (animals, frames, keypoints)
    pose/<video>/meta.toml     animals = [...] (row order), frame0 = <project frame of row 0>

Keypoints follow the order of ``[skeleton] keypoints`` in ``project.toml``.
Frames are on the project ``frame`` grid.  Set the confidence of anything
the tracker did not observe (interpolated, filled in) below
``min_confidence`` and it will not be drawn.  ``.npy`` rather than
``.npz`` so the arrays are memory-mapped.  Write them with
:func:`boutscope.pose.write_pose`.

Sampling
--------

For a label and a duration range, boutscope builds the full draw order
once: within each stratum the bouts are shuffled, the strata take turns
(round-robin), and an optional cap limits bouts per animal.  The random
stream comes from the user's seed and a hash of the request, so the same
request gives the same order on every machine, and "show 20 more" never
repeats a bout.  Typical / atypical modes first keep only the top /
bottom typicality decile of the candidates.
