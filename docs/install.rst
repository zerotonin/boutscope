Installation
============

1. Set up your paths first
--------------------------

boutscope keeps every machine-specific path out of the code.  Before
anything else, copy the template and fill it in::

    cp local_paths.template.json local_paths.json

For an installed package, put the file at
``~/.config/boutscope/local_paths.json`` instead, or point the
``BOUTSCOPE_PATHS`` environment variable at it.

The keys that matter:

``video_root``
    The folder the video paths in a project's ``videos.toml`` are
    relative to.  A network folder works the same as a local one.

``project_root``
    The folder that holds your boutscope projects.

``BOUTSCOPE_VIDEO_ROOT`` and ``BOUTSCOPE_PROJECT_ROOT`` override the file;
``BOUTSCOPE_PROFILE`` selects a profile.  A command that needs a path that
is not configured stops with a message naming the template.

2. Install
----------

With conda::

    conda env create -f environment.yml
    conda activate boutscope

or with pip into an existing environment::

    pip install -e ".[dev]"

3. Check
--------

::

    pytest
    boutscope --help
