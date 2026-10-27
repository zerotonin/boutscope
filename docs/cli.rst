Command line
============

Import a label table (CSV, TSV or Parquet) into a new or existing project::

    boutscope import generic labels.csv myproject --labelling km36 --fps 50 \
        --video-col movie --label-col cluster --strata condition \
        --videos videos.toml --meta label_names.csv

Inspect a project::

    boutscope info myproject --labelling km36

Draw bouts as the GUI will::

    boutscope sample myproject --labelling km36 --label 7 \
        --min-duration 0.1 --max-duration 0.5 --strata condition -n 20

The label meta CSV has the columns ``label, name, colour`` and optionally
``description``.
