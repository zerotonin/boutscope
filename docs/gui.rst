Viewer
======

Start it with ``boutscope`` (or ``boutscope gui <project>``).  To try it
without data::

    boutscope demo /tmp/bs_demo --open

Workflow
--------

1. **Pick a labelling and a label** in the list on the left.  The table
   shows each label's share of frames, number of bouts and median bout
   duration.
2. **Choose a duration range** by dragging the shaded region over the
   bout-duration histogram (log axis).  Below it: how many bouts are in
   range, overall and per stratum.
3. **Set the sample**: stratify by animal, video or any stratum column of
   the project; choose the mode (typical / atypical need a typicality
   score); the number of bouts, the seed and an optional cap per animal.
   **Draw** builds the seeded order; **More** continues it without
   repeating a bout.
4. **Watch.**  Each bout plays zoomed on its animal, with ±0.5 s of
   context.  Context frames have a dashed grey border, bout frames a solid
   border in the label's colour.  The skeleton shows only keypoints at or
   above the project's confidence threshold.  Under the video, the timeline
   shows the bout and one strip per labelling for the same frames; hover
   for the labels, click or drag to seek.

Keys
----

==========  ==================================
Space       play / pause
← / →       one frame back / forward
N / B       next / previous bout
L           loop on / off
\+ / −      faster / slower
==========  ==================================

Clips are read in a background thread and the next two bouts are
prefetched.  Bouts longer than 30 s are shown from their start.  A clip
that would need more than 512 MB of memory is downscaled.
