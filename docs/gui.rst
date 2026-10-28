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

Rating bouts
------------

The first time you open a project, boutscope asks who is annotating; the
name is stored with every verdict (*Session → Change annotator* switches
it).  Every **Draw** is stored as a session in ``boutscope.sqlite`` inside
the project, with its full draw order.

For each bout give one verdict:

========  ==============  ==================================================
key       verdict         when
========  ==============  ==================================================
1         fits            the bout is what the label stands for
2         doesn't fit     it is something else (say what in *looks like*)
3         artefact        a tracking or labelling failure, not behaviour
4         unsure          cannot tell
========  ==============  ==================================================

A note is optional.  Rated bouts get a coloured dot in the list; with
*next bout after a verdict* ticked the viewer moves on by itself.  Below
the verdicts you can propose a **name for the label**; every proposal is
kept, the latest per annotator counts.

**Blind sessions** (*Session → New blind session*) mix bouts of two or
more labels in equal shares.  The label colour and the strips are hidden;
answer with the label you think it is (keys 1–9), **0** for *none of
these*, **A** for artefact, **U** for unsure.  The true label is shown
after you answer.

*Session → Resume session* reopens any stored session at its first
unrated bout; "More" then continues the same order.  A session whose
labelling has changed since (different content hash) cannot be resumed,
because its bout ids would point at other bouts.

*Session → Export verdicts as CSV* writes ``verdicts.csv``,
``sessions.csv`` and ``label_names.csv``.

Keys
----

==========  ==================================
Space       play / pause
← / →       one frame back / forward
N / B       next / previous bout
L           loop on / off
\+ / −      faster / slower
1–4         verdict (blind: 1–9 answer, 0 none, A, U)
==========  ==================================

Clips are read in a background thread and the next two bouts are
prefetched.  Bouts longer than 30 s are shown from their start.  A clip
that would need more than 512 MB of memory is downscaled.
