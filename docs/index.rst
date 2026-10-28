boutscope
=========

Ground truth for behavioural categorisers.

Clustering and segmentation tools (k-means prototype movements,
keypoint-MoSeq, B-SOiD, VAME, SimBA, ...) label every video frame, but a
label is only useful once someone has looked at what it contains.
boutscope indexes any categoriser's labels into bouts, draws seeded and
stratified samples of bouts in a duration range you choose, shows them in
the original video, and records a verdict per bout.  From the verdicts it
estimates how often each label is right, with a confidence interval.

.. note::

   Version 0.2 contains the project format, the generic adapter, the
   bout index, the sampler and the viewer.  Annotation (verdicts,
   sessions, blind mode) follows.

.. toctree::
   :maxdepth: 2

   install
   gui
   project_format
   cli
   api
