# -*- coding: utf-8 -*-
"""dataset2 strategies.

The shipped dataset2 member is produced by the ranker + basket + row-order chain
already tracked in ``src/`` (``ranker_basket_ds2.py``, ``crf_promote.py``) with
the MF basket geometry in ``src/ds2_mf_basket_pack.py``. No frozen dataset2
score postprocessor has been graduated into this package yet -- the dataset2
mechanisms are model-side, not postprocessors.
"""
