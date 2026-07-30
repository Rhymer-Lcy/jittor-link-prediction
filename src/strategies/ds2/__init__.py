# -*- coding: utf-8 -*-
"""dataset2 strategies.

The dataset2 base score matrix is produced by the ranker + basket + row-order
chain already tracked in ``src/`` (``ranker_basket_ds2.py``, ``crf_promote.py``)
with the MF basket geometry in ``src/ds2_mf_basket_pack.py``. Those are
model-side mechanisms, not postprocessors.

One frozen dataset2 score postprocessor is graduated here:
:mod:`cross_time_exclusivity`, the final decode that the accepted A-board member
carries on top of the CRF output.
"""

from . import cross_time_exclusivity  # noqa: F401
