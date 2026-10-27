"""Building blocks shared by more than one OME-NGFF spec version.

CONTRACT: everything here is part of *every* version that uses it, including
released (frozen) versions, so changing behavior here changes those versions too.

- Only put code here if the governing spec text is identical across the versions
  that use it, AND its current behavior is correct for all of them.
- Never branch on version (no version string literals here).  If a newer version
  needs different behavior, fork the class into that version's package instead
  ("fork, don't flag").
- Never import from version packages (`yaozarrs.v04`, `yaozarrs.v05`, ...).

Guarded by tests/test_frozen_schemas.py and tests/test_architecture.py.
"""
