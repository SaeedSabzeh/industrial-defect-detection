# Notebooks (archival)

These are the two original submitted notebooks, kept for provenance.
`01_original_width8.ipynb` and `02_original_width64.ipynb` are ~95% identical
code differing only in channel width — the duplication that `src/models.py`
now replaces with a single `width` parameter.

**Their results are not valid.** Both use the ordered CSV split that leaked
augmented variants across the train/test boundary (see the main README).
They are here to document what was originally run, not to be cited.

Embedded figures were extracted to `../assets/`; the pair went from 15.0 MB to
0.16 MB. Training logs are preserved.

Use `python -m src.main` for anything you intend to trust.
