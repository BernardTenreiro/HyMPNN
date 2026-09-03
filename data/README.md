# Data directory

Place the QM9 source files in `data/raw/qm9/`. Prepared NumPy splits are cached
in `data/processed/qm9/` on first use. Dataset contents are intentionally not
tracked by Git.

Required source files:

- `dsgdb9nsd.xyz.tar.bz2`
- `uncharacterized.txt`

`atomref.txt` is optional and is downloaded when thermochemical targets are
needed.
