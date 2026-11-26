# Vendored NGFF spec sources

Unmodified copies of the NGFF specification source files, and of the JSON schemas that
rules cite, at the commits pinned in
`src/yaozarrs/_rules/rules.toml` (`[spec.*]`). `tests/test_rules.py` uses them to
check that every rule's quote is at the recorded lines, without network access.

Refresh with `uv run python scripts/ngff_rules.py fetch`.

These are the files that https://ngff.openmicroscopy.org publishes: its source repo,
`ome/ngff`, includes each spec version as a submodule of `ome/ngff-spec`, and renders
the version's `index.md`. `uv run python scripts/ngff_rules.py site` checks that the
pins still match.

Copyright © the OME-NGFF contributors.
