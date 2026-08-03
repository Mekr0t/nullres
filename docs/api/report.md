# Rendering

Every function here takes a result object and returns a string. Nothing
computes, reads a file, or prints — a test fails if `report.py` so much as
mentions `load_bars` or `open(`.

Commands that take minutes have their output split in two: `*_header` renders
what is known before the work starts, so it can be shown while the user waits,
and `format_*` renders the whole report for a caller that just wants the text.

::: nullres.report
