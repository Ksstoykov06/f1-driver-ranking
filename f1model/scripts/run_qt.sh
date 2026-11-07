#!/bin/bash
cd /home/claude/f1model
export MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=0  # glibc arenas otherwise grow about 90 MB a minute and hit the 6 GB limit at 40 minutes
export LINK_W=800 LINK_T=8 LINK_SUF=_long PYTHONPATH=.
[ -f outputs_v2/gap/qt_long_pairs.pkl ] || python3 scripts/link_test.py qt
[ -f outputs_v2/gap/qt_long_fullscores.csv ] || LINK_FULL=1 python3 scripts/link_test.py qt
