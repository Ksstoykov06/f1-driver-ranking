#!/bin/bash
cd /home/claude/f1model
export LINK_W=800 LINK_T=8 LINK_SUF=_long PYTHONPATH=. MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=0
[ -f outputs_v2/gap/base_long_fullscores.csv ] || LINK_FULL=1 python3 scripts/link_test.py base
