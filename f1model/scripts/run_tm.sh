#!/bin/bash
# remove-a-teammate rerun: run_tm.sh x y  (full-data fit, same settings as base_long)
cd /home/claude/f1model
export LINK_W=800 LINK_T=8 LINK_SUF=_$1__$2 LINK_FULL=1 TM_DROP=$1,$2 PYTHONPATH=. MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=0
date; python3 scripts/link_test.py tm; date
