#!/bin/bash
export PYTHONPATH=.
for s in 21 22; do for b in 0 1; do [ -f outputs_v2/dominant/s${s}_b${b}_scores.csv ] || python3 scripts/dominant_test.py $s $b; done; done
