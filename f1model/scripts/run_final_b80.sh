#!/bin/bash
cd /home/claude/f1model; export PYTHONPATH=.
[ -f outputs_v2/B80_scores.csv ] || python3 run_v2.py B80
[ -f outputs_v2/B80_beyond_car.csv ] || python3 scripts/beyond_car.py B80 1966 > /dev/null
[ -f outputs_v2/B80_holdout.csv ] || python3 run_v2.py holdout B80
echo finished
