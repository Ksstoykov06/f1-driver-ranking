#!/bin/bash
cd /home/claude/f1model; export PYTHONPATH=.
for v in "C66 1966" "C61 1961"; do set -- $v
 [ -f outputs_v2/$1_scores.csv ] || python3 run_v2.py $1
 [ -f outputs_v2/$1_beyond_car.csv ] || python3 scripts/beyond_car.py $1 $2 > /dev/null
 [ -f outputs_v2/$1_holdout.csv ] || python3 run_v2.py holdout $1
done; echo finished
