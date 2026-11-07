#!/usr/bin/env bash
# Restartable: stages with finished outputs are skipped.
export PYTHONPATH=.; O=outputs_v2; mkdir -p $O/logs $O/ckpt
s(){ local out="$1"; shift; [ -f "$O/$out" ] || python3 run_v2.py "$@" > "$O/logs/${out}.log" 2>&1; }
s A18_adapt.csv A18; s B18_adapt.csv B18
s A18_holdout.csv holdout A18; s B18_holdout.csv holdout B18
s A18_synth.csv synth A18; s B18_synth.csv synth B18
s B80_adapt.csv B80
s B80_holdout.csv holdout B80; s B80_synth.csv synth B80
s sensitivity.csv sens

[ -f $O/B80_beyond_car.csv ] || python3 scripts/beyond_car.py > $O/logs/beyond.log 2>&1
