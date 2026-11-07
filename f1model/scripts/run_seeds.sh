#!/bin/bash
cd /home/claude/f1model
for s in 12 13; do [ -f outputs_v2/B80s${s}_synth.csv ] || SYNTH_SEED=$s python3 run_v2.py synth B80s$s; cat outputs_v2/B80s${s}_synth.csv; done
