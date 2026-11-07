#!/bin/bash
cd /home/claude/f1model; export PYTHONPATH=.
[ -f outputs_v2/B80_synth.csv ] || python3 run_v2.py synth B80
python3 run_v2.py sens80
echo finished
