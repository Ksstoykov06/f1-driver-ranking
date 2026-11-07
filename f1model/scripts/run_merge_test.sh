#!/bin/bash
cd /home/claude/f1model; export F1_MERGE_WET=1
[ -f outputs_v2/B80m_synth.csv ] || python3 run_v2.py synth B80m
[ -f outputs_v2/B80m_holdout.csv ] || python3 run_v2.py holdout B80m
cat outputs_v2/B80m_synth.csv; grep ",all," outputs_v2/B80m_holdout.csv
