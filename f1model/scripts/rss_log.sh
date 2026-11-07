#!/bin/bash
# Logs memory of running fits once a minute, to find what ran out of memory.
while true; do echo "$(date +%T) $(ps -C python3 -o rss=,etimes=,args= | cut -c1-70 | tr '\n' ';')" >> outputs_v2/gap/rss.log; sleep 60; done
