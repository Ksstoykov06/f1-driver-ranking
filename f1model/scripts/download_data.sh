#!/usr/bin/env bash
# Downloads the two public datasets the pilot uses, keeping only the files it needs.
#   F1DB (results, CC BY 4.0):           github.com/f1db/f1db
#   TracingInsights (FastF1-style laps): github.com/TracingInsights/<year>
set -u
DATA="${1:-data/raw}"; mkdir -p "$DATA"; cd "$DATA"
[ -d f1db ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/f1db/f1db f1db
for y in 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
  d="laps$y"
  [ -d "$d" ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --no-checkout "https://github.com/TracingInsights/$y" "$d"
  git -C "$d" sparse-checkout set --no-cone '/*/Qualifying/*/laptimes.json' '/*/Race/*/laptimes.json' '/*/Practice */*/laptimes.json' '/*/Sprint/*/laptimes.json' '/*/Sprint Qualifying/*/laptimes.json' '/*/Sprint Race/*/laptimes.json' '/*/Sprint Shootout/*/laptimes.json'
  git -C "$d" checkout -q HEAD && echo "$y: $(find "$d" -name laptimes.json | wc -l) driver-session files"
done
