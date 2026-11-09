#!/bin/bash
# Open a waveform in GTKWave with the signals, radix, zoom and marker already set.
# Usage:  bash docs/gtkwave/view_waves.sh 1      (1..6, see list below)
#   1 scorer: sweep start     2 scorer: sweep done      3 scorer: eviction under pressure
#   4 scorer: reset           5 codec pipeline          6 block table
cd "$(dirname "$0")/../.." || exit 1
case "$1" in
  1) T=g01_scorer_start;   V=trace_sweep_scorer ;;
  2) T=g02_scorer_done;    V=trace_sweep_scorer ;;
  3) T=g03_scorer_evict;   V=trace_sweep_scorer ;;
  4) T=g04_scorer_reset;   V=trace_sweep_scorer ;;
  5) T=g05_codec_pipeline; V=trace_kv_codec ;;
  6) T=g06_block_table;    V=trace_block_table ;;
  *) sed -n 2,6p "$0"; exit 1 ;;
esac
gtkwave -S docs/gtkwave/$T.tcl docs/wave/$V.vcd
