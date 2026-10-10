#!/usr/bin/env bash
# The terminal session recorded into docs/assets/demo.gif.
#
# Every command runs for real, in the directory record.sh prepares (this
# folder's my_ideas.py and sweep.yaml, nothing else). Only the typing is
# simulated. pause() writes a marker that retime.py turns into a pause of
# that many seconds and then removes; a terminal ignores it.
set -u

pause() { printf '\033]1337;LasagnaDemoPause=%s\007' "$1"; }

run() {
  local cmd=$1 i
  printf '\033[1;32m$\033[0m '
  for ((i = 0; i < ${#cmd}; i++)); do
    printf '%s' "${cmd:i:1}"
    sleep 0.03
  done
  sleep 0.4
  printf '\n'
  eval "$cmd"
}

clear
run "cat my_ideas.py"
pause 2.5
run "cat sweep.yaml"
pause 3.5
clear
run "lasagna benchmark experiment sweep.yaml"
pause 3
run "lasagna results significance runs/keyword_overlap/results.json --all"
