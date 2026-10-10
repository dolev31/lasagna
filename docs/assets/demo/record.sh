#!/usr/bin/env bash
# Regenerate docs/assets/demo.gif and docs/assets/demo.mp4 from a real run.
#
#   LASAGNA_PYTHON=.venv/bin/python AGG=agg docs/assets/demo/record.sh
#
# Needs an interpreter that imports lasagna, uv (it fetches asciinema 2.4),
# agg 1.9 (https://github.com/asciinema/agg/releases) and ffmpeg (FFMPEG=).
# The font is JetBrains Mono: agg finds it when installed, or FONT_DIR= names a
# folder of its .ttf files; without it agg falls back to Menlo or DejaVu. The run
# downloads BAAI/bge-small-en-v1.5, cross-encoder/ms-marco-MiniLM-L-6-v2 and
# BEIR SciFact on first use; set HF_HOME and HF_HUB_OFFLINE=1 to use a cache.
# RENDER_ONLY=1 skips the run and re-renders the committed demo.cast.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
assets=$(dirname "$here")
py=${LASAGNA_PYTHON:-python3}
agg=${AGG:-agg}
ffmpeg=${FFMPEG:-ffmpeg}

if [[ -z ${RENDER_ONLY:-} ]]; then
  work=$(mktemp -d)
  mkdir -p "$work/bin" "$work/home" "$work/project"
  cp "$here/my_ideas.py" "$here/sweep.yaml" "$work/project/"
  # Shown as `lasagna` in the recording; it is `python -m lasagna`.
  printf '#!/bin/sh\nexec %q -m lasagna "$@"\n' "$("$py" -c 'import sys; print(sys.executable)')" \
    > "$work/bin/lasagna"
  chmod +x "$work/bin/lasagna"

  # A fresh HOME, so the run starts with no index and no results. The rest
  # keeps library chatter out of the frame:
  #   LASAGNA_LOG_LEVEL=ERROR  hides Lasagna's warnings (default WARNING)
  #   LOGURU_FORMAT            prints the corpus progress line without the
  #                            timestamp and module prefix
  #   GRPC_VERBOSITY=ERROR     Milvus Lite's gRPC fork notices
  #   HF_HUB_DISABLE_PROGRESS_BARS=1, DATASETS_VERBOSITY=error
  #                            model-loading bars and dataset-cache notices
  (
    cd "$work/project"
    env HOME="$work/home" PATH="$work/bin:$PATH" TERM=xterm-256color \
      LASAGNA_LOG_LEVEL=ERROR LOGURU_FORMAT='{message}' GRPC_VERBOSITY=ERROR \
      HF_HUB_DISABLE_PROGRESS_BARS=1 DATASETS_VERBOSITY=error \
      uvx --from asciinema==2.4.0 asciinema rec --overwrite --quiet --cols 100 --rows 36 \
      --command "bash $(printf %q "$here/session.sh")" "$work/raw.cast" </dev/null
  )
  # The run takes about five minutes; cut its idle waits. Output is unchanged.
  python3 "$here/retime.py" "$work/raw.cast" "$here/demo.cast" \
    --expect-markers "$(grep -c '^pause ' "$here/session.sh")"
  echo "The run and its results are in $work/project"
fi

frames=$(mktemp -d)
"$agg" --quiet ${FONT_DIR:+--font-dir "$FONT_DIR"} --theme github-dark \
  --font-size 16 --line-height 1.35 --idle-time-limit 10 --last-frame-duration 6 --fps-cap 15 \
  "$here/demo.cast" "$frames/terminal.gif"
# Title card (3 s) and end card (4.5 s) around the run: the hero image with
# what the run answers, then where to get Lasagna. One palette from every frame
# keeps the cards' colors exact, and rectangle diffs store only the pixels that
# change, so the cards cost little.
"$py" "$here/cards.py" "$frames/terminal.gif" "$frames/title.png" "$frames/end.png" "${FONT_DIR:-}"
"$ffmpeg" -y -loglevel error \
  -loop 1 -framerate 15 -t 3 -i "$frames/title.png" \
  -i "$frames/terminal.gif" \
  -loop 1 -framerate 15 -t 4.5 -i "$frames/end.png" \
  -filter_complex "[0:v]fps=15,format=rgb24,setsar=1[s];[1:v]fps=15,format=rgb24,setsar=1[t];[2:v]fps=15,format=rgb24,setsar=1[e];[s][t][e]concat=n=3:v=1:a=0,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
  -loop 0 "$assets/demo.gif"
# fps=15 keeps the GIF's final hold, which a straight conversion cuts short.
"$ffmpeg" -y -loglevel error -i "$assets/demo.gif" -movflags +faststart \
  -vf "fps=15,scale=trunc(iw/2)*2:trunc(ih/2)*2" -pix_fmt yuv420p -c:v libx264 -crf 24 \
  "$assets/demo.mp4"
ls -l "$assets/demo.gif" "$assets/demo.mp4"
