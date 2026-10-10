# The README demo

`docs/assets/demo.gif` (and `demo.mp4`, the same frames) is a recording of a
real run: the commands in [`session.sh`](session.sh), executed against
[`my_ideas.py`](my_ideas.py) and [`sweep.yaml`](sweep.yaml) in this folder, on
100 BEIR SciFact queries with the `speed` profile. The typing is scripted, and
every line of output is what the commands printed. Three things differ from
watching it live:

- **Time is compressed.** The run takes about five minutes on a laptop CPU,
  most of it embedding the 5,183-document corpus. [`retime.py`](retime.py)
  caps every wait between two lines of output at 0.7 s and holds the scripted
  pauses; the corpus progress lines still say how long it really took.
- **Library chatter is switched off** with environment variables, so the frame
  shows Lasagna's own output:

  | Variable | Hides |
  |---|---|
  | `LASAGNA_LOG_LEVEL=ERROR` | Lasagna's warnings (the CLI default is `WARNING`) |
  | `LOGURU_FORMAT='{message}'` | the timestamp and module prefix on the corpus progress line |
  | `GRPC_VERBOSITY=ERROR` | Milvus Lite's gRPC fork notices |
  | `HF_HUB_DISABLE_PROGRESS_BARS=1` | model-loading progress bars |
  | `DATASETS_VERBOSITY=error` | dataset-cache notices |

- **Two cards frame the run**: the README's hero image with the question the
  run answers (3 s) before it, and with where to get Lasagna (4.5 s) after it,
  drawn by [`cards.py`](cards.py). They are the only frames not printed by a
  command.

`lasagna` on screen is `python -m lasagna`, through a two-line wrapper that
`record.sh` puts on `PATH`.

## Regenerate

You need an interpreter with Lasagna installed (`pip install "lasagna-rag[eval]"`,
or a checkout with `PYTHONPATH=src`), [uv](https://docs.astral.sh/uv/) (it
fetches asciinema 2.4), [agg](https://github.com/asciinema/agg/releases) 1.9 and
ffmpeg (`FFMPEG=` names one). `cards.py` needs Pillow, which the
interpreter above already has. The font is JetBrains Mono: agg finds it when it is installed, or point
`FONT_DIR` at a folder holding its `.ttf` files.

```bash
LASAGNA_PYTHON=.venv/bin/python AGG=agg docs/assets/demo/record.sh
```

This records a fresh run in a temporary directory with an empty `HOME`, writes
the retimed recording to `demo.cast`, and renders `demo.gif` and `demo.mp4` into
`docs/assets/`, with the title and end cards around it. The first run downloads `BAAI/bge-small-en-v1.5`,
`cross-encoder/ms-marco-MiniLM-L-6-v2` and SciFact; with them cached, set
`HF_HUB_OFFLINE=1`.

To change only the look (theme, font size, speed), edit the `agg` line in
`record.sh` and re-render the committed recording without running anything:

```bash
RENDER_ONLY=1 AGG=agg docs/assets/demo/record.sh
```

The terminal is 100 columns by 36 rows, rendered at 16 px in agg's
`github-dark` theme: about 980 px wide, legible at the 800 px a README shows.
