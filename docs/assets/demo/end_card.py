"""Draw the end card that record.sh appends to docs/assets/demo.gif.

The card is the README's hero image, one sentence on what the run just showed,
and where to get Lasagna. It is drawn here rather than recorded so the run
itself stays a real, unedited terminal session.

    python end_card.py TERMINAL.gif CARD.png [FONT_DIR]

The card is drawn at the size of TERMINAL.gif.

FONT_DIR holds JetBrainsMono-Regular.ttf (the font agg renders the terminal
in); without it Pillow's default font is used. Needs Pillow, which
sentence-transformers already installs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
HERO = ASSETS / "hero-dark.png"

BACKGROUND = (13, 15, 18)  # the hero image's own background
TEXT = (230, 237, 243)
ACCENT = (126, 231, 135)
LINES = [
    ("Write a component. Sweep it against a real stack.", TEXT),
    ("A corrected significance test says if it helped.", TEXT),
    ("", TEXT),
    ("pip install lasagna-rag  ·  github.com/dolev31/lasagna", ACCENT),
]


def _font(font_dir: Path | None, size: int) -> ImageFont.ImageFont:
    if font_dir is not None:
        path = font_dir / "JetBrainsMono-Regular.ttf"
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def main() -> None:
    terminal, out = Path(sys.argv[1]), Path(sys.argv[2])
    font_dir = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None

    with Image.open(terminal) as gif:
        width, height = gif.size

    card = Image.new("RGB", (width, height), BACKGROUND)
    hero = Image.open(HERO).convert("RGB")
    hero_width = width - 40
    hero = hero.resize((hero_width, round(hero.height * hero_width / hero.width)), Image.LANCZOS)

    font = _font(font_dir, 20)
    line_height = 34
    block = hero.height + 40 + line_height * len(LINES)
    top = (height - block) // 2
    card.paste(hero, ((width - hero.width) // 2, top))

    draw = ImageDraw.Draw(card)
    y = top + hero.height + 40
    for text, color in LINES:
        if text:
            line_width = draw.textlength(text, font=font)
            draw.text(((width - line_width) / 2, y), text, font=font, fill=color)
        y += line_height
    card.save(out)


if __name__ == "__main__":
    main()
