"""Draw the title and end cards that record.sh puts around docs/assets/demo.gif.

Both cards are the README's hero image with two lines under it. The title card
says what the run is about to answer; the end card says where to get Lasagna.
They are drawn here rather than recorded so the run itself stays a real,
unedited terminal session.

    python cards.py TERMINAL.gif TITLE.png END.png [FONT_DIR]

The cards are drawn at the size of TERMINAL.gif. FONT_DIR holds
JetBrainsMono-Regular.ttf (the font agg renders the terminal in); without it
Pillow's default font is used. Needs Pillow, which sentence-transformers
already installs.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
HERO = HERE.parent / "hero-dark.png"

BACKGROUND = (13, 15, 18)  # the hero image's own background
TEXT = (230, 237, 243)
ACCENT = (126, 231, 135)

TITLE = [
    ("Does my reranker beat a real baseline?", TEXT),
    ("One function, one sweep, one significance test.", TEXT),
]
END = [
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


def draw_card(size: tuple[int, int], lines: list, font: ImageFont.ImageFont) -> Image.Image:
    width, height = size
    card = Image.new("RGB", (width, height), BACKGROUND)
    hero = Image.open(HERO).convert("RGB")
    hero_width = width - 40
    hero = hero.resize((hero_width, round(hero.height * hero_width / hero.width)), Image.LANCZOS)

    line_height = 34
    block = hero.height + 40 + line_height * len(lines)
    top = (height - block) // 2
    card.paste(hero, ((width - hero.width) // 2, top))

    draw = ImageDraw.Draw(card)
    y = top + hero.height + 40
    for text, color in lines:
        if text:
            line_width = draw.textlength(text, font=font)
            draw.text(((width - line_width) / 2, y), text, font=font, fill=color)
        y += line_height
    return card


def main() -> None:
    terminal, title, end = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    font_dir = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None

    with Image.open(terminal) as gif:
        size = gif.size
    font = _font(font_dir, 20)
    draw_card(size, TITLE, font).save(title)
    draw_card(size, END, font).save(end)


if __name__ == "__main__":
    main()
