#!/usr/bin/env python3
"""Render the social preview image (site/assets/og-image.png) from the
landing page's hero design: eyebrow, gradient headline, scrub.js code
window and zvidlib branding, at the standard 1200x630 Open Graph size.

Re-run this whenever the hero copy in site/index.html changes:

    python3 site/assets/generate_og_image.py

Requires Pillow (`pip install pillow`) and is not part of CI; the PNG it
writes is committed so site/index.html's og:image and twitter:image tags
have something to point at without a build step.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

WIDTH, HEIGHT = 1200, 630
OUT_PATH = Path(__file__).parent / "og-image.png"

BG = "#0b0d17"
PANEL = "#151935"
LINE = "#262b52"
TEXT = "#e8eaf6"
MUTED = "#9aa0c8"
VIOLET = "#8b5cf6"
CYAN = "#22d3ee"
PINK = "#f472b6"
GREEN = "#34d399"
AMBER = "#fbbf24"

FONTS_DIR = Path(r"C:\Windows\Fonts")
SANS_BOLD = str(FONTS_DIR / "segoeuib.ttf")
SANS_REGULAR = str(FONTS_DIR / "segoeui.ttf")
MONO_REGULAR = str(FONTS_DIR / "consola.ttf")
MONO_BOLD = str(FONTS_DIR / "consolab.ttf")
MONO_ITALIC = str(FONTS_DIR / "consolai.ttf")


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def glow_blob(canvas: Image.Image, center: tuple[int, int], radius: tuple[int, int], color: str, alpha: int) -> None:
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = center
    rx, ry = radius
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=(*ImageColor_to_rgb(color), alpha))
    layer = layer.filter(ImageFilter.GaussianBlur(radius=min(rx, ry) // 2))
    canvas.alpha_composite(layer)


def ImageColor_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_letterspaced(draw: ImageDraw.ImageDraw, pos, text, fnt, fill, spacing):
    x, y = pos
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        x += draw.textlength(ch, font=fnt) + spacing
    return x


def gradient_text(canvas: Image.Image, pos, text, fnt, stops):
    bbox = fnt.getbbox(text)
    w = bbox[2] - bbox[0] + 4
    h = bbox[3] - bbox[1] + 4
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.text((-bbox[0] + 2, -bbox[1] + 2), text, font=fnt, fill=255)

    # linear_gradient runs top(black end) to bottom(white end); rotate it
    # into a left-to-right gradient before resizing to the text's box.
    grad_lr = Image.linear_gradient("L").rotate(90, expand=True).resize((w, h))
    colored = ImageOps.colorize(grad_lr, black=stops[0], mid=stops[1], white=stops[2]).convert("RGB")

    canvas.paste(colored, pos, mask)


def rounded_rect(draw, box, radius, **kwargs):
    draw.rounded_rectangle(box, radius=radius, **kwargs)


def draw_logo_badge(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    size = x1 - x0
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    grad_lr = Image.linear_gradient("L").rotate(45, expand=True).resize((size * 2, size * 2)).crop((size // 2, size // 2, size // 2 + size, size // 2 + size))
    colored = ImageOps.colorize(grad_lr, black=VIOLET, white=CYAN).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, size - 1, size - 1], radius=size * 0.22, fill=255)
    tile.paste(colored, (0, 0), mask)

    draw = ImageDraw.Draw(tile)
    inset = size * 0.16
    stroke = max(2, round(size * 0.07))
    draw.rounded_rectangle(
        [inset, inset * 1.1, size - inset, size - inset * 1.1],
        radius=size * 0.08,
        outline=BG,
        width=stroke,
    )
    mid_y = size / 2
    left_x = inset + size * 0.08
    right_x = size - inset - size * 0.08
    draw.line([(left_x, mid_y - size * 0.11), (right_x, mid_y - size * 0.11), (left_x, mid_y + size * 0.11), (right_x, mid_y + size * 0.11)], fill=BG, width=stroke, joint="curve")

    canvas.paste(tile, (x0, y0), tile)


def draw_code_window(canvas: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(canvas, "RGBA")

    rounded_rect(draw, (x0, y0, x1, y1), 16, fill=(21, 25, 53, 235), outline=ImageColor_to_rgb(LINE), width=2)

    bar_h = 46
    draw.line([(x0, y0 + bar_h), (x1, y0 + bar_h)], fill=ImageColor_to_rgb(LINE), width=2)
    dot_y = y0 + bar_h // 2
    dot_r = 6
    for i, color in enumerate(["#f87171", AMBER, GREEN]):
        cx = x0 + 26 + i * 24
        draw.ellipse([cx - dot_r, dot_y - dot_r, cx + dot_r, dot_y + dot_r], fill=ImageColor_to_rgb(color))

    label_font = font(MONO_ITALIC, 18)
    label = "scrub.js"
    lw = draw.textlength(label, font=label_font)
    draw.text((x1 - 22 - lw, dot_y - 11), label, font=label_font, fill=ImageColor_to_rgb(MUTED))

    code_font = font(MONO_REGULAR, 22)
    pad_x = 28
    pad_top = bar_h + 20
    line_h = 34

    def seg(x, y, pieces):
        for text, color in pieces:
            draw.text((x, y), text, font=code_font, fill=ImageColor_to_rgb(color))
            x += draw.textlength(text, font=code_font)
        return x

    tok_k, tok_s, tok_n, tok_t = "#c4a7ff", "#86efac", AMBER, "#67e8f9"
    x, y = x0 + pad_x, y0 + pad_top
    seg(x, y, [("import", tok_k), (" init, { MediaInput } ", TEXT), ("from", tok_k), (' "zvidlib";', tok_s)])
    y += line_h
    seg(x, y, [("await", tok_k), (" init();", TEXT)])
    y += line_h * 1.4
    seg(x, y, [("const", tok_k), (" frame = ", TEXT), ("await", tok_k), (" video.", TEXT), ("get", tok_t), ("(", TEXT), ("4812n", tok_n), (");", TEXT)])
    y += line_h
    seg(x, y, [("draw", tok_t), ("(frame.pixels, frame.width, frame.height);", TEXT)])


def main() -> None:
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), ImageColor_to_rgb(BG) + (255,))

    glow_blob(canvas, (int(WIDTH * 0.88), -40), (420, 320), VIOLET, 70)
    glow_blob(canvas, (-60, int(HEIGHT * 0.28)), (380, 300), CYAN, 42)

    draw = ImageDraw.Draw(canvas, "RGBA")

    margin_x = 70
    top = 55

    eyebrow_font = font(MONO_BOLD, 24)
    draw_letterspaced(draw, (margin_x, top), "RUST · WEBASSEMBLY · NO FFMPEG", eyebrow_font, ImageColor_to_rgb(CYAN), 3)

    headline_font = font(SANS_BOLD, 52)
    h1_y = top + 46
    draw.text((margin_x, h1_y), "Frame-exact video and audio,", font=headline_font, fill=ImageColor_to_rgb(TEXT))
    h2_y = h1_y + 70
    gradient_text(canvas, (margin_x, h2_y), "native and in the browser.", headline_font, (VIOLET, CYAN, PINK))

    code_top = h2_y + 70 + 22
    code_bottom = HEIGHT - 50 - 48 - 20
    draw_code_window(canvas, (margin_x, code_top, WIDTH - margin_x, code_bottom))

    badge_size = 48
    badge_y = HEIGHT - 50 - badge_size
    draw_logo_badge(canvas, (margin_x, badge_y, margin_x + badge_size, badge_y + badge_size))
    brand_font = font(SANS_BOLD, 30)
    draw.text((margin_x + badge_size + 16, badge_y + 6), "zvidlib", font=brand_font, fill=ImageColor_to_rgb(TEXT))

    url_font = font(MONO_REGULAR, 22)
    url_text = "lsegal.github.io/zvidlib"
    url_w = draw.textlength(url_text, font=url_font)
    draw.text((WIDTH - margin_x - url_w, badge_y + 13), url_text, font=url_font, fill=ImageColor_to_rgb(MUTED))

    rgb = canvas.convert("RGB")
    rgb.save(OUT_PATH, format="PNG", optimize=True)
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes, {rgb.width}x{rgb.height})")


if __name__ == "__main__":
    main()
