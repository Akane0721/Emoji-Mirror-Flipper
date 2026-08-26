"""Mirror-symmetry core logic for static images and animated GIFs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageSequence, UnidentifiedImageError

DIRECTIONS = ("l2r", "r2l", "t2b", "b2t")

DIRECTION_LABELS = {
    "l2r": "左→右",
    "r2l": "右→左",
    "t2b": "上→下",
    "b2t": "下→上",
}

# Kaleidoscope is a different kind of operation from the four mirror
# directions: it takes a segment count instead of an axis.
KALEIDO = "kaleido"
KALEIDO_LABEL = "万花筒"
# Must be even, otherwise the mirrored cell can't tile the circle exactly.
KALEIDO_SEGMENTS = (4, 6, 8, 12, 16)
DEFAULT_SEGMENTS = 8

MODES = DIRECTIONS + (KALEIDO,)

# Accepted input formats, mapping Pillow's detected format name to an extension
FORMAT_EXT = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}

MAX_PIXELS = 50_000_000  # decompression-bomb guard, roughly 7000x7000

_VARIANT_RE = re.compile(r"\A(?:(l2r|r2l|t2b|b2t)|kaleido(\d{1,2}))\Z")


class MirrorError(Exception):
    """Raised when the input can't be processed; the message is user-facing."""


@dataclass(frozen=True)
class ImageInfo:
    fmt: str
    ext: str
    width: int
    height: int
    animated: bool
    n_frames: int


# --------------------------------------------------------------------------
# Variant keys
#
# A "variant" is the single token that names one rendered result, used both as
# the output filename stem and as the URL path segment. Mirror modes are their
# own name; kaleidoscope folds the segment count in, so 6 and 12 segments don't
# collide in the output cache.
# --------------------------------------------------------------------------

def variant_key(mode: str, segments: int | None = None) -> str:
    if mode == KALEIDO:
        return f"{KALEIDO}{segments if segments is not None else DEFAULT_SEGMENTS}"
    return mode


def parse_variant(variant: str) -> tuple[str, int | None]:
    """Turn a variant token back into (mode, segments), validating both."""
    match = _VARIANT_RE.match(variant or "")
    if match is None:
        raise MirrorError("未知的处理方式")
    if match.group(1):
        return match.group(1), None
    segments = int(match.group(2))
    if segments not in KALEIDO_SEGMENTS:
        raise MirrorError(f"万花筒瓣数只能是 {', '.join(map(str, KALEIDO_SEGMENTS))}")
    return KALEIDO, segments


def probe(path: Path) -> ImageInfo:
    """Verify the file really is an image and report basic facts about it.

    Trusts the file contents only, never the extension.
    """
    try:
        with Image.open(path) as im:
            fmt = im.format or ""
            if fmt not in FORMAT_EXT:
                raise MirrorError(f"不支持的格式：{fmt or '无法识别'}（只吃 jpg/png/gif/webp）")

            width, height = im.size
            if width < 2 or height < 2:
                raise MirrorError("图片太小了，至少得有 2x2")
            if width * height > MAX_PIXELS:
                raise MirrorError("图片像素太多了，换张小点的")

            n_frames = getattr(im, "n_frames", 1)
            return ImageInfo(
                fmt=fmt,
                ext=FORMAT_EXT[fmt],
                width=width,
                height=height,
                animated=bool(getattr(im, "is_animated", False)),
                n_frames=n_frames,
            )
    except UnidentifiedImageError:
        raise MirrorError("这不是一个能识别的图片文件") from None
    except MirrorError:
        raise
    except OSError as exc:
        raise MirrorError(f"图片读取失败：{exc}") from None


# --------------------------------------------------------------------------
# Per-frame transforms
# --------------------------------------------------------------------------

def _mirror_frame(frame: Image.Image, direction: str) -> Image.Image:
    """Mirror a single frame. Takes and returns RGBA, preserving alpha."""
    if direction not in DIRECTIONS:
        raise MirrorError(f"未知的翻转方向：{direction}")

    width, height = frame.size

    # Round the half up: on odd dimensions the middle row/column gets pasted
    # twice with the same value, so there's neither a black seam nor an offset.
    # (The original `// 2` dropped a column outright on odd widths.)
    if direction in ("l2r", "r2l"):
        half = math.ceil(width / 2)
        transpose = Image.Transpose.FLIP_LEFT_RIGHT
        if direction == "l2r":
            kept = frame.crop((0, 0, half, height))
            kept_at, mirror_at = (0, 0), (width - half, 0)
        else:
            kept = frame.crop((width - half, 0, width, height))
            kept_at, mirror_at = (width - half, 0), (0, 0)
    else:
        half = math.ceil(height / 2)
        transpose = Image.Transpose.FLIP_TOP_BOTTOM
        if direction == "t2b":
            kept = frame.crop((0, 0, width, half))
            kept_at, mirror_at = (0, 0), (0, height - half)
        else:
            kept = frame.crop((0, height - half, width, height))
            kept_at, mirror_at = (0, height - half), (0, 0)

    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    out.paste(kept, kept_at)
    out.paste(kept.transpose(transpose), mirror_at)
    return out


def _wedge_mask(size: int, wedge_deg: float) -> Image.Image:
    """Build an antialiased mask covering one pie wedge starting at 3 o'clock.

    The mask is drawn at 4x and shrunk back down, otherwise the two straight
    radial edges come out visibly stair-stepped.
    """
    scale = 4
    big = Image.new("L", (size * scale, size * scale), 0)
    center = size * scale / 2
    # Overshoot the radius so the wedge still covers the square's corners.
    radius = size * scale
    ImageDraw.Draw(big).pieslice(
        [center - radius, center - radius, center + radius, center + radius],
        0,
        wedge_deg,
        fill=255,
    )
    return big.resize((size, size), Image.LANCZOS)


def _kaleido_frame(frame: Image.Image, segments: int) -> Image.Image:
    """Build an n-fold kaleidoscope from a single frame.

    Crops to a centred square, cuts one wedge of 360/n degrees, mirrors it into
    a cell spanning twice that angle, then rotates the cell around the centre
    n/2 times to close the circle. The result has n mirror lines through the
    centre, which is what makes it read as a kaleidoscope rather than as a
    plain rotation.
    """
    if segments not in KALEIDO_SEGMENTS:
        raise MirrorError(f"万花筒瓣数只能是 {', '.join(map(str, KALEIDO_SEGMENTS))}")

    size = min(frame.size)
    left = (frame.width - size) // 2
    top = (frame.height - size) // 2
    base = frame.crop((left, top, left + size, top + size))

    wedge_deg = 360 / segments
    centre = size / 2

    wedge = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    wedge.paste(base, (0, 0), _wedge_mask(size, wedge_deg))

    # Flipping vertically reflects across the horizontal centre line, i.e.
    # maps angle t to -t, so wedge + flipped wedge spans [-wedge, +wedge].
    cell = Image.alpha_composite(
        wedge.transpose(Image.Transpose.FLIP_TOP_BOTTOM), wedge
    )

    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for k in range(segments // 2):
        rotated = cell.rotate(
            -k * 2 * wedge_deg,
            resample=Image.BICUBIC,
            center=(centre, centre),
        )
        out = Image.alpha_composite(out, rotated)
    return out


def _transform(frame: Image.Image, mode: str, segments: int | None) -> Image.Image:
    if mode == KALEIDO:
        return _kaleido_frame(frame, segments or DEFAULT_SEGMENTS)
    return _mirror_frame(frame, mode)


# --------------------------------------------------------------------------
# File-level entry points
# --------------------------------------------------------------------------

def _flip_static(src: Path, dst: Path, mode: str, segments: int | None) -> None:
    with Image.open(src) as im:
        im.load()
        frame = im.convert("RGBA")

    out = _transform(frame, mode, segments)

    if dst.suffix.lower() in (".jpg", ".jpeg"):
        # JPEG has no alpha channel, so composite onto white first
        canvas = Image.new("RGB", out.size, (255, 255, 255))
        canvas.paste(out, mask=out.getchannel("A"))
        canvas.save(dst, quality=95, subsampling=0)
    else:
        out.save(dst)


def _flip_animated(src: Path, dst: Path, mode: str, segments: int | None) -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []

    with Image.open(src) as im:
        default_duration = im.info.get("duration", 100)
        loop = im.info.get("loop", 0)
        for frame in ImageSequence.Iterator(im):
            # convert() composites the current frame per the GIF disposal rules
            frames.append(_transform(frame.convert("RGBA"), mode, segments))
            durations.append(frame.info.get("duration", default_duration))

    if not frames:
        raise MirrorError("这个动图里一帧都没有")

    frames[0].save(
        dst,
        save_all=True,
        append_images=frames[1:],
        duration=durations,  # was omitted before, so output fell back to default speed
        loop=loop,
        disposal=2,
        optimize=False,
    )


def flip(
    src: Path,
    dst: Path,
    mode: str,
    animated: bool,
    segments: int | None = None,
) -> None:
    """Transform src into dst. `animated` comes from probe().

    `mode` is one of DIRECTIONS or KALEIDO; `segments` only applies to the
    latter and falls back to DEFAULT_SEGMENTS.
    """
    if mode not in MODES:
        raise MirrorError(f"未知的处理方式：{mode}")

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if animated:
            _flip_animated(src, dst, mode, segments)
        else:
            _flip_static(src, dst, mode, segments)
    except MirrorError:
        raise
    except OSError as exc:
        raise MirrorError(f"处理失败：{exc}") from None
