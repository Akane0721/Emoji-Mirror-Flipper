"""Mirror-symmetry core logic for static images and animated GIFs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageSequence, UnidentifiedImageError

# --------------------------------------------------------------------------
# Modes
#
# Axis mirrors and the quad fold keep the source's aspect ratio. The radial
# modes (kaleidoscope, pinwheel) crop to a centred square and take a segment
# count plus a start angle.
# --------------------------------------------------------------------------

DIRECTIONS = ("l2r", "r2l", "t2b", "b2t")
QUAD = "quad"
KALEIDO = "kaleido"
PINWHEEL = "pinwheel"

RADIAL_MODES = (KALEIDO, PINWHEEL)
MODES = DIRECTIONS + (QUAD, KALEIDO, PINWHEEL)

MODE_LABELS = {
    "l2r": "左→右",
    "r2l": "右→左",
    "t2b": "上→下",
    "b2t": "下→上",
    QUAD: "田 四象限",
    KALEIDO: "🔮 万花筒",
    PINWHEEL: "🌀 风车",
}

# Kept for callers that only care about the four axis mirrors.
DIRECTION_LABELS = {key: MODE_LABELS[key] for key in DIRECTIONS}

# The kaleidoscope mirrors each wedge, so two wedges make one cell and the
# count has to be even for the cells to tile the circle exactly. The pinwheel
# only rotates, so any count works -- 3 and 5 blades look good and are only
# reachable here.
SEGMENTS_BY_MODE = {
    KALEIDO: (4, 6, 8, 12, 16),
    PINWHEEL: (3, 4, 5, 6, 8, 12, 16),
}
DEFAULT_SEGMENTS = 8

# Which wedge of the source gets sampled. Everything outside it is discarded,
# so this changes the result far more than it sounds like it would.
OFFSET_STEP = 5
OFFSET_CHOICES = tuple(range(0, 360, OFFSET_STEP))
DEFAULT_OFFSET = 0

FORMAT_EXT = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}

MAX_PIXELS = 50_000_000  # decompression-bomb guard, roughly 7000x7000

_PLAIN_RE = re.compile(r"\A(l2r|r2l|t2b|b2t|quad)\Z")
_RADIAL_RE = re.compile(r"\A(kaleido|pinwheel)_(\d{1,2})_(\d{1,3})\Z")


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
# A "variant" names one rendered result and is used both as the output
# filename stem and as the URL path segment. Radial modes fold their
# parameters in so that different settings cache separately.
# --------------------------------------------------------------------------

def variant_key(mode: str, segments: int | None = None, offset: int | None = None) -> str:
    if mode in RADIAL_MODES:
        seg = DEFAULT_SEGMENTS if segments is None else segments
        off = DEFAULT_OFFSET if offset is None else offset
        return f"{mode}_{seg}_{off}"
    return mode


def parse_variant(variant: str) -> tuple[str, int | None, int | None]:
    """Turn a variant token back into (mode, segments, offset), validating it."""
    variant = variant or ""
    if _PLAIN_RE.match(variant):
        return variant, None, None

    match = _RADIAL_RE.match(variant)
    if match is None:
        raise MirrorError("未知的处理方式")

    mode = match.group(1)
    segments = int(match.group(2))
    offset = int(match.group(3))
    check_segments(mode, segments)
    check_offset(offset)
    return mode, segments, offset


def check_segments(mode: str, segments: int) -> None:
    allowed = SEGMENTS_BY_MODE.get(mode, ())
    if segments not in allowed:
        raise MirrorError(f"瓣数只能是 {', '.join(map(str, allowed))}")


def check_offset(offset: int) -> None:
    if not 0 <= offset < 360:
        raise MirrorError("起始角必须在 0 到 359 度之间")


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

            return ImageInfo(
                fmt=fmt,
                ext=FORMAT_EXT[fmt],
                width=width,
                height=height,
                animated=bool(getattr(im, "is_animated", False)),
                n_frames=getattr(im, "n_frames", 1),
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
    """Mirror a single frame across one axis. RGBA in, RGBA out."""
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


def _quad_frame(frame: Image.Image) -> Image.Image:
    """Fold the top-left quadrant out to all four. Keeps the aspect ratio."""
    return _mirror_frame(_mirror_frame(frame, "l2r"), "t2b")


def _wedge_mask(size: int, wedge_deg: float, start_deg: float) -> Image.Image:
    """Antialiased mask covering one pie wedge, measured clockwise from 3 o'clock.

    Drawn at 4x and shrunk back down, otherwise the two straight radial edges
    come out visibly stair-stepped.
    """
    scale = 4
    big = Image.new("L", (size * scale, size * scale), 0)
    centre = size * scale / 2
    # Overshoot the radius so the wedge still reaches past the square's corners.
    radius = size * scale
    ImageDraw.Draw(big).pieslice(
        [centre - radius, centre - radius, centre + radius, centre + radius],
        start_deg,
        start_deg + wedge_deg,
        fill=255,
    )
    return big.resize((size, size), Image.LANCZOS)


def _radial_frame(
    frame: Image.Image, mode: str, segments: int, offset: int
) -> Image.Image:
    """Build an n-fold radial pattern from a single frame.

    Crops to a centred square and cuts one wedge of 360/n degrees starting at
    `offset`. The kaleidoscope then mirrors that wedge into a cell spanning
    twice the angle and rotates the cell n/2 times; the pinwheel skips the
    mirror and rotates the bare wedge n times.

    The mirror is what makes a kaleidoscope read as a kaleidoscope: it leaves
    n mirror lines through the centre. Rotation alone gives a pinwheel, which
    has a direction of spin and no mirror lines at all.
    """
    check_segments(mode, segments)
    check_offset(offset)

    size = min(frame.size)
    left = (frame.width - size) // 2
    top = (frame.height - size) // 2
    base = frame.crop((left, top, left + size, top + size))

    # Assemble on a canvas scaled up by sqrt(2), then crop the centre back to
    # `size`. Without this the output loses its corners: a wedge sampled near
    # an edge midpoint only carries content out to radius size/2, while the
    # output's corners sit at size*0.707, and every rotation step that isn't a
    # multiple of 90 degrees drags those short wedges into the corner region.
    # Scaling first makes the working canvas' inscribed circle reach the final
    # corners, so every direction has content all the way out.
    work = math.ceil(size * math.sqrt(2)) + 2
    # Keep the padding even on both sides, otherwise the final centre crop sits
    # half a pixel off and the mirror lines stop being exact.
    if (work - size) % 2:
        work += 1
    base = base.resize((work, work), Image.LANCZOS)

    wedge_deg = 360 / segments
    centre = work / 2

    wedge = Image.new("RGBA", (work, work), (0, 0, 0, 0))
    wedge.paste(base, (0, 0), _wedge_mask(work, wedge_deg, offset))

    # rotate(+a) maps content at angle a back to angle 0, so this brings the
    # sampled wedge into the canonical [0, wedge_deg] band and the assembly
    # below stays identical for every offset.
    if offset:
        wedge = wedge.rotate(offset, resample=Image.BICUBIC, center=(centre, centre))

    if mode == KALEIDO:
        # Flipping vertically reflects across the horizontal centre line, i.e.
        # maps angle t to -t, so wedge + flipped wedge spans [-wedge, +wedge].
        cell = Image.alpha_composite(
            wedge.transpose(Image.Transpose.FLIP_TOP_BOTTOM), wedge
        )
        step, count = 2 * wedge_deg, segments // 2
    else:
        cell, step, count = wedge, wedge_deg, segments

    out = Image.new("RGBA", (work, work), (0, 0, 0, 0))
    for k in range(count):
        out = Image.alpha_composite(
            out,
            cell.rotate(-k * step, resample=Image.BICUBIC, center=(centre, centre)),
        )

    inset = (work - size) // 2
    return out.crop((inset, inset, inset + size, inset + size))


def _transform(
    frame: Image.Image, mode: str, segments: int | None, offset: int | None
) -> Image.Image:
    if mode in RADIAL_MODES:
        return _radial_frame(
            frame,
            mode,
            DEFAULT_SEGMENTS if segments is None else segments,
            DEFAULT_OFFSET if offset is None else offset,
        )
    if mode == QUAD:
        return _quad_frame(frame)
    return _mirror_frame(frame, mode)


# --------------------------------------------------------------------------
# File-level entry points
# --------------------------------------------------------------------------

def _flip_static(
    src: Path, dst: Path, mode: str, segments: int | None, offset: int | None
) -> None:
    with Image.open(src) as im:
        im.load()
        frame = im.convert("RGBA")

    out = _transform(frame, mode, segments, offset)

    if dst.suffix.lower() in (".jpg", ".jpeg"):
        # JPEG has no alpha channel, so composite onto white first
        canvas = Image.new("RGB", out.size, (255, 255, 255))
        canvas.paste(out, mask=out.getchannel("A"))
        canvas.save(dst, quality=95, subsampling=0)
    else:
        out.save(dst)


def _flip_animated(
    src: Path, dst: Path, mode: str, segments: int | None, offset: int | None
) -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []

    with Image.open(src) as im:
        default_duration = im.info.get("duration", 100)
        loop = im.info.get("loop", 0)
        for frame in ImageSequence.Iterator(im):
            # convert() composites the current frame per the GIF disposal rules
            frames.append(_transform(frame.convert("RGBA"), mode, segments, offset))
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
    offset: int | None = None,
) -> None:
    """Transform src into dst. `animated` comes from probe().

    `segments` and `offset` only apply to the radial modes and fall back to
    DEFAULT_SEGMENTS / DEFAULT_OFFSET.
    """
    if mode not in MODES:
        raise MirrorError(f"未知的处理方式：{mode}")

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if animated:
            _flip_animated(src, dst, mode, segments, offset)
        else:
            _flip_static(src, dst, mode, segments, offset)
    except MirrorError:
        raise
    except OSError as exc:
        raise MirrorError(f"处理失败：{exc}") from None
