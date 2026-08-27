"""Symmetry core logic for static images and animated GIFs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageSequence, UnidentifiedImageError

# --------------------------------------------------------------------------
# Modes
#
# Axis mirrors, the quad fold and the mirror tiling keep the source's aspect
# ratio. The diagonal mirrors and the radial modes crop to a centred square,
# because reflecting across a diagonal or around a centre only makes sense on
# one.
# --------------------------------------------------------------------------

DIRECTIONS = ("l2r", "r2l", "t2b", "b2t")
DIAGONALS = ("d1", "d2")
QUAD = "quad"
KALEIDO = "kaleido"
PINWHEEL = "pinwheel"
SPIRAL = "spiral"
TILE = "tile"
PAPERCUT = "papercut"

# Modes assembled from angular wedges around the centre.
RADIAL_MODES = (KALEIDO, PINWHEEL, SPIRAL)
# Modes whose count means "cells per side" rather than "segments".
GRID_MODES = (TILE, PAPERCUT)
# Modes that take a numeric parameter from the same selector.
COUNT_MODES = RADIAL_MODES + GRID_MODES
# Only the spiral has a twist amount on top of count and start angle.
TWIST_MODES = (SPIRAL,)
# Modes with no parameters at all.
AXIS_MODES = DIRECTIONS
FOLD_MODES = DIAGONALS + (QUAD,)
PLAIN_MODES = AXIS_MODES + FOLD_MODES

MODES = PLAIN_MODES + COUNT_MODES

# How the UI groups the buttons, in display order.
MODE_GROUPS = (
    ("单轴", AXIS_MODES),
    ("折叠", FOLD_MODES),
    ("图案", (KALEIDO, PINWHEEL, SPIRAL, TILE, PAPERCUT)),
)

MODE_LABELS = {
    "l2r": "➡️ 左→右",
    "r2l": "⬅️ 右→左",
    "t2b": "⬇️ 上→下",
    "b2t": "⬆️ 下→上",
    "d1": "↘️ 主对角",
    "d2": "↗️ 副对角",
    QUAD: "🪟 四重存在",
    KALEIDO: "🔮 万华镜",
    PINWHEEL: "🌀 风车",
    SPIRAL: "🌪️ 螺旋",
    TILE: "🧱 镜面平铺",
    PAPERCUT: "🪷 窗花",
}

# Plain names for download filenames, without the emoji.
MODE_FILENAMES = {
    "l2r": "左右对称",
    "r2l": "右左对称",
    "t2b": "上下对称",
    "b2t": "下上对称",
    "d1": "主对角对称",
    "d2": "副对角对称",
    QUAD: "四重存在",
    KALEIDO: "万华镜",
    PINWHEEL: "风车",
    SPIRAL: "螺旋",
    TILE: "镜面平铺",
    PAPERCUT: "窗花",
}

# The kaleidoscope mirrors each wedge, so two wedges make one cell and the
# count has to be even for the cells to tile the circle exactly. The pinwheel
# only rotates, so any count works -- 3 and 5 blades are only reachable there.
COUNTS_BY_MODE = {
    KALEIDO: (4, 6, 8, 12, 16),
    PINWHEEL: (3, 4, 5, 6, 8, 12, 16),
    SPIRAL: (2, 3, 4, 5, 6, 8, 12),
    TILE: (2, 3, 4, 6),
    PAPERCUT: (2, 3, 4, 6),
}
DEFAULT_COUNT_BY_MODE = {KALEIDO: 8, PINWHEEL: 8, SPIRAL: 6, TILE: 3, PAPERCUT: 3}
# The grid modes count cells per side rather than segments around a centre, so
# their selector is labelled differently and their readouts spell out the whole
# n x n grid instead of a bare number.
COUNT_LABELS = {KALEIDO: "瓣数", PINWHEEL: "叶数", SPIRAL: "臂数",
                TILE: "格数", PAPERCUT: "格数"}
COUNT_UNITS = {KALEIDO: "瓣", PINWHEEL: "叶", SPIRAL: "臂",
               TILE: "格", PAPERCUT: "格"}

# Total twist in degrees, strongest at the centre and fading to nothing at the
# rim so the frame edges stay put.
TWIST_STEP = 15
DEFAULT_TWIST = 180
MAX_TWIST = 720


def describe_count(mode: str, count: int) -> str:
    """Render a count for labels and filenames.

    Radial modes get a bare number plus their unit; grid modes get the full
    "n x n" form, since there the count means cells per side.
    """
    if mode in GRID_MODES:
        return f"{count}x{count}{COUNT_UNITS[mode]}"
    return f"{count}{COUNT_UNITS[mode]}"


# Which wedge of the source the radial modes sample. Everything outside it is
# discarded, so this changes the result far more than it sounds like it would.
OFFSET_STEP = 5
DEFAULT_OFFSET = 0

FORMAT_EXT = {"JPEG": "jpg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}

MAX_PIXELS = 50_000_000  # decompression-bomb guard, roughly 7000x7000

_PLAIN_RE = re.compile(r"\A(l2r|r2l|t2b|b2t|d1|d2|quad)\Z")
_COUNT_RE = re.compile(r"\A(kaleido|pinwheel|tile|papercut)_(\d{1,2})_(\d{1,3})\Z")
# The spiral carries a fourth field for its twist.
_SPIRAL_RE = re.compile(r"\A(spiral)_(\d{1,2})_(\d{1,3})_(\d{1,3})\Z")


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


@dataclass(frozen=True)
class Step:
    """One effect in a chain, with whatever parameters that mode takes."""
    mode: str
    count: int | None = None
    offset: int | None = None
    twist: int | None = None


# Effects can be stacked. Each extra step is another full pass over every
# frame, so the chain is capped to keep animated GIFs from crawling.
MAX_STEPS = 4
# Safe because no mode name and no parameter ever contains a hyphen, so a
# single-step key stays byte-identical to what it was before chaining existed.
CHAIN_SEP = "-"


# --------------------------------------------------------------------------
# Variant keys
#
# A "variant" names one rendered result and is used both as the output
# filename stem and as the URL path segment. Parameterised modes fold their
# settings in so different settings cache separately.
# --------------------------------------------------------------------------

def default_count(mode: str) -> int:
    return DEFAULT_COUNT_BY_MODE.get(mode, 8)


def variant_key(
    mode: str,
    count: int | None = None,
    offset: int | None = None,
    twist: int | None = None,
) -> str:
    if mode not in COUNT_MODES:
        return mode

    n = default_count(mode) if count is None else count
    off = DEFAULT_OFFSET if offset is None else offset
    if mode not in RADIAL_MODES:
        off = 0  # only the radial modes have a start angle
    if mode in TWIST_MODES:
        tw = DEFAULT_TWIST if twist is None else twist
        return f"{mode}_{n}_{off}_{tw}"
    return f"{mode}_{n}_{off}"


def parse_variant(variant: str) -> tuple[str, int | None, int | None, int | None]:
    """Turn a variant token back into (mode, count, offset, twist)."""
    variant = variant or ""
    if _PLAIN_RE.match(variant):
        return variant, None, None, None

    match = _SPIRAL_RE.match(variant)
    if match is not None:
        mode, count, offset, twist = (match.group(1), int(match.group(2)),
                                      int(match.group(3)), int(match.group(4)))
        check_count(mode, count)
        check_offset(mode, offset)
        check_twist(mode, twist)
        return mode, count, offset, twist

    match = _COUNT_RE.match(variant)
    if match is None:
        raise MirrorError("未知的处理方式")

    mode = match.group(1)
    count = int(match.group(2))
    offset = int(match.group(3))
    check_count(mode, count)
    check_offset(mode, offset)
    return mode, count, offset, None


def step_key(step: Step) -> str:
    return variant_key(step.mode, step.count, step.offset, step.twist)


def chain_key(steps: list[Step]) -> str:
    """The variant token for a whole chain, e.g. 'd1-kaleido_6_45'."""
    if not steps:
        raise MirrorError("至少要选一个效果")
    if len(steps) > MAX_STEPS:
        raise MirrorError(f"最多叠加 {MAX_STEPS} 个效果")
    return CHAIN_SEP.join(step_key(s) for s in steps)


def parse_chain(variant: str) -> list[Step]:
    """Turn a variant token back into a list of Steps, validating every part."""
    parts = (variant or "").split(CHAIN_SEP)
    if not 1 <= len(parts) <= MAX_STEPS:
        raise MirrorError("效果链长度不对")
    return [Step(*parse_variant(part)) for part in parts]


def check_count(mode: str, count: int) -> None:
    allowed = COUNTS_BY_MODE.get(mode, ())
    if count not in allowed:
        label = COUNT_LABELS.get(mode, "参数")
        raise MirrorError(f"{label}只能是 {', '.join(map(str, allowed))}")


def check_offset(mode: str, offset: int) -> None:
    if mode not in RADIAL_MODES:
        if offset:
            raise MirrorError("这个模式没有起始角")
        return
    if not 0 <= offset < 360:
        raise MirrorError("起始角必须在 0 到 359 度之间")


def check_twist(mode: str, twist: int) -> None:
    if mode not in TWIST_MODES:
        if twist:
            raise MirrorError("这个模式没有扭曲")
        return
    if not 0 <= twist <= MAX_TWIST:
        raise MirrorError(f"扭曲必须在 0 到 {MAX_TWIST} 度之间")
    if twist % TWIST_STEP:
        raise MirrorError(f"扭曲必须是 {TWIST_STEP} 的倍数")


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

def _centre_square(frame: Image.Image) -> Image.Image:
    size = min(frame.size)
    left = (frame.width - size) // 2
    top = (frame.height - size) // 2
    return frame.crop((left, top, left + size, top + size))


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


def _triangle_mask(size: int, main: bool) -> Image.Image:
    """Antialiased mask for the half of a square on one side of a diagonal.

    Drawn at 4x and shrunk back so the diagonal edge isn't stair-stepped.
    """
    scale = 4
    big = Image.new("L", (size * scale, size * scale), 0)
    end = size * scale
    # main diagonal runs top-left to bottom-right, so its upper-right side is
    # the triangle (0,0)-(end,0)-(end,end)
    points = ([(0, 0), (end, 0), (end, end)] if main
              else [(0, 0), (end, 0), (0, end)])
    ImageDraw.Draw(big).polygon(points, fill=255)
    return big.resize((size, size), Image.LANCZOS)


def _diagonal_frame(frame: Image.Image, which: str) -> Image.Image:
    """Reflect a centred square across one of its diagonals.

    TRANSPOSE reflects across the main diagonal (top-left to bottom-right),
    TRANSVERSE across the anti-diagonal. Pasting the reflected copy over one
    triangle leaves the whole square symmetric about that line.
    """
    if which not in DIAGONALS:
        raise MirrorError(f"未知的对角方向：{which}")

    base = _centre_square(frame)
    size = base.width
    main = which == "d1"
    reflected = base.transpose(
        Image.Transpose.TRANSPOSE if main else Image.Transpose.TRANSVERSE
    )
    out = base.copy()
    out.paste(reflected, (0, 0), _triangle_mask(size, main))
    return out


def _quad_frame(frame: Image.Image) -> Image.Image:
    """Fold the top-left quadrant out to all four. Keeps the aspect ratio."""
    return _mirror_frame(_mirror_frame(frame, "l2r"), "t2b")


def _tile_frame(frame: Image.Image, repeat: int) -> Image.Image:
    """Shrink the frame and tile it `repeat` times each way, mirroring
    alternate cells so the seams line up.

    Neighbouring cells are reflections of each other, which is what makes the
    tiling seamless -- a plain repeat would show a hard edge wherever two
    copies meet. Output keeps the source's size and aspect ratio.
    """
    check_count(TILE, repeat)

    width, height = frame.size
    cell_w = math.ceil(width / repeat)
    cell_h = math.ceil(height / repeat)
    cell = frame.resize((cell_w, cell_h), Image.LANCZOS)

    flip_h = cell.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    flip_v = cell.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    flip_hv = flip_h.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for row in range(repeat):
        for col in range(repeat):
            if col % 2 == 0:
                piece = cell if row % 2 == 0 else flip_v
            else:
                piece = flip_h if row % 2 == 0 else flip_hv
            out.paste(piece, (col * cell_w, row * cell_h))
    return out


def _d4_cell(square: Image.Image) -> Image.Image:
    """Fold a square until it has all four mirror lines of a square (D4).

    Folding the whole square across a diagonal would undo any left-right and
    top-bottom symmetry already there, because it throws away one triangle
    outright. The fundamental domain of D4 is an eighth of the square, so the
    fold has to happen one level down: make a single quadrant symmetric about
    its own diagonal, then mirror that quadrant out to the other three.
    """
    side = square.width
    half = math.ceil(side / 2)
    quadrant = _diagonal_frame(square.crop((0, 0, half, half)), "d1")

    flip_h = quadrant.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    out.paste(quadrant, (0, 0))
    out.paste(flip_h, (side - half, 0))
    out.paste(quadrant.transpose(Image.Transpose.FLIP_TOP_BOTTOM), (0, side - half))
    out.paste(flip_h.transpose(Image.Transpose.FLIP_TOP_BOTTOM), (side - half, side - half))
    return out


def _papercut_frame(frame: Image.Image, repeat: int) -> Image.Image:
    """Tile a folded, fully symmetric motif across the frame.

    Named after cut-paper window flowers, which are made exactly this way:
    fold down to an eighth, cut, unfold. Only that eighth of the centre square
    survives -- unlike the plain tiling, which keeps the whole image.

    Each cell carries the full D4 symmetry of a square, so opposite edges are
    identical and plain repetition already lines up; no alternating flips
    needed. Every cell reads as a self-contained motif, unlike the plain tiling
    where a motif only completes across a 2x2 group.
    """
    check_count(PAPERCUT, repeat)

    width, height = frame.size
    side = math.ceil(max(width, height) / repeat)
    cell = _d4_cell(_centre_square(frame).resize((side, side), Image.LANCZOS))

    out = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for row in range(math.ceil(height / side)):
        for col in range(math.ceil(width / side)):
            out.paste(cell, (col * side, row * side))
    return out


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
    frame: Image.Image, mode: str, segments: int, offset: int,
    count_mode: str | None = None,
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
    # The spiral builds on the pinwheel assembly but has its own allowed
    # segment counts, so validation can target a different mode than the build.
    check_count(count_mode or mode, segments)
    check_offset(count_mode or mode, offset)

    base = _centre_square(frame)
    size = base.width

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


def _twist_frame(frame: Image.Image, degrees: int, cells: int = 56) -> Image.Image:
    """Rotate each pixel about the centre by an amount that depends on radius.

    Because the rotation only ever depends on the radius, the warp commutes
    with rotation about the same centre -- so an n-fold pattern stays exactly
    n-fold after twisting. That is what lets the spiral reuse the pinwheel
    assembly and still count as a symmetry rather than a plain distortion.

    The twist is strongest at the centre and fades to zero at the *inscribed*
    circle, not at the corners. That boundary matters: rotation preserves
    radius, so a point inside the inscribed circle always lands on another
    point inside it, which is guaranteed to have source pixels. Fading out at
    the corner radius instead would rotate near-corner points onto positions
    the square doesn't cover, punching transparent holes into the result.

    Implemented as one Pillow MESH transform over a `cells` x `cells` grid:
    Pillow interpolates inside each quad in C, so this stays fast without
    needing a per-pixel loop or numpy.
    """
    if not degrees:
        return frame

    width, height = frame.size
    cx, cy = width / 2, height / 2
    rim = min(cx, cy)
    strength = math.radians(degrees)
    falloff = math.log1p(rim)

    def source(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        radius = math.hypot(dx, dy)
        if radius < 1e-9:
            return cx, cy
        if radius >= rim:
            return x, y
        angle = math.atan2(dy, dx) + strength * (1 - math.log1p(radius) / falloff)
        return cx + radius * math.cos(angle), cy + radius * math.sin(angle)

    mesh = []
    for row in range(cells):
        y0, y1 = height * row / cells, height * (row + 1) / cells
        for col in range(cells):
            x0, x1 = width * col / cells, width * (col + 1) / cells
            box = (int(x0), int(y0), math.ceil(x1), math.ceil(y1))
            # Pillow wants the source quad as upper-left, lower-left,
            # lower-right, upper-right.
            quad = [
                c
                for point in (source(x0, y0), source(x0, y1),
                              source(x1, y1), source(x1, y0))
                for c in point
            ]
            mesh.append((box, quad))

    return frame.transform(
        (width, height), Image.MESH, mesh, resample=Image.BICUBIC
    )


def _spiral_frame(
    frame: Image.Image, segments: int, offset: int, twist: int
) -> Image.Image:
    """An n-armed pinwheel with a logarithmic twist applied on top."""
    check_twist(SPIRAL, twist)
    base = _radial_frame(frame, PINWHEEL, segments, offset, count_mode=SPIRAL)
    return _twist_frame(base, twist)


def _transform(
    frame: Image.Image, mode: str, count: int | None, offset: int | None,
    twist: int | None = None,
) -> Image.Image:
    if mode == SPIRAL:
        return _spiral_frame(
            frame,
            default_count(SPIRAL) if count is None else count,
            DEFAULT_OFFSET if offset is None else offset,
            DEFAULT_TWIST if twist is None else twist,
        )
    if mode in RADIAL_MODES:
        return _radial_frame(
            frame,
            mode,
            default_count(mode) if count is None else count,
            DEFAULT_OFFSET if offset is None else offset,
        )
    if mode == TILE:
        return _tile_frame(frame, default_count(TILE) if count is None else count)
    if mode == PAPERCUT:
        return _papercut_frame(frame, default_count(PAPERCUT) if count is None else count)
    if mode == QUAD:
        return _quad_frame(frame)
    if mode in DIAGONALS:
        return _diagonal_frame(frame, mode)
    return _mirror_frame(frame, mode)


# --------------------------------------------------------------------------
# File-level entry points
# --------------------------------------------------------------------------

def _apply_steps(frame: Image.Image, steps: list[Step]) -> Image.Image:
    """Run a whole chain over one frame, in order."""
    for step in steps:
        frame = _transform(frame, step.mode, step.count, step.offset, step.twist)
    return frame


def _flip_static(src: Path, dst: Path, steps: list[Step]) -> None:
    with Image.open(src) as im:
        im.load()
        frame = im.convert("RGBA")

    out = _apply_steps(frame, steps)

    if dst.suffix.lower() in (".jpg", ".jpeg"):
        # JPEG has no alpha channel, so composite onto white first
        canvas = Image.new("RGB", out.size, (255, 255, 255))
        canvas.paste(out, mask=out.getchannel("A"))
        canvas.save(dst, quality=95, subsampling=0)
    else:
        out.save(dst)


def _flip_animated(src: Path, dst: Path, steps: list[Step]) -> None:
    frames: list[Image.Image] = []
    durations: list[int] = []

    with Image.open(src) as im:
        default_duration = im.info.get("duration", 100)
        loop = im.info.get("loop", 0)
        for frame in ImageSequence.Iterator(im):
            # convert() composites the current frame per the GIF disposal rules
            frames.append(_apply_steps(frame.convert("RGBA"), steps))
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


def flip_chain(src: Path, dst: Path, steps: list[Step], animated: bool) -> None:
    """Apply a chain of effects to src and write the result to dst.

    Steps run in the order given. Note that some of them change the frame's
    dimensions -- the radial and diagonal modes crop to a centred square -- so
    the order genuinely matters.
    """
    if not steps:
        raise MirrorError("至少要选一个效果")
    if len(steps) > MAX_STEPS:
        raise MirrorError(f"最多叠加 {MAX_STEPS} 个效果")
    for step in steps:
        if step.mode not in MODES:
            raise MirrorError(f"未知的处理方式：{step.mode}")

    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if animated:
            _flip_animated(src, dst, steps)
        else:
            _flip_static(src, dst, steps)
    except MirrorError:
        raise
    except OSError as exc:
        raise MirrorError(f"处理失败：{exc}") from None


def flip(
    src: Path,
    dst: Path,
    mode: str,
    animated: bool,
    count: int | None = None,
    offset: int | None = None,
    twist: int | None = None,
) -> None:
    """Single-effect convenience wrapper around flip_chain().

    `count`, `offset` and `twist` only apply to the modes that take them and
    fall back to that mode's defaults.
    """
    flip_chain(src, dst, [Step(mode, count, offset, twist)], animated)
