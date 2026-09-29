"""release: deterministic native-40x fields, no model or feature extraction.

448x448 level-0 fields become full-field bilinear 224x224 ImageNet-normalized
RGB tensors. Coordinates are slide-local level-0 top-left [x,y] integer pixels;
they are not cross-slide physical distances. Grid order is y-major then x-major.

Pyramid RGB cannot bound native near-white pixel counts (downsampling loses
information). Stage 1 therefore supplies coarse QC only, rejecting ZERO grid
locations. Stage 2 scans every full native field. This correctness fallback is
intentional, not an approximate tissue mask. The iterator keeps one patch at a
time; the optional collector has an explicit memory limit and never truncates.
"""
from mcminet.config import default

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import math
import time

import numpy as np
from PIL import Image
import torch

from mcminet.data.real_data_manifest import RawPatientRecord
from mcminet.data.raw_patient_validator import _openslide_reader

NATIVE_SIZE = default("WSI.preprocessing.patch_size")
STRIDE = default("WSI.preprocessing.stride")
MODEL_SIZE = default("WSI.preprocessing.model_size")
BLANK_FRACTION_THRESHOLD = default("WSI.preprocessing.legacy_rgb_blank_fraction_max")
IMAGENET_MEAN = np.array(default("WSI.preprocessing.imagenet_mean"), dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array(default("WSI.preprocessing.imagenet_std"), dtype=np.float32)[:, None, None]


@dataclass(frozen=True)
class SlideMetadata:
    dimensions: tuple[int, int]
    objective_power: float
    mpp_x: float
    mpp_y: float
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    sampling_level: int = 0


@dataclass(frozen=True)
class CoarseScreen:
    level: int
    dimensions: tuple[int, int]
    blank_fraction: float
    rejected_candidates: int = 0
    warning: str = 'Pyramid pixels cannot safely exclude native fields; all full-grid fields require native evaluation.'


@dataclass(frozen=True)
class PatchEvaluation:
    coordinate: tuple[int, int]
    blank_fraction: float
    patch: torch.Tensor | None  # None means rejected by final native criterion


@dataclass(frozen=True)
class WSIPreprocessingResult:
    patient_id: str
    patches: torch.Tensor
    coordinates: torch.Tensor
    metadata: SlideMetadata
    candidate_patch_count: int
    retained_patch_count: int
    rejected_blank_count: int
    blank_fraction_summary: tuple[float, float, float]
    runtime_seconds: float
    warnings: tuple[str, ...]
    native_patch_size: int = NATIVE_SIZE
    model_patch_size: int = MODEL_SIZE
    stride: int = STRIDE
    blank_threshold: int = default("WSI.preprocessing.legacy_rgb_threshold")
    blank_fraction_threshold: float = BLANK_FRACTION_THRESHOLD


def inspect_metadata(slide) -> SlideMetadata:
    """Support native 40x only. 0.18..0.30 um/px is an explicit plausibility
    gate, not a universal magnification conversion. Require both MPP axes and
    <=5% anisotropy; unsupported metadata requires manual review.
    """
    def number(key):
        try:
            value = float(slide.properties[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'Missing or invalid {key}; native 40x cannot be established.') from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid {key}.')
        return value
    objective = number('openslide.objective-power')
    mx, my = number('openslide.mpp-x'), number('openslide.mpp-y')
    if not math.isclose(objective, 40., abs_tol=.01):
        raise ValueError('Only metadata-confirmed native 40x is supported; no upsampling of other objectives.')
    if not (.18 <= mx <= .30 and .18 <= my <= .30) or max(mx, my) / min(mx, my) > 1.05:
        raise ValueError('MPP contradicts the supported native-40x plausibility range or isotropy; review required.')
    dimensions = tuple(map(int, slide.dimensions))
    levels = tuple(tuple(map(int, d)) for d in slide.level_dimensions)
    downsamples = tuple(map(float, slide.level_downsamples))
    if (len(dimensions) != 2 or min(dimensions) <= 0 or not levels
            or len(levels) != int(slide.level_count) or len(downsamples) != len(levels)
            or levels[0] != dimensions or not math.isclose(downsamples[0], 1.)
            or any(len(d) != 2 or min(d) <= 0 for d in levels)
            or any(not math.isfinite(d) or d <= 0 for d in downsamples)):
        raise ValueError('Invalid slide pyramid metadata.')
    for i, (dims, downsample) in enumerate(zip(levels, downsamples)):
        if i and downsamples[i-1] >= downsample:
            raise ValueError('Pyramid downsamples must increase.')
        if any(abs(base / downsample - size) > 2 for base, size in zip(dimensions, dims)):
            raise ValueError('Pyramid dimensions contradict downsamples.')
    return SlideMetadata(dimensions, objective, mx, my, levels, downsamples)


def patch_grid(dimensions: tuple[int, int]):
    """Full fields only, anchored at (0,0), no padding or partial border fields."""
    width, height = dimensions
    for y in range(0, height - NATIVE_SIZE + 1, STRIDE):
        for x in range(0, width - NATIVE_SIZE + 1, STRIDE):
            yield x, y


def candidate_count(metadata: SlideMetadata) -> int:
    width, height = metadata.dimensions
    return (width // NATIVE_SIZE) * (height // NATIVE_SIZE)


def to_rgb(image: Image.Image) -> Image.Image:
    """Explicit RGB conversion; transparent areas composite onto white."""
    if image.mode == 'RGBA':
        white = Image.new('RGBA', image.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, image).convert('RGB')
    return image.convert('RGB')


def blank_fraction(rgb: Image.Image | np.ndarray, threshold: int = default("WSI.preprocessing.legacy_rgb_threshold")) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, int) or not 0 <= threshold <= 255:
        raise ValueError('Near-white threshold must be an integer in [0,255].')
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or not array.size or array.dtype != np.uint8:
        raise ValueError('Blank detection requires nonempty uint8 RGB pixels.')
    return float(np.count_nonzero(np.all(array >= threshold, axis=2)) / (array.shape[0] * array.shape[1]))


def retain_fraction(fraction: float) -> bool:
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('Invalid blank fraction.')
    return fraction <= BLANK_FRACTION_THRESHOLD


def patch_tensor(rgb: Image.Image) -> torch.Tensor:
    if rgb.mode != 'RGB' or rgb.size != (NATIVE_SIZE, NATIVE_SIZE):
        raise ValueError('Expected native 448x448 RGB field.')
    resized = rgb.resize((MODEL_SIZE, MODEL_SIZE), resample=Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / np.float32(255.)
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array))


def coarse_screen(slide, metadata: SlideMetadata, threshold: int = default("WSI.preprocessing.legacy_rgb_threshold")) -> CoarseScreen:
    """Bounded low-resolution overview; no unsound tissue-based exclusions."""
    candidates = [i for i, dims in enumerate(metadata.level_dimensions) if dims[0] * dims[1] <= 1024 * 1024]
    if not candidates:
        return CoarseScreen(-1, (0, 0), float('nan'), warning='No bounded pyramid overview available; native scan required.')
    level = candidates[0]
    dims = metadata.level_dimensions[level]
    overview = to_rgb(slide.read_region((0, 0), level, dims))
    if overview.size != dims:
        raise ValueError('Coarse reader returned incorrect dimensions.')
    return CoarseScreen(level, dims, blank_fraction(overview, threshold))


def iter_patch_evaluations(slide, metadata: SlideMetadata, blank_threshold: int = default("WSI.preprocessing.legacy_rgb_threshold")):
    """Caller owns slide lifetime. Yield every native evaluation in grid order.
    No region larger than 448x448 is requested here. Native blank fraction is
    computed before tensor conversion; rejected fields never get resized.
    """
    for coordinate in patch_grid(metadata.dimensions):
        rgb = to_rgb(slide.read_region(coordinate, 0, (NATIVE_SIZE, NATIVE_SIZE)))
        if rgb.size != (NATIVE_SIZE, NATIVE_SIZE):
            raise ValueError('Native reader returned incorrect field dimensions.')
        fraction = blank_fraction(rgb, blank_threshold)
        yield PatchEvaluation(coordinate, fraction, patch_tensor(rgb) if retain_fraction(fraction) else None)


def preprocess_wsi(record: RawPatientRecord, slide_opener: Callable | None = None,
                   blank_threshold: int = default("WSI.preprocessing.legacy_rgb_threshold"), max_tensor_bytes: int = 1024**3) -> WSIPreprocessingResult:
    """Collect complete tensors, or raise on empty tissue/memory limit. Never
    return a truncated result. Budget includes list tensors plus final stack.
    For large scans consume iter_patch_evaluations incrementally instead.
    """
    if max_tensor_bytes <= 0:
        raise ValueError('Tensor memory budget must be positive.')
    start = time.monotonic()
    slide = (slide_opener or _openslide_reader())(str(record.wsi_path))
    try:
        metadata = inspect_metadata(slide)
        coarse = coarse_screen(slide, metadata, blank_threshold)
        patches, coordinates, fractions = [], [], []
        for evaluation in iter_patch_evaluations(slide, metadata, blank_threshold):
            fractions.append(evaluation.blank_fraction)
            if evaluation.patch is not None:
                if (len(patches) + 1) * 3 * MODEL_SIZE**2 * 4 * 2 > max_tensor_bytes:
                    raise MemoryError('Full tensor collection exceeds memory budget; use streaming extraction. No complete result produced.')
                patches.append(evaluation.patch)
                coordinates.append(evaluation.coordinate)
        if not patches:
            raise ValueError('No retained full tissue patches; no model input produced.')
        count = candidate_count(metadata)
        return WSIPreprocessingResult(record.patient_id, torch.stack(patches), torch.tensor(coordinates, dtype=torch.int64),
            metadata, count, len(patches), count-len(patches),
            (float(np.min(fractions)), float(np.median(fractions)), float(np.max(fractions))),
            time.monotonic()-start, (coarse.warning,), blank_threshold=blank_threshold)
    finally:
        slide.close()
