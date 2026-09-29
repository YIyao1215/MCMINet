"""Deterministic release MRI-only preprocessing; raw files are read-only.

Volume axes are [row, column, slice], with a voxel-to-RAS-mm affine. Slice
positions increase inferior-to-superior. Only axial acquisitions are supported;
oblique/nonaxial acquisitions require review, not implicit MRI resampling.
Missing physical geometry is rejected: InstanceNumber cannot establish mask
alignment and is deliberately not used as a fallback. The manifest series is
upstream-selected; no DCE phase identification is performed.

Z-score uses all finite image-volume voxels (including finite zero background),
after stored-value rescaling, once per patient. Nonfinite volumes are rejected.
Masks must contain binary {0,1} values. Crops retain every MRI pixel inside the
mask-derived rectangle. No resizing, pixel masking, augmentation or WSI work.
"""
from mcminet.config import default

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
import warnings

import nibabel as nib
import numpy as np
import pydicom
from scipy.ndimage import affine_transform, distance_transform_edt
import torch

from mcminet.data.real_data_manifest import RawPatientRecord
from mcminet.data.raw_patient_validator import validate_dicom_directory


@dataclass(frozen=True)
class MRIVolume:
    values: np.ndarray
    affine_ras_mm: np.ndarray
    spacing_mm: tuple[float, float, float]
    slice_positions_mm: tuple[float, ...]


@dataclass(frozen=True)
class MRIPreprocessingResult:
    patient_id: str
    tumor_roi: torch.Tensor
    peritumoral_roi: torch.Tensor
    lymph_node_roi: torch.Tensor
    tumor_slice_index: int
    lymph_node_slice_index: int
    tumor_slice_position_mm: float
    lymph_node_slice_position_mm: float
    tumor_bbox: tuple[int, int, int, int]
    peritumoral_bbox: tuple[int, int, int, int]
    lymph_node_bbox: tuple[int, int, int, int]
    original_mri_shape: tuple[int, ...]
    original_spacing_mm: tuple[float, ...]
    tumor_foreground_voxels: int
    lymph_node_foreground_voxels: int
    tumor_slice_area: int
    lymph_node_slice_area: int
    peritumoral_area: int
    peritumoral_boundary_clipped: bool
    mask_resampling_performed: bool
    tumor_mask_resampled: bool
    lymph_node_mask_resampled: bool
    normalization_mean: float
    normalization_std: float
    warnings: tuple[str, ...]


def load_mri_volume(directory: str | Path) -> MRIVolume:
    """Decode a single regular axial series, never use display/window LUTs."""
    summary = validate_dicom_directory(directory)
    if summary.errors or summary.series_count != 1:
        raise ValueError('DICOM: one valid, unambiguous image series is required. ' + ' '.join(summary.errors))
    if summary.unreadable_files:
        raise ValueError('DICOM: unreadable candidates require review before pixel loading.')
    series = summary.series[0]
    if any(k in series.missing_metadata for k in ('ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing')):
        raise ValueError('DICOM: missing physical geometry; InstanceNumber fallback is not safe for mask alignment.')
    slices = []
    try:
        for path in sorted(Path(directory).rglob('*')):
            if not path.is_file():
                continue
            with warnings.catch_warnings(record=True):
                warnings.simplefilter('always')
                with path.open('rb') as handle:
                    ds = pydicom.dcmread(handle, force=True)
                orientation = np.asarray(ds.ImageOrientationPatient, dtype=float)
                position = np.asarray(ds.ImagePositionPatient, dtype=float)
                spacing = np.asarray(ds.PixelSpacing, dtype=float)
                slope = float(getattr(ds, 'RescaleSlope', 1.))
                intercept = float(getattr(ds, 'RescaleIntercept', 0.))
                pixels = np.asarray(ds.pixel_array, dtype=np.float64) * slope + intercept
            if (orientation.shape != (6,) or position.shape != (3,) or spacing.shape != (2,)
                    or pixels.shape != (int(ds.Rows), int(ds.Columns))
                    or not all(np.isfinite(a).all() for a in (orientation, position, spacing, pixels))
                    or np.any(spacing <= 0) or not np.isfinite(slope) or slope == 0 or not np.isfinite(intercept)):
                raise ValueError
            slices.append((orientation, position, spacing, pixels))
    except Exception as exc:
        raise ValueError(f'DICOM pixel/geometry loading failed ({type(exc).__name__}); reader details withheld.') from None
    if len(slices) < 2:
        raise ValueError('DICOM: at least two physical slice positions required.')
    orientation, _, spacing, _ = slices[0]
    u, v = orientation[:3], orientation[3:]
    normal = np.cross(u, v)
    if not np.allclose([np.linalg.norm(u), np.linalg.norm(v), u @ v], [1, 1, 0], atol=1e-5):
        raise ValueError('DICOM: invalid orientation cosines.')
    if not np.allclose(np.abs(normal), [0, 0, 1], atol=1e-4):
        raise ValueError('DICOM: nonaxial/oblique acquisition requires geometry review; MRI not resampled.')
    if normal[2] < 0:
        normal = -normal
    slices.sort(key=lambda s: float(s[1] @ normal))
    positions = np.array([s[1] for s in slices])
    differences = np.diff(positions, axis=0)
    step = differences.mean(axis=0)
    if (step @ normal <= 1e-5 or not np.allclose(differences, step, atol=1e-3, rtol=1e-4)
            or not np.allclose(step, normal * (step @ normal), atol=1e-3)):
        raise ValueError('DICOM: duplicate, nonuniform or ambiguous physical slice positions.')
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack((v * spacing[0], u * spacing[1], step))
    affine[:3, 3] = positions[0]
    affine = np.diag([-1., -1., 1., 1.]) @ affine
    volume = np.stack([s[3] for s in slices], axis=2)
    return MRIVolume(volume, affine, (float(spacing[0]), float(spacing[1]), float(np.linalg.norm(step))),
                     tuple(float(p @ normal) for p in positions))


def align_mask(path: str | Path, volume: MRIVolume) -> tuple[np.ndarray, bool]:
    """Map NIfTI voxel centers into the MRI grid. Exact signed permutations
    use transpose/flip only; other grids use order=0 nearest neighbor, zero
    outside source extent. Return whether interpolation/resampling occurred.
    """
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            image = nib.load(str(path), mmap='r', keep_file_open=False)
            data = np.asanyarray(image.dataobj)
        scale = {'mm': 1., 'meter': 1000., 'micron': .001}.get(image.header.get_xyzt_units()[0])
        if (data.ndim != 3 or not data.size or not np.isfinite(data).all()
                or not np.isin(data, [0, 1]).all() or not np.any(data)):
            raise ValueError('Expected a nonempty finite binary 3D mask.')
        if scale is None or (int(image.header['qform_code']) == 0 and int(image.header['sform_code']) == 0):
            raise ValueError('Mask requires physical units and a coded affine.')
        affine = np.array(image.affine, dtype=float, copy=True)
        affine[:3] *= scale
        if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
            raise ValueError('Mask affine is nonfinite or singular.')
        if not np.allclose(np.linalg.norm(affine[:3, :3], axis=0), np.array(image.header.get_zooms()) * scale, atol=1e-3):
            raise ValueError('Mask affine and header spacing disagree.')
        transform = np.linalg.solve(affine, volume.affine_ras_mm)
        for order in permutations(range(3)):
            linear = transform[:3, :3]
            signs = [1 if linear[axis, j] >= 0 else -1 for j, axis in enumerate(order)]
            expected = np.zeros((3, 3))
            origin = np.zeros(3)
            for j, axis in enumerate(order):
                expected[axis, j] = signs[j]
                if signs[j] < 0:
                    origin[axis] = data.shape[axis] - 1
            if (tuple(data.shape[a] for a in order) == volume.values.shape
                    and np.allclose(linear, expected, atol=1e-4, rtol=0)
                    and np.allclose(transform[:3, 3], origin, atol=1e-3, rtol=0)):
                aligned = np.transpose(data, order)
                for j, sign in enumerate(signs):
                    if sign < 0:
                        aligned = np.flip(aligned, axis=j)
                return aligned.astype(bool, copy=True), False
        aligned = affine_transform(data, transform[:3, :3], offset=transform[:3, 3],
                                   output_shape=volume.values.shape, order=0, mode='constant', cval=0, prefilter=False)
        if not np.any(aligned):
            raise ValueError('Mask foreground does not map onto MRI geometry.')
        return aligned.astype(bool), True
    except (OSError, nib.filebasedimages.ImageFileError):
        raise ValueError('Mask file cannot be read.') from None


def normalize_volume(values: np.ndarray, epsilon: float = 1e-8) -> tuple[np.ndarray, float, float]:
    """Population z-score over the entire finite volume, including background."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 3 or not data.size or not np.isfinite(data).all():
        raise ValueError('MRI must be a nonempty finite 3D volume.')
    mean, std = float(data.mean()), float(data.std())
    if not np.isfinite(std) or std <= epsilon:
        raise ValueError('MRI has zero or insufficient intensity variance.')
    normalized = (data - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError('MRI normalization produced nonfinite values.')
    return normalized, mean, std


def representative_slice(mask: np.ndarray) -> int:
    """Largest cross-section; ties choose first physically ordered slice.
    For LN this is a computational proxy, not a short-axis measurement.
    """
    if mask.ndim != 3 or not mask.size or not np.any(mask):
        raise ValueError('Mask has no foreground in a nonempty 3D grid.')
    return int(np.argmax(np.count_nonzero(mask, axis=(0, 1))))


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (row_start, row_stop, column_start, column_stop), stops exclusive."""
    if mask.ndim != 2 or not mask.size or not np.any(mask):
        raise ValueError('Cannot crop an empty 2D foreground.')
    rows, columns = np.nonzero(mask)
    return int(rows.min()), int(rows.max() + 1), int(columns.min()), int(columns.max() + 1)


def peritumoral_ring(mask: np.ndarray, spacing_mm: tuple[float, float]) -> tuple[np.ndarray, bool]:
    """2D center-to-center Euclidean distance <=4 mm, excluding tumor."""
    tight_bbox(mask)
    spacing = np.asarray(spacing_mm, dtype=float)
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Invalid in-plane physical spacing.')
    foreground = np.asarray(mask, dtype=bool)
    distance = distance_transform_edt(~foreground, sampling=spacing)
    ring = (distance <= default("MRI.peritumoral_radius_mm")) & ~foreground
    if not ring.any():
        raise ValueError('Peritumoral mask is empty.')
    r0, r1, c0, c1 = tight_bbox(foreground)
    # Would any pixel center just beyond the image lie within 4 mm?
    clipped = bool(min((r0 + 1) * spacing[0], (mask.shape[0] - r1 + 1) * spacing[0],
                       (c0 + 1) * spacing[1], (mask.shape[1] - c1 + 1) * spacing[1]) <= default("MRI.peritumoral_radius_mm"))
    return ring, clipped


def preprocess_mri(record: RawPatientRecord) -> MRIPreprocessingResult:
    """MRI-only result; no PatientSample or model invocation."""
    volume = load_mri_volume(record.mri_dicom_dir)
    tumor, tumor_resampled = align_mask(record.tumor_mask_path, volume)
    ln, ln_resampled = align_mask(record.lymph_node_mask_path, volume)
    normalized, mean, std = normalize_volume(volume.values)
    zt, zl = representative_slice(tumor), representative_slice(ln)
    ring, clipped = peritumoral_ring(tumor[:, :, zt], volume.spacing_mm[:2])
    bt, bp, bl = tight_bbox(tumor[:, :, zt]), tight_bbox(ring), tight_bbox(ln[:, :, zl])
    def crop(z, box):
        r0, r1, c0, c1 = box
        result = torch.from_numpy(normalized[r0:r1, c0:c1, z].copy()).to(torch.float32).unsqueeze(0)
        if not torch.isfinite(result).all():
            raise ValueError('ROI contains nonfinite values.')
        return result
    notes = []
    if tumor_resampled:
        notes.append('Tumor mask resampled onto MRI grid using nearest neighbor; review alignment/coverage.')
    if ln_resampled:
        notes.append('LN mask resampled onto MRI grid using nearest neighbor; review alignment/coverage.')
    if clipped:
        notes.append('4 mm peritumoral expansion clipped by image boundary.')
    return MRIPreprocessingResult(record.patient_id, crop(zt, bt), crop(zt, bp), crop(zl, bl),
        zt, zl, volume.slice_positions_mm[zt], volume.slice_positions_mm[zl], bt, bp, bl,
        volume.values.shape, volume.spacing_mm, int(tumor.sum()), int(ln.sum()),
        int(tumor[:, :, zt].sum()), int(ln[:, :, zl].sum()), int(ring.sum()), clipped,
        tumor_resampled or ln_resampled, tumor_resampled, ln_resampled, mean, std, tuple(notes))
