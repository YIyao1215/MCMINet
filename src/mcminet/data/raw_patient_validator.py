"""Read-only raw diagnostics, stopping before preprocessing or tensor creation.

DICOM metadata are allowlisted and held in memory only. Reports must not dump
instance metadata, UIDs, arbitrary text descriptions, exception text or private
properties. No pixel decoding is performed for DICOM or WSI; their readability
here means metadata/header readability, not exhaustive pixel integrity.
"""
from dataclasses import dataclass, field
from itertools import product, permutations
from pathlib import Path
from typing import Any, Callable
import importlib
import warnings as python_warnings

import numpy as np

from mcminet.data.real_data_manifest import RawPatientRecord

DICOM_TAGS = (
    'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID', 'InstanceNumber',
    'Rows', 'Columns', 'PixelSpacing', 'SliceThickness', 'SpacingBetweenSlices',
    'ImagePositionPatient', 'ImageOrientationPatient', 'SeriesDescription',
    'ProtocolName', 'SequenceName', 'TemporalPositionIdentifier', 'AcquisitionNumber',
    'EchoTime', 'RepetitionTime', 'NumberOfFrames',
)
GEOMETRY = ('Rows', 'Columns', 'PixelSpacing', 'ImagePositionPatient', 'ImageOrientationPatient')


@dataclass
class SeriesSummary:
    instances: list[dict[str, Any]] = field(default_factory=list, repr=False)
    values: dict[str, list[Any]] = field(default_factory=dict, repr=False)
    instance_number_range: tuple[int, int] | None = None
    missing_metadata: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def instance_count(self) -> int:
        return len(self.instances)


@dataclass
class DICOMSummary:
    inspected_files: int = 0
    readable_files: int = 0
    unreadable_files: int = 0
    series: list[SeriesSummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def series_count(self) -> int:
        return len(self.series)


@dataclass
class MaskSummary:
    filename: str
    shape: tuple[int, ...] = ()
    ndim: int | None = None
    dtype: str | None = None
    affine: list[list[float]] | None = None
    voxel_sizes: tuple[float, ...] = ()
    spatial_unit: str | None = None
    affine_codes: tuple[int, int] | None = None
    minimum_finite_value: float | None = None
    maximum_finite_value: float | None = None
    finite_voxel_count: int = 0
    nan_count: int = 0
    inf_count: int = 0
    nonzero_voxel_count: int = 0
    nonzero_fraction: float = 0.0
    unique_finite_values: list[float] | None = None
    mask_value_class: str = 'unknown'
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class Compatibility:
    status: str
    reasons: list[str]


@dataclass
class WSISummary:
    filename: str
    readability: str = 'unreadable'
    level_count: int | None = None
    level_dimensions: tuple[tuple[int, int], ...] = ()
    level_downsamples: tuple[float, ...] = ()
    level0_dimensions: tuple[int, int] | None = None
    vendor: str | None = None
    objective_power: str | None = None
    mpp_x: str | None = None
    mpp_y: str | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class PatientValidation:
    patient_id: str
    label: int
    cohort: str
    filesystem: dict[str, dict[str, Any]]
    dicom: DICOMSummary
    tumor_mask: MaskSummary
    lymph_node_mask: MaskSummary
    tumor_mri_compatibility: Compatibility
    lymph_node_mri_compatibility: Compatibility
    wsi: WSISummary
    valid_for_next_preprocessing_stage: bool
    errors: list[str]
    warnings: list[str]


def _file_status(path: Path, directory: bool = False) -> dict[str, Any]:
    try:
        if not path.exists():
            return {'ok': False, 'reason': 'Path does not exist.'}
        if directory:
            if not path.is_dir():
                return {'ok': False, 'reason': 'Expected a directory.'}
            list(path.iterdir())
        else:
            if not path.is_file():
                return {'ok': False, 'reason': 'Expected a regular file.'}
            with path.open('rb') as handle:
                handle.read(1)
        return {'ok': True, 'reason': 'Readable directory.' if directory else 'Readable regular file.'}
    except OSError as exc:
        return {'ok': False, 'reason': f'Filesystem access failed ({type(exc).__name__}).'}


def _plain(value):
    if isinstance(value, (str, int, float)):
        return value if type(value) in (str, int, float) else str(value)
    try:
        return tuple(_plain(v) for v in value)
    except TypeError:
        return str(value)


def validate_dicom_directory(path: str | Path) -> DICOMSummary:
    """Inspect all regular files recursively, regardless of extension.

    Preamble-free datasets are retried with force=True but must contain a
    SOPClassUID plus image Rows/Columns; arbitrary byte streams are not accepted.
    Multi-frame objects are reported but not treated as simple slice volumes.
    """
    result = DICOMSummary()
    path = Path(path)
    status = _file_status(path, directory=True)
    if not status['ok']:
        result.errors.append(status['reason'])
        return result
    try:
        pydicom = importlib.import_module('pydicom')
    except ImportError:
        result.errors.append('Dependency blocked: pydicom Python package unavailable.')
        return result
    groups = {}
    try:
        files = sorted(p for p in path.rglob('*') if p.is_file())
    except OSError as exc:
        result.errors.append(f'Directory traversal failed ({type(exc).__name__}).')
        return result
    for ordinal, candidate in enumerate(files, 1):
        result.inspected_files += 1
        try:
            with python_warnings.catch_warnings(record=True) as caught:
                python_warnings.simplefilter('always')
                with candidate.open('rb') as handle:
                    try:
                        ds = pydicom.dcmread(handle, stop_before_pixels=True, specific_tags=(*DICOM_TAGS, 'SOPClassUID'))
                    except pydicom.errors.InvalidDicomError:
                        handle.seek(0)
                        ds = pydicom.dcmread(handle, stop_before_pixels=True, force=True,
                                             specific_tags=(*DICOM_TAGS, 'SOPClassUID'))
                if not getattr(ds, 'SOPClassUID', None) or not getattr(ds, 'Rows', None) or not getattr(ds, 'Columns', None):
                    raise ValueError('Not an image DICOM candidate')
                metadata = {key: _plain(getattr(ds, key)) if key in ds else None for key in DICOM_TAGS}
            if caught:
                result.warnings.append(f'Candidate {ordinal}: DICOM reader emitted {len(caught)} warning(s); text suppressed for privacy.')
            groups.setdefault(metadata['SeriesInstanceUID'], []).append(metadata)
            result.readable_files += 1
        except Exception as exc:
            result.unreadable_files += 1
            result.warnings.append(f'Candidate {ordinal}: unreadable as image DICOM ({type(exc).__name__}); filename omitted for privacy.')
    for uid, instances in groups.items():
        series = SeriesSummary(instances=instances)
        for key in DICOM_TAGS:
            values = [r[key] for r in instances if r[key] is not None and r[key] != '']
            series.values[key] = list(dict.fromkeys(values))
            if len(values) != len(instances):
                series.missing_metadata[key] = len(instances) - len(values)
        if series.missing_metadata:
            series.warnings.append('Missing metadata: ' + ', '.join(series.missing_metadata) + '.')
        if uid is None or not uid:
            series.errors.append('Missing SeriesInstanceUID; cannot identify a candidate series.')
        for key in ('Rows', 'Columns', 'PixelSpacing', 'ImageOrientationPatient'):
            if len(series.values[key]) > 1:
                # Numeric geometry comparisons tolerate serialization precision.
                try:
                    arrays = [np.asarray(v, dtype=float) for v in series.values[key]]
                    inconsistent = any(a.shape != arrays[0].shape or not np.allclose(a, arrays[0], atol=1e-5, rtol=1e-5) for a in arrays[1:])
                except (ValueError, TypeError):
                    inconsistent = True
                if inconsistent:
                    series.errors.append(f'Inconsistent {key} within series.')
        for key in ('SOPInstanceUID', 'InstanceNumber'):
            values = [r[key] for r in instances if r[key] is not None]
            if len(set(values)) < len(values):
                target = series.errors if key == 'SOPInstanceUID' else series.warnings
                target.append(f'Duplicate {key} within series.')
        numbers = series.values['InstanceNumber']
        if numbers:
            try:
                series.instance_number_range = (min(map(int, numbers)), max(map(int, numbers)))
            except (ValueError, TypeError):
                series.warnings.append('Invalid InstanceNumber values.')
        if any(key in series.missing_metadata for key in GEOMETRY):
            series.warnings.append('Missing geometry metadata; spatial compatibility may be indeterminate.')
        try:
            if any(int(v) != 1 for v in series.values['NumberOfFrames']):
                series.errors.append('Multi-frame DICOM requires a later dedicated geometry reader.')
        except (ValueError, TypeError):
            series.errors.append('Invalid NumberOfFrames.')
        result.series.append(series)
    if not result.readable_files:
        result.errors.append('No readable image DICOM candidates.')
    if result.series_count > 1:
        result.errors.append('Multiple SeriesInstanceUID values: no series selected; MRI preprocessing is blocked.')
    for i, series in enumerate(result.series, 1):
        result.errors.extend(f'Series {i}: {e}' for e in series.errors)
        result.warnings.extend(f'Series {i}: {w}' for w in series.warnings)
    return result


def validate_nifti_mask(path: str | Path) -> MaskSummary:
    """Inspect stored image values without thresholding, casting or rewriting.

    Nonzero count uses finite nonzero voxels; NaN/Inf are counted separately.
    Small means <=32 unique finite values. Two values including zero are
    binary_like; small integral sets are integer_label_like, otherwise continuous.
    """
    path = Path(path)
    result = MaskSummary(path.name)
    status = _file_status(path)
    if not status['ok']:
        result.errors.append(status['reason'])
        return result
    try:
        nib = importlib.import_module('nibabel')
    except ImportError:
        result.errors.append('Dependency blocked: nibabel Python package unavailable.')
        return result
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter('always')
            image = nib.load(str(path), mmap='r', keep_file_open=False)
            result.shape = tuple(int(d) for d in image.shape)
            result.ndim = len(result.shape)
            result.dtype = str(image.get_data_dtype())
            result.affine = image.affine.tolist()
            result.voxel_sizes = tuple(float(v) for v in image.header.get_zooms())
            result.spatial_unit = image.header.get_xyzt_units()[0]
            result.affine_codes = (int(image.header['qform_code']), int(image.header['sform_code']))
            data = np.asanyarray(image.dataobj)
        if caught:
            result.warnings.append(f'NIfTI reader emitted {len(caught)} warning(s); text suppressed for privacy.')
        if np.iscomplexobj(data):
            result.errors.append('Complex-valued segmentation data cannot be assessed.')
            return result
        finite = np.isfinite(data)
        values = data[finite]
        result.finite_voxel_count = int(finite.sum())
        result.nan_count = int(np.isnan(data).sum())
        result.inf_count = int(np.isinf(data).sum())
        result.nonzero_voxel_count = int(np.count_nonzero(values))
        result.nonzero_fraction = result.nonzero_voxel_count / data.size if data.size else 0.0
        if values.size:
            result.minimum_finite_value = float(values.min())
            result.maximum_finite_value = float(values.max())
            unique = np.unique(values)
            if len(unique) <= 32:
                result.unique_finite_values = [float(v) for v in unique]
            if len(unique) == 2 and 0 in unique:
                result.mask_value_class = 'binary_like'
            elif len(unique) <= 32 and np.all(unique == np.round(unique)):
                result.mask_value_class = 'integer_label_like'
            else:
                result.mask_value_class = 'continuous_valued'
                result.warnings.append('Suspicious continuous-valued segmentation field; inspect finite range and unique-value evidence before preprocessing.')
        if not data.size or any(d == 0 for d in result.shape):
            result.errors.append('Empty mask dimension.')
        if result.ndim != 3:
            result.errors.append('Unexpected dimensionality: expected a 3D segmentation field.')
        if result.nan_count or result.inf_count:
            result.errors.append('Mask contains NaN/Inf.')
        if data.size and result.finite_voxel_count == data.size:
            if result.nonzero_voxel_count == 0:
                result.errors.append('All-zero mask: no foreground.')
            if result.nonzero_voxel_count == data.size:
                result.warnings.append('All-nonzero mask: no zero background.')
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
            result.errors.append('Nonfinite or singular affine.')
        if not all(np.isfinite(v) and v > 0 for v in result.voxel_sizes):
            result.errors.append('Invalid header voxel sizes.')
        if result.affine_codes == (0, 0):
            result.warnings.append('Neither qform nor sform is coded; affine is a fallback.')
        if result.spatial_unit == 'unknown':
            result.warnings.append('NIfTI spatial units missing; millimeters cannot be assumed.')
    except Exception as exc:
        result.errors.append(f'NIfTI read failed ({type(exc).__name__}); exception text suppressed for privacy.')
    return result


def assess_mri_mask_compatibility(dicom: DICOMSummary, mask: MaskSummary) -> Compatibility:
    """Compare header geometry in RAS millimeters; never transform image data.

    DICOM grid axes here are [column,row,slice]. PixelSpacing is [row,column].
    LPS -> RAS changes signs of world X/Y. Signed axis permutations are allowed;
    matching world-space voxel centers, spacing and sizes establishes compatible.
    Extents are voxel-center bounds, not segmentation correctness evidence.
    Overlapping but different grids are only potentially compatible. Missing
    units, uncoded affine, nonuniform slices or multiple series are indeterminate.
    """
    def answer(status, reason):
        return Compatibility(status, [reason])
    if dicom.series_count != 1:
        return answer('indeterminate', 'Exactly one unambiguous DICOM series is required; none selected.')
    if dicom.errors or dicom.series[0].errors:
        return answer('indeterminate', 'DICOM validation errors prevent a reliable candidate volume.')
    if mask.affine is None or mask.ndim != 3 or any(d == 0 for d in mask.shape):
        return answer('indeterminate', 'Mask does not have a readable nonempty 3D geometry.')
    scale = {'mm': 1., 'meter': 1000., 'micron': .001}.get(mask.spatial_unit)
    if scale is None or mask.affine_codes == (0, 0):
        return answer('indeterminate', 'NIfTI spatial units or coded affine are missing.')
    series = dicom.series[0]
    if any(key in series.missing_metadata for key in GEOMETRY):
        return answer('indeterminate', 'DICOM geometry is missing for one or more instances.')
    try:
        first = series.instances[0]
        rows, columns = int(first['Rows']), int(first['Columns'])
        ps = np.asarray(first['PixelSpacing'], dtype=float)
        iop = np.asarray(first['ImageOrientationPatient'], dtype=float)
        positions = np.array([r['ImagePositionPatient'] for r in series.instances], dtype=float)
        if ps.shape != (2,) or iop.shape != (6,) or positions.shape != (series.instance_count, 3):
            raise ValueError
        if rows <= 0 or columns <= 0 or not np.all(ps > 0) or not all(np.isfinite(a).all() for a in (ps, iop, positions)):
            raise ValueError
        u, v = iop[:3], iop[3:]
        if not np.allclose([np.linalg.norm(u), np.linalg.norm(v), u @ v], [1, 1, 0], atol=1e-4):
            raise ValueError
        normal = np.cross(u, v)
        positions = positions[np.argsort(positions @ normal)]
        if len(positions) < 2:
            return answer('indeterminate', 'Single slice: volumetric slice spacing and extent cannot be established.')
        deltas = np.diff(positions, axis=0)
        step = deltas.mean(axis=0)
        if step @ normal <= 1e-5 or not np.allclose(deltas, step, atol=1e-3, rtol=1e-3):
            return answer('indeterminate', 'Duplicate or nonuniform slice positions; no regular 3D volume assumed.')
        d_affine = np.eye(4)
        d_affine[:3, :3] = np.column_stack((u * ps[1], v * ps[0], step))
        d_affine[:3, 3] = positions[0]
        d_affine = np.diag([-1., -1., 1., 1.]) @ d_affine
        d_shape = np.array([columns, rows, len(positions)])
        n_affine = np.asarray(mask.affine, dtype=float).copy()
        n_affine[:3] *= scale
        if not np.isfinite(n_affine).all() or abs(np.linalg.det(n_affine[:3, :3])) < 1e-12:
            raise ValueError
        n_shape = np.array(mask.shape)
        n_spacings = np.linalg.norm(n_affine[:3, :3], axis=0)
        if not np.allclose(n_spacings, np.array(mask.voxel_sizes[:3]) * scale, atol=1e-3, rtol=1e-3):
            return answer('indeterminate', 'NIfTI header zooms disagree with affine column lengths.')
    except (TypeError, ValueError, IndexError, np.linalg.LinAlgError):
        return answer('indeterminate', 'Geometry values are malformed, nonfinite, nonpositive or singular.')
    def corners(affine, shape):
        indices = np.array(list(product(*[(0, int(d) - 1) for d in shape])))
        return indices @ affine[:3, :3].T + affine[:3, 3]
    dc, nc = corners(d_affine, d_shape), corners(n_affine, n_shape)
    reasons = [f'DICOM [columns,rows,slices]={tuple(map(int, d_shape))}; NIfTI shape={mask.shape}.',
               f'DICOM axis spacing (mm)={np.linalg.norm(d_affine[:3,:3], axis=0).tolist()}; NIfTI affine spacing (mm)={n_spacings.tolist()}.',
               f'DICOM voxel-center RAS bounds (mm)={dc.min(0).tolist()} to {dc.max(0).tolist()}; NIfTI={nc.min(0).tolist()} to {nc.max(0).tolist()}.']
    transform = np.linalg.solve(d_affine, n_affine)
    for order in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            expected = np.zeros((3, 3))
            origin = np.zeros(3)
            for j, axis in enumerate(order):
                expected[axis, j] = signs[j]
                if signs[j] == -1:
                    origin[axis] = d_shape[axis] - 1
            if (np.array_equal(n_shape, d_shape[list(order)])
                    and np.allclose(transform[:3, :3], expected, atol=1e-3, rtol=1e-3)
                    and np.allclose(transform[:3, 3], origin, atol=1e-2, rtol=0)):
                return Compatibility('compatible', reasons + ['Same voxel-center grid under a signed axis permutation, including LPS/RAS convention.'])
    if np.any(dc.max(0) < nc.min(0) - 1.) or np.any(nc.max(0) < dc.min(0) - 1.):
        return Compatibility('incompatible', reasons + ['World-space voxel-center bounds are disjoint by more than 1 mm.'])
    return Compatibility('potentially_compatible', reasons + ['World bounds overlap but grid/orientation/spacing differs; manual geometry review required, no resampling performed.'])


def _openslide_reader():
    try:
        module = importlib.import_module('openslide')
    except ModuleNotFoundError as exc:
        if exc.name == 'openslide':
            raise RuntimeError('Dependency blocked: openslide-python package missing.') from None
        if exc.name in ('openslide_bin',):
            raise RuntimeError('Dependency blocked: OpenSlide native library missing; Python package is installed.') from None
        raise RuntimeError('Dependency blocked: OpenSlide Python dependency missing.') from None
    except (ImportError, OSError):
        raise RuntimeError('Dependency blocked: OpenSlide native library unavailable; Python package is installed.') from None
    return module.OpenSlide


def validate_wsi(path: str | Path, slide_opener: Callable | None = None) -> WSISummary:
    """Open exact path read-only and inspect allowlisted metadata; no read_region."""
    path = Path(path)
    result = WSISummary(path.name)
    status = _file_status(path)
    if not status['ok']:
        result.errors.append(status['reason'])
        return result
    if slide_opener is None:
        try:
            slide_opener = _openslide_reader()
        except RuntimeError as exc:
            result.readability = 'blocked_dependency'
            result.errors.append(str(exc))  # Our fixed messages, never native exception text.
            return result
    slide = None
    try:
        slide = slide_opener(str(path))
        result.level_count = int(slide.level_count)
        result.level_dimensions = tuple(tuple(int(v) for v in d) for d in slide.level_dimensions)
        result.level_downsamples = tuple(float(v) for v in slide.level_downsamples)
        result.level0_dimensions = tuple(int(v) for v in slide.dimensions)
        if (result.level_count < 1 or len(result.level_dimensions) != result.level_count
                or len(result.level_downsamples) != result.level_count
                or any(len(d) != 2 or min(d) <= 0 for d in result.level_dimensions)
                or result.level0_dimensions != result.level_dimensions[0]
                or not all(np.isfinite(v) and v > 0 for v in result.level_downsamples)):
            raise ValueError('Invalid pyramid metadata')
        props = slide.properties
        for attribute, key in [('vendor', 'openslide.vendor'), ('objective_power', 'openslide.objective-power'),
                               ('mpp_x', 'openslide.mpp-x'), ('mpp_y', 'openslide.mpp-y')]:
            value = props.get(key)
            setattr(result, attribute, str(value) if value not in (None, '') else None)
            if getattr(result, attribute) is None:
                result.warnings.append(f'Missing {attribute}; no default supplied.')
        result.readability = 'readable'
    except Exception as exc:
        result.errors.append(f'OpenSlide metadata read failed ({type(exc).__name__}); exception text suppressed for privacy.')
    finally:
        if slide is not None:
            try:
                slide.close()
            except Exception as exc:
                result.errors.append(f'OpenSlide close failed ({type(exc).__name__}).')
    return result


def validate_raw_patient(record: RawPatientRecord, slide_opener: Callable | None = None) -> PatientValidation:
    """Technical readiness only, never clinical/segmentation/DCE-phase approval.

    Conservative readiness requires compatible grids, readable metadata and
    usable masks. Continuous/all-nonzero fields require review, not automatic
    correction or a claim that their values are intrinsically invalid.
    """
    filesystem = {name: _file_status(getattr(record, name), name == 'mri_dicom_dir') for name in (
        'mri_dicom_dir', 'tumor_mask_path', 'lymph_node_mask_path', 'wsi_path')}
    dicom = validate_dicom_directory(record.mri_dicom_dir)
    tumor = validate_nifti_mask(record.tumor_mask_path)
    ln = validate_nifti_mask(record.lymph_node_mask_path)
    wsi = validate_wsi(record.wsi_path, slide_opener)
    tumor_compat = assess_mri_mask_compatibility(dicom, tumor)
    ln_compat = assess_mri_mask_compatibility(dicom, ln)
    errors, warnings = [], []
    for name, summary in [('DICOM', dicom), ('tumor mask', tumor), ('LN mask', ln), ('WSI', wsi)]:
        errors.extend(f'{record.patient_id} / {name}: {e}' for e in summary.errors)
        warnings.extend(f'{record.patient_id} / {name}: {w}' for w in summary.warnings)
    for name, compatibility, mask in [('tumor', tumor_compat, tumor), ('LN', ln_compat, ln)]:
        if compatibility.status != 'compatible':
            errors.append(f'{record.patient_id} / {name}-MRI: {compatibility.status}; geometry must be resolved before preprocessing.')
        if mask.mask_value_class == 'continuous_valued' or any('All-nonzero' in w for w in mask.warnings):
            errors.append(f'{record.patient_id} / {name} mask: value semantics need manual review; data not automatically declared invalid.')
    warnings.append(f'{record.patient_id}: Header/metadata checks do not establish full DICOM/WSI pixel integrity, correct DCE phase or correct segmentation.')
    return PatientValidation(record.patient_id, record.label, record.cohort, filesystem, dicom, tumor, ln,
                             tumor_compat, ln_compat, wsi, not errors, errors, warnings)
