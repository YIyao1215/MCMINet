"""release: immutable QuPath geometry and read-only diagnostics.

Observed export schema: FeatureCollection / Feature /
properties.classification.name (case-sensitive). Polygon and MultiPolygon only.
Coordinates remain original XY level-0 pixels. Rings must already be closed;
no repair, clipping, snapping, simplification, precedence or patch decisions.
Requires Shapely >=2 for immutable geometry objects. Source IDs stay internal.
"""
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from types import MappingProxyType
import csv
import json
import math

from PIL import Image, ImageDraw
import shapely
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import explain_validity

if int(shapely.__version__.split('.')[0]) < 2:
    raise ImportError('release requires immutable Shapely 2 geometry objects')

SUPPORTED_CLASSES = ('Tumor', 'Stroma')


@dataclass(frozen=True)
class QuPathAnnotation:
    annotation_index: int
    annotation_id: str | None
    classification: str | None
    geometry_type: str | None
    geometry: BaseGeometry | None
    coordinates: tuple
    bounds: tuple[float, float, float, float] | None
    area_pixels2: float | None
    is_valid: bool
    bounds_status: str
    issues: tuple[str, ...]


@dataclass(frozen=True)
class QuPathAnnotationSet:
    patient_id: str
    slide_path: Path
    geojson_path: Path
    slide_dimensions: tuple[int, int]
    annotations: tuple[QuPathAnnotation, ...]
    class_counts: Mapping[str, int]
    warnings: tuple[str, ...]
    objective_power: float | None
    mpp_x: float | None
    mpp_y: float | None


def manifest_slide_paths(path, patient_ids=('P001', 'P002')):
    """Read only research ID and WSI path values, ignoring clinical columns."""
    path = Path(path).resolve()
    found = {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)) or not {'patient_id', 'wsi_path'} <= set(fields):
            raise ValueError('Manifest requires unique patient_id and wsi_path columns')
        for row in reader:
            pid = (row.get('patient_id') or '').strip()
            if pid in patient_ids:
                value = (row.get('wsi_path') or '').strip()
                if pid in found or not value:
                    raise ValueError('Duplicate research ID or missing WSI path')
                found[pid] = (path.parent / value).resolve()
    if set(found) != set(patient_ids):
        raise ValueError('Requested research IDs are missing from manifest')
    return tuple((pid, found[pid]) for pid in patient_ids)


def paired_geojson(slide_path):
    slide_path = Path(slide_path).resolve()
    if slide_path.suffix.lower() != '.ndpi' or not slide_path.is_file():
        raise ValueError('Manifest WSI must be an existing NDPI file')
    expected = slide_path.with_suffix('.geojson')
    if not expected.is_file():
        raise ValueError('Missing same-directory basename-matched GeoJSON; no fallback file selected')
    return expected


def classification_name(properties):
    if not isinstance(properties, dict):
        return None
    classification = properties.get('classification')
    if not isinstance(classification, dict):
        return None
    name = classification.get('name')
    return name if isinstance(name, str) and name else None


def _dimensions(dimensions):
    if (len(dimensions) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                                    for v in dimensions)):
        raise ValueError('Slide dimensions must be two positive integers')
    return tuple(dimensions)


def _polygon_coordinates(value):
    """Validate before Shapely can implicitly close a ring or ignore a Z axis."""
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError('Empty or malformed polygon coordinates')
    rings = []
    for ring in value:
        if not isinstance(ring, (list, tuple)) or len(ring) < 4:
            raise ValueError('Polygon rings require at least four positions')
        points = []
        for point in ring:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError('Coordinates must be exactly two-dimensional XY')
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in point):
                raise ValueError('Coordinates must be numeric')
            try:
                finite = all(math.isfinite(v) for v in point)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError('Coordinates must be finite; NaN/Inf or overflow rejected')
            points.append(tuple(point))
        if points[0] != points[-1]:
            raise ValueError('Unclosed GeoJSON ring; automatic closure prohibited')
        rings.append(tuple(points))
    return tuple(rings)


def bounds_status(geometry, dimensions):
    """Exact domain [0,W]x[0,H]. Exterior-only boundary contact is outside.

    A valid ROI inside the domain is 'touches boundary' if any boundary meets
    the slide boundary. An exterior ROI with zero interior intersection area
    is 'completely out of bounds', even if its boundary touches the slide.
    """
    domain = box(0, 0, *_dimensions(dimensions))
    if domain.covers(geometry):
        return 'touches boundary' if geometry.intersects(domain.boundary) else 'fully in bounds'
    return ('partially out of bounds' if geometry.intersection(domain).area > 0
            else 'completely out of bounds')


def parse_feature_collection(data, dimensions, supported_classes=SUPPORTED_CLASSES):
    """Preserve source feature order, including rejected-feature diagnostics.

    Structurally rejected geometries have geometry=None. Constructible but
    invalid polygons are retained unmodified and excluded from topology work.
    Missing/unexpected classifications are retained and flagged, never mapped.
    """
    dimensions = _dimensions(dimensions)
    if not isinstance(data, dict) or data.get('type') != 'FeatureCollection':
        raise ValueError('Expected GeoJSON FeatureCollection')
    if data.get('crs') is not None:
        raise ValueError('Explicit CRS requires review; adapter accepts level-0 XY pixels only')
    features = data.get('features')
    if not isinstance(features, list):
        raise ValueError('FeatureCollection requires a features array')
    annotations = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or feature.get('type') != 'Feature':
            raise ValueError(f'Annotation {index}: expected Feature')
        classification = classification_name(feature.get('properties'))
        issues = []
        if classification is None:
            issues.append('Missing/null/malformed classification')
        elif classification not in supported_classes:
            issues.append('Unsupported/unexpected classification (preserved without mapping)')
        raw = feature.get('geometry')
        geometry_type = raw.get('type') if isinstance(raw, dict) else None
        if not isinstance(geometry_type, str):
            geometry_type = None
        geometry, coordinates, bounds, area = None, (), None, None
        valid, status = False, 'not evaluated'
        try:
            if not isinstance(raw, dict):
                raise ValueError('Null or malformed geometry rejected')
            if geometry_type == 'Polygon':
                coordinates = _polygon_coordinates(raw.get('coordinates'))
                geometry = Polygon(coordinates[0], coordinates[1:])
            elif geometry_type == 'MultiPolygon':
                values = raw.get('coordinates')
                if not isinstance(values, (list, tuple)) or not values:
                    raise ValueError('Empty or malformed MultiPolygon rejected')
                coordinates = tuple(_polygon_coordinates(v) for v in values)
                geometry = MultiPolygon([Polygon(v[0], v[1:]) for v in coordinates])
            else:
                raise ValueError('Unsupported ROI geometry type; only Polygon/MultiPolygon accepted')
            bounds = tuple(float(v) for v in geometry.bounds)
            area = float(geometry.area)
            if geometry.is_empty:
                issues.append('Empty geometry rejected')
            if len(bounds) != 4 or not all(math.isfinite(v) for v in bounds):
                issues.append('Nonfinite/invalid geometry bounds')
                bounds = None
            if not math.isfinite(area) or area <= 0:
                issues.append('Geometry must have finite positive area')
                if not math.isfinite(area):
                    area = None
            if not geometry.is_valid:
                issues.append('Invalid geometry: ' + explain_validity(geometry))
            valid = bool(not geometry.is_empty and geometry.is_valid and bounds is not None
                         and area is not None and area > 0)
            if valid:
                status = bounds_status(geometry, dimensions)
                if status in ('partially out of bounds', 'completely out of bounds'):
                    issues.append('Geometry is ' + status + '; no clipping performed')
        except ValueError as exc:
            issues.append(str(exc))
        annotation_id = feature.get('id')
        annotation_id = str(annotation_id) if isinstance(annotation_id, (str, int)) else None
        annotations.append(QuPathAnnotation(index, annotation_id, classification, geometry_type,
            geometry, coordinates, bounds, area, valid, status, tuple(issues)))
    return tuple(annotations)


def _optional_positive(properties, key):
    try:
        value = float(properties[key])
        return value if math.isfinite(value) and value > 0 else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def load_annotations(patient_id, slide_path, slide, supported_classes=SUPPORTED_CLASSES):
    path = paired_geojson(slide_path)
    # No geometry schema errors or source paths are printed by this loader.
    with path.open(encoding='utf-8-sig') as handle:
        data = json.load(handle)
    dimensions = _dimensions(tuple(slide.dimensions))
    annotations = parse_feature_collection(data, dimensions, supported_classes)
    counts = Counter(a.classification for a in annotations if a.classification is not None)
    warnings = [f'Annotation {a.annotation_index}: {issue}' for a in annotations for issue in a.issues]
    for name in SUPPORTED_CLASSES:
        if counts[name] == 0:
            warnings.append(f'Expected class {name} is absent')
    properties = slide.properties
    mx, my = (_optional_positive(properties, 'openslide.mpp-' + axis) for axis in ('x', 'y'))
    if mx is None or my is None:
        warnings.append('MPP incomplete/unavailable; physical areas not computed')
    return QuPathAnnotationSet(patient_id, Path(slide_path).resolve(), path, dimensions,
        annotations, MappingProxyType(dict(sorted(counts.items()))), tuple(warnings),
        _optional_positive(properties, 'openslide.objective-power'), mx, my)


def area_units(area, mpp_x=None, mpp_y=None):
    valid_mpp = all(v is not None and math.isfinite(v) and v > 0 for v in (mpp_x, mpp_y))
    um2 = area*mpp_x*mpp_y if valid_mpp else None
    return dict(pixels2=area, um2=um2, mm2=um2/1e6 if um2 is not None else None)


def _components(annotation):
    if annotation.geometry is None:
        return ()
    return ((annotation.geometry,) if annotation.geometry_type == 'Polygon'
            else tuple(annotation.geometry.geoms))


def spatial_relationships(annotation_set):
    """Diagnostic unions only; no replacement of canonical individual ROIs.

    Same-class overlap means positive intersection area, including containment
    and duplicates; boundary-only touching is separate. Containment uses covers
    (boundary-inclusive), and equality is reported separately. Polygon-component
    IDs are [source annotation index, component index]. No precision grid used.
    Any invalid/unparsed ROI makes topology unavailable, rather than measuring
    a silently reduced subset. Missing classes have area zero, fraction None.
    """
    annotations = annotation_set.annotations
    if any(not a.is_valid for a in annotations):
        return dict(available=False, reason='Invalid/rejected geometry; no partial-subset union computed')
    groups = {name: [(a.annotation_index, i, geom) for a in annotations if a.classification == name
                     for i, geom in enumerate(_components(a))] for name in SUPPORTED_CLASSES}
    unions = {name: unary_union([g for _, _, g in group]) for name, group in groups.items()}
    result = dict(available=True, classes={})
    for name, group in groups.items():
        overlaps, touches = [], []
        for a, b in combinations(group, 2):
            intersection = a[2].intersection(b[2])
            if intersection.area > 0:
                overlaps.append(dict(a=list(a[:2]), b=list(b[:2]), area_pixels2=float(intersection.area)))
            elif a[2].intersects(b[2]):
                touches.append(dict(a=list(a[:2]), b=list(b[:2])))
        result['classes'][name] = dict(
            annotation_count=sum(a.classification == name for a in annotations),
            polygon_component_count=len(group),
            summed_area=area_units(sum(g.area for _, _, g in group), annotation_set.mpp_x, annotation_set.mpp_y),
            union_area=area_units(float(unions[name].area), annotation_set.mpp_x, annotation_set.mpp_y),
            within_class_overlap=bool(overlaps), overlap_pairs=overlaps, boundary_touch_pairs=touches)
    overlap = float(unions['Tumor'].intersection(unions['Stroma']).area)
    result['tumor_stroma_intersection'] = area_units(overlap, annotation_set.mpp_x, annotation_set.mpp_y)
    for name in SUPPORTED_CLASSES:
        result[name.lower() + '_overlap_fraction'] = overlap/unions[name].area if unions[name].area else None
    containment = dict(tumor_covered_by_stroma=[], stroma_covered_by_tumor=[], equal_polygon_pairs=[])
    for t, s in product(groups['Tumor'], groups['Stroma']):
        pair = dict(tumor=list(t[:2]), stroma=list(s[:2]))
        if s[2].covers(t[2]): containment['tumor_covered_by_stroma'].append(pair)
        if t[2].covers(s[2]): containment['stroma_covered_by_tumor'].append(pair)
        if t[2].equals(s[2]): containment['equal_polygon_pairs'].append(pair)
    result['containment'] = containment
    return result


def diagnostic_report(annotation_set):
    annotations = annotation_set.annotations
    relationships = spatial_relationships(annotation_set)
    rejected = sum(not a.is_valid for a in annotations)
    out = sum(a.bounds_status in ('partially out of bounds', 'completely out of bounds') for a in annotations)
    unexpected = sum(a.classification not in SUPPORTED_CLASSES for a in annotations)
    ready = bool(annotations and not (rejected or out or unexpected)
                 and all(annotation_set.class_counts.get(c, 0) for c in SUPPORTED_CLASSES))
    return dict(patient_id=annotation_set.patient_id, basename_match=True,
        slide_resolved_from_manifest=True, level0_dimensions=list(annotation_set.slide_dimensions),
        objective_power=annotation_set.objective_power, mpp_x=annotation_set.mpp_x, mpp_y=annotation_set.mpp_y,
        total_annotation_count=len(annotations), class_counts=dict(annotation_set.class_counts),
        polygon_component_counts={c:sum(len(_components(a)) for a in annotations if a.classification == c)
                                  for c in SUPPORTED_CLASSES},
        unsupported_or_missing_class_count=unexpected,
        geometry_types=dict(Counter(a.geometry_type or 'null' for a in annotations)),
        invalid_or_rejected_geometry_count=rejected, out_of_bounds_count=out,
        bounds_not_evaluated_count=sum(a.bounds_status == 'not evaluated' for a in annotations),
        bounds_status_counts=dict(Counter(a.bounds_status for a in annotations)),
        numerical_level0_compatibility=bool(annotations and not rejected and not out),
        readiness='READY' if ready else 'NOT READY',
        readiness_scope='Geometry diagnostics ready for manual review of later patch-overlap policy; visual correctness unconfirmed',
        warnings=list(annotation_set.warnings), relationships=relationships,
        annotations=[dict(annotation_index=a.annotation_index, classification=a.classification,
            geometry_type=a.geometry_type, bounds=a.bounds, is_valid=a.is_valid,
            bounds_status=a.bounds_status, issues=list(a.issues),
            area=area_units(a.area_pixels2, annotation_set.mpp_x, annotation_set.mpp_y)
                 if a.area_pixels2 is not None else None) for a in annotations])


def overlay_point(point, level0_dimensions, overview_dimensions):
    """Display-only affine XY: (x*Wt/W0, y*Ht/H0); no flip or axis swap."""
    w0, h0 = _dimensions(level0_dimensions)
    wt, ht = _dimensions(overview_dimensions)
    return point[0]*wt/w0, point[1]*ht/h0


def bounded_overview(slide, max_side=1600, max_level_pixels=4_000_000):
    """Read one bounded pyramid level, never an unbounded full level-0 slide.

    The returned image represents the whole level-0 extent. Exact per-axis
    ratios use its actual post-thumbnail dimensions, not rounded downsample.
    """
    levels = [(i, tuple(d)) for i, d in enumerate(slide.level_dimensions)
              if d[0]*d[1] <= max_level_pixels]
    if not levels:
        raise ValueError('No bounded overview pyramid level available')
    level, dimensions = max(levels, key=lambda item: item[1][0]*item[1][1])
    image = slide.read_region((0, 0), level, dimensions)
    if image.size != dimensions:
        raise ValueError('Overview reader returned incorrect dimensions')
    if image.mode == 'RGBA':
        image = Image.alpha_composite(Image.new('RGBA', image.size, 'white'), image).convert('RGB')
    else:
        image = image.convert('RGB')
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return image


def render_overlay(overview, annotation_set, classes=SUPPORTED_CLASSES):
    """Outline exterior AND interior rings; preserve holes, source, and image.

    A header is added only after geometry is rendered in overview coordinates.
    Unknown classes are amber on the all-annotation view when explicitly passed.
    Invalid constructible outlines are shown for review, never repaired.
    """
    image = overview.convert('RGB').copy()
    draw = ImageDraw.Draw(image)
    colors = {'Tumor': '#ff2020', 'Stroma': '#00a020'}
    for annotation in annotation_set.annotations:
        if annotation.classification not in classes:
            continue
        for polygon in _components(annotation):
            for ring in (polygon.exterior, *polygon.interiors):
                points = [overlay_point(p, annotation_set.slide_dimensions, image.size) for p in ring.coords]
                draw.line(points, fill=colors.get(annotation.classification, '#e0a000'), width=2)
    canvas = Image.new('RGB', (max(image.width, 540), image.height + 40), 'white')
    canvas.paste(image, (0, 40))
    ImageDraw.Draw(canvas).text((8, 8),
        annotation_set.patient_id + ' | Tumor: red | Stroma: green | manual review required', fill='black')
    return canvas
