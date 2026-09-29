"""Explicit raw paths and research IDs only; no cohort or label inference."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import csv
from itertools import combinations
from pathlib import Path


@dataclass(frozen=True)
class RawPatientRecord:
    patient_id: str
    label: int
    cohort: str
    mri_dicom_dir: Path
    tumor_mask_path: Path
    lymph_node_mask_path: Path
    wsi_path: Path


def load_patient_manifest(path: str | Path) -> tuple[RawPatientRecord, ...]:
    """Read ordered records; resolve exact paths relative to the CSV directory.

    Empty manifests are rejected. Extra columns are ignored. Raw path existence
    is checked separately. Labels accept only stripped CSV strings '0'/'1'.
    No MP mapping, file renaming, patient identity inference or cohort splitting.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError('Manifest must exist and be a regular file.')
    required = tuple(RawPatientRecord.__dataclass_fields__)
    records, seen = [], set()
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle, strict=True)
            fields = reader.fieldnames or []
            if len(fields) != len(set(fields)) or not set(required) <= set(fields):
                raise ValueError('Manifest has missing required or duplicate columns.')
            for number, row in enumerate(reader, start=2):
                values = {key: (row.get(key) or '').strip() for key in required}
                if any(not value for value in values.values()):
                    raise ValueError(f'Manifest row {number}: required fields must be nonempty.')
                if values['label'] not in ('0', '1'):
                    raise ValueError(f'Manifest row {number}: label must be canonical 0 or 1.')
                if values['patient_id'] in seen:
                    raise ValueError(f'Manifest row {number}: duplicate patient_id.')
                seen.add(values['patient_id'])
                paths = {key: (path.parent / values[key]).resolve() for key in required[3:]}
                records.append(RawPatientRecord(values['patient_id'], int(values['label']), values['cohort'], **paths))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f'Manifest cannot be read as CSV ({type(exc).__name__}).') from None
    if not records:
        raise ValueError('Manifest must contain at least one patient row.')
    return tuple(records)


def find_patient_id_overlaps(
    cohorts: Mapping[str, Sequence[RawPatientRecord]],
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Return every collection pair and its sorted explicit-ID intersection.

    Empty intersections are included. Names identify supplied collections only;
    this neither splits cohorts nor matches identities using medical metadata.
    """
    ids = {name: {r.patient_id for r in records} for name, records in cohorts.items()}
    return {(a, b): tuple(sorted(ids[a] & ids[b])) for a, b in combinations(ids, 2)}
