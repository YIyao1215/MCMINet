"""Hash-bound scientific sidecar with strict AUROC selection.

Best checkpoint selection and min-delta early stopping have separate state.
An atomic last.json commits both LAST and BEST references together. best.json is
a convenience view; resume always uses last.json as the authoritative commit.
"""
import json
from pathlib import Path
import uuid
import torch
from mcminet.data.wsi_feature_cache import sha256_file
from mcminet.training.training_history import atomic_write
from mcminet.training.native_mri_checkpoint import save_native_checkpoint, load_native_checkpoint
from mcminet.training.scientific_run_history import validate_scientific_history


def write_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False)
    atomic_write(path, lambda p: Path(p).write_text(text, encoding='utf-8'))


def _read(root, reference):
    if set(reference) != {'bundle', 'scientific_sha256'}:
        raise ValueError('Invalid checkpoint reference')
    name = reference['bundle']
    if not isinstance(name, str) or Path(name).name != name or not name.startswith('epoch_'):
        raise ValueError('Invalid checkpoint bundle path')
    folder = root / name
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Missing/unsafe checkpoint bundle')
    sidecar = folder / 'scientific.json'
    if sha256_file(sidecar) != reference['scientific_sha256']:
        raise ValueError('Scientific metadata hash mismatch')
    meta = json.loads(sidecar.read_text())
    if sha256_file(folder / 'state.pt') != meta['checkpoint_sha256']:
        raise ValueError('Tensor checkpoint hash mismatch')
    return folder, meta


def save_bundle(root, *, epoch, metadata, rows, scientific_early, previous_best, checkpoint_args):
    root = Path(root)
    folder = root / f'epoch_{epoch:04d}_{uuid.uuid4().hex}'
    folder.mkdir(parents=True, exist_ok=False)
    save_native_checkpoint(folder / 'state.pt', epoch=epoch, **checkpoint_args)
    meta = dict(schema_version=2, run=metadata, epoch=epoch,
                history=rows, scientific_early_stopping=scientific_early.state_dict(),
                best_epoch=rows[-1]["best_epoch"], best_validation_auroc=rows[-1]["best_validation_auroc"],
                checkpoint_sha256=sha256_file(folder / 'state.pt'),
                selection_semantics='strict_auroc_best_separate_from_min_delta_stopper')
    write_json(folder / 'scientific.json', meta)
    ref = dict(bundle=folder.name, scientific_sha256=sha256_file(folder / 'scientific.json'))
    best = ref if rows[-1]["best_epoch"] == epoch else previous_best
    if best is None:
        raise ValueError('Missing scientific best checkpoint')
    # Commit point: both references advance together, after all payloads exist.
    write_json(root / 'last.json', dict(last=ref, best=best))
    write_json(root / 'best.json', dict(best=best))
    return best


def load_bundle(path, *, metadata, checkpoint_args):
    path = Path(path)
    if path.name != 'last.json':
        raise ValueError('Resume must use LAST epoch-boundary manifest last.json')
    pointers = json.loads(path.read_text())
    if set(pointers) != {'last', 'best'}:
        raise ValueError('Invalid last checkpoint manifest')
    folder, meta = _read(path.parent, pointers['last'])
    _, best_meta = _read(path.parent, pointers['best'])
    for candidate in (meta, best_meta):
        if candidate['schema_version'] != 2 or candidate['run'] != metadata:
            raise ValueError('Incompatible candidate checkpoint schema/protocol/role/seed/cohort; legacy bundles cannot resume')
    # Validate all scientific state BEFORE invoking the transactional validated loader.
    payload = torch.load(folder / 'state.pt', map_location='cpu', weights_only=True)
    early = validate_scientific_history(meta['history'], meta['scientific_early_stopping'], payload['history'])
    best_epoch = meta['history'][-1]['best_epoch']
    best_auc = meta['history'][-1]['best_validation_auroc']
    if (meta['epoch'] != payload['epoch'] or meta['epoch'] != len(meta['history']) - 1 or
        meta['best_epoch'] != best_epoch or meta['best_validation_auroc'] != best_auc or
        best_meta['epoch'] != best_epoch or best_meta['best_validation_auroc'] != best_auc or
        best_meta['history'] != meta['history'][:best_epoch + 1]):
        raise ValueError('Best/last or epoch continuity mismatch')
    expected_patients = dict(metadata['validation_roster'])
    for row in meta['history']:
        if {r['patient_id']: r['label'] for r in row['validation_patients']} != expected_patients:
            raise ValueError('Resumed validation cohort coverage mismatch')
    del payload
    info = load_native_checkpoint(folder / 'state.pt', **checkpoint_args)
    return info['next_epoch'], meta['history'], early, pointers['best']
