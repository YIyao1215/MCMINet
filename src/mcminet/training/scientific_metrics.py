"""One patient-level AUROC per complete validation epoch, using raw logits."""
import math
from mcminet.training.native_epoch_runner import evaluate_native_one_epoch


def patient_auroc(records):
    if not records:
        raise ValueError('Empty validation epoch')
    seen = set()
    for row in records:
        if set(row) != {'patient_id', 'label', 'raw_logit'}:
            raise ValueError('Invalid patient score schema')
        pid, label, score = row['patient_id'], row['label'], row['raw_logit']
        if not isinstance(pid, str) or not pid or pid in seen:
            raise ValueError('Missing/duplicate validation patient ID')
        seen.add(pid)
        if type(label) is not int or label not in (0, 1):
            raise ValueError('Binary validation labels required')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Finite raw logits required')
    positives = sum(r['label'] for r in records)
    negatives = len(records) - positives
    if not positives or not negatives:
        raise ValueError('Validation AUROC requires both classes')
    # Mann-Whitney U with midranks: tied positive/negative pairs count as 0.5.
    # No sklearn dependency was present in the accepted project source.
    ordered = sorted(records, key=lambda r: r['raw_logit'])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j]['raw_logit'] == ordered[i]['raw_logit']:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(r['label'] for r in ordered[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def evaluate_with_scores(model, objective, batches, offline_encoder, config, *, device='cpu'):
    """Observe the existing evaluator's single forward; never repeat its mechanics."""
    records = []
    pending = []

    def observed_batches():
        for batch in batches:
            if pending:
                raise ValueError('Previous validation forward was not observed')
            pending.append(batch)
            yield batch

    def collect(module, args, output):
        if len(pending) != 1:
            raise ValueError('Unexpected validation model forward')
        batch = pending.pop()
        logits = output['logits'].detach().cpu()
        if logits.ndim != 1 or len(logits) != len(batch['patient_ids']):
            raise ValueError('Patient/logit alignment mismatch')
        for pid, label, score in zip(batch['patient_ids'], batch['labels'].tolist(), logits.tolist()):
            records.append(dict(patient_id=pid, label=label, raw_logit=score))

    handle = model.register_forward_hook(collect)
    try:
        result = evaluate_native_one_epoch(model, objective, observed_batches(),
                                          offline_encoder, config, device=device)
    finally:
        handle.remove()
    if pending or len(records) != result.patient_count:
        raise ValueError('Incomplete validation score collection')
    return result, patient_auroc(records), records


def loss_scale_diagnostic(loss_rows):
    """Engineering flag only: positive-component max/min > 1000; never tune."""
    from mcminet.training.epoch_runner import LOSS_NAMES
    rows = []
    for losses in loss_rows:
        if set(losses) != set(LOSS_NAMES):
            raise ValueError('Six loss components required')
        flags = []
        for name, value in losses.items():
            if not math.isfinite(value):
                flags.append('nonfinite:' + name)
            elif value == 0:
                flags.append(('zero_check_context:' if name == 'intra_modal' else 'unexpected_zero:') + name)
        positive = [v for k, v in losses.items() if k != 'total' and math.isfinite(v) and v > 0]
        ratio = max(positive) / min(positive) if positive else None
        if ratio is not None and ratio > 1000:
            flags.append('extreme_positive_component_ratio')
        rows.append(dict(losses=losses, positive_component_ratio=ratio, flags=flags))
    return dict(purpose='synthetic engineering diagnostic only', threshold_ratio=1000,
                same_class_intra_zero_expected=True, tuning_performed=False, rows=rows)
