"""Strict AUROC best selection, independent of min-delta early stopping."""
import copy
from mcminet.training.training_config import TrainingConfig
from mcminet.training.early_stopping import EarlyStopping
from mcminet.training.scheduler import ValidationMetric
from mcminet.training.scientific_metrics import patient_auroc


def scientific_early_stopping():
    return EarlyStopping.from_config(TrainingConfig())


def complete_epoch(epoch, training, validation, auc, records, optimizer, scheduler,
                   checkpoint_early, engineering_history, scientific_early, rows):
    if epoch != len(rows):
        raise ValueError('Scientific epoch continuity mismatch')
    if auc != patient_auroc(records) or len(records) != validation.patient_count:
        raise ValueError('Validation patient AUROC mismatch')
    before = {g['name']: g['lr'] for g in optimizer.param_groups}
    loss_metric = validation.selection(scheduler.config.scheduler_metric)
    scheduler.step(loss_metric)
    selection_metric = ValidationMetric(scheduler.config.selection_metric, auc)
    checkpoint_early.update(selection_metric, epoch)
    lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
    engineering_history.append(epoch, training, validation, selection_metric, lrs)
    old_best = rows[-1]["best_validation_auroc"] if rows else None
    scientific_early.update(ValidationMetric(scientific_early.metric_name, auc), epoch)
    improved = old_best is None or auc > old_best
    row = dict(epoch=epoch, train=training.to_dict(),
               validation=dict(**validation.to_dict(), AUROC=auc),
               validation_patients=copy.deepcopy(records), learning_rates=lrs,
               scheduler=dict(metric='validation_total_loss', value=loss_metric.value,
                              before=before, current=lrs, reduced=before != lrs),
               best_validation_auroc=auc if improved else old_best,
               best_epoch=epoch if improved else rows[-1]["best_epoch"],
               early_stopping=scientific_early.state_dict(),
               selected_best=improved)
    rows.append(row)
    return improved


def validate_scientific_history(rows, saved_early, engineering_rows):
    if not isinstance(rows, list) or not rows or len(rows) != len(engineering_rows):
        raise ValueError('Incomplete scientific checkpoint history')
    early = scientific_early_stopping()
    previous_lrs = None
    best_auc, best_epoch = None, None
    for epoch, (row, engineering) in enumerate(zip(rows, engineering_rows)):
        if row['epoch'] != epoch or engineering['epoch'] != epoch:
            raise ValueError('Scientific epoch continuity mismatch')
        if early.should_stop:
            raise ValueError('History continued after scientific early stopping')
        val = dict(row['validation'])
        auc = val.pop('AUROC')
        if row['train'] != engineering['training'] or val != engineering['validation']:
            raise ValueError('Scientific losses differ from validated checkpoint history')
        if auc != patient_auroc(row['validation_patients']) or len(row['validation_patients']) != val['patient_count']:
            raise ValueError('Scientific AUROC/coverage mismatch')
        improved = best_auc is None or auc > best_auc
        if improved:
            best_auc, best_epoch = auc, epoch
        early.update(ValidationMetric(early.metric_name, auc), epoch)
        if (row['early_stopping'] != early.state_dict() or row['best_epoch'] != best_epoch or
            row['best_validation_auroc'] != best_auc or row['selected_best'] != improved):
            raise ValueError('Scientific selection history mismatch')
        if engineering['selection'] != dict(name=early.metric_name, value=auc, source='validation'):
            raise ValueError('Engineering/scientific selection mismatch')
        schedule = row['scheduler']
        if (row['learning_rates'] != engineering['learning_rates'] or schedule['current'] != row['learning_rates'] or
            schedule['metric'] != 'validation_total_loss' or schedule['value'] != val['losses']['total'] or
            schedule['reduced'] != (schedule['before'] != schedule['current']) or
            (previous_lrs is not None and schedule['before'] != previous_lrs)):
            raise ValueError('Scientific scheduler history mismatch')
        previous_lrs = row['learning_rates']
    if saved_early != early.state_dict():
        raise ValueError('Scientific early-stop state mismatch')
    return early
