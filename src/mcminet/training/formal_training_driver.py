"""Multi-epoch scientific orchestration; no objective/optimizer/epoch math copies."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import torch
from mcminet.training.scientific_protocol import ScientificTrainingProtocol, engineering_config, validate_config, validate_role
from mcminet.training.scientific_cohorts import CohortLoader, validate_cohorts
from mcminet.training.scientific_metrics import evaluate_with_scores
from mcminet.training.scientific_run_history import complete_epoch, scientific_early_stopping
from mcminet.training.scientific_checkpoint_bundle import save_bundle, load_bundle, write_json
from mcminet.training.native_epoch_runner import train_native_one_epoch
from mcminet.training.training_config import build_objective
from mcminet.training.training_policy import apply_training_policy
from mcminet.training.optimizer_builder import build_optimizer, audit_optimizer, initial_group_lr
from mcminet.training.scheduler import ValidationScheduler
from mcminet.training.early_stopping import EarlyStopping
from mcminet.training.training_history import TrainingHistory
from mcminet.training.reproducibility import seed_everything, seeded_generator
from mcminet.training.mri_batchnorm_policy import audit_mri_batchnorm_policy
from mcminet.models.variable_mri_batching import VariableMRICachedMCMINet, policy_fingerprint
from mcminet.utils.state_fingerprint import state_fingerprint


@dataclass(frozen=True)
class ModelInputs:
    canonical_model: torch.nn.Module
    cache_identity: dict


@dataclass
class RunResult:
    history: list
    model: torch.nn.Module
    objective: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: ValidationScheduler
    scientific_early_stopping: EarlyStopping
    metadata: dict
    stopped_early: bool


def run_training(*, model_factory: Callable, train: CohortLoader, validation: CohortLoader,
                 output_directory, role='PRIMARY', seed=None, device='cpu',
                 protocol=None, config=None, resume_checkpoint=None,
                 generator=None):
    """Normal entrypoint: max_epochs=100 is locked; no execution-cap override.

    Factory executes AFTER seed initialization and returns canonical model plus
    trusted cache identity. Objective/proxies, policy and optimizer use validated builders.
    Callers explicitly provide the roster and roles; external testing is absent.
    """
    return _run(model_factory=model_factory, train=train, validation=validation,
                output_directory=output_directory, role=role, seed=seed, device=device,
                protocol=protocol, config=config, resume_checkpoint=resume_checkpoint,
                generator=generator, dry_run_cap=None)


def _run(*, model_factory, train, validation, output_directory, role, seed, device,
         protocol, config, resume_checkpoint, generator, dry_run_cap):
    protocol = ScientificTrainingProtocol() if protocol is None else protocol
    seed = protocol.primary_seed if seed is None else seed
    protocol.validate()
    validate_role(role, seed)
    config = engineering_config(seed) if config is None else config
    validate_config(config, seed)
    validate_cohorts(train, validation)
    if dry_run_cap is not None:
        if not train.synthetic or not validation.synthetic:
            raise ValueError('Test-only execution cap requires explicitly synthetic cohorts')
        if type(dry_run_cap) is not int or not 1 <= dry_run_cap <= 5:
            raise ValueError('Dry-run execution cap must be 1..5; protocol stays at 100')
    root = Path(output_directory)
    if resume_checkpoint is None and root.exists() and any(root.iterdir()):
        raise ValueError('New run requires an empty output directory')
    if resume_checkpoint is not None and Path(resume_checkpoint).resolve() != (root / 'last.json').resolve():
        raise ValueError('Resume must use this run directory LAST checkpoint')
    if generator is None:
        generator = seeded_generator(seed)
    if generator.device.type != 'cpu' or generator.initial_seed() != seed:
        raise ValueError('Checkpointed data generator must use the declared run seed on CPU')
    for cohort in (train, validation):
        for owner in (cohort.loader, getattr(cohort.loader, 'sampler', None)):
            owned = getattr(owner, 'generator', None)
            if owned is not None and owned is not generator:
                raise ValueError('Loader/sampler generator must be the checkpointed generator')
        if getattr(cohort.loader, 'persistent_workers', False):
            raise ValueError('Persistent worker RNG cannot be resumed at this epoch boundary')
    seed_everything(seed, deterministic_algorithms=config.deterministic_algorithms)
    inputs = model_factory()
    canonical = inputs.canonical_model
    from mcminet.training.model_contract import validate_model_contract
    validate_model_contract(canonical, protocol.config, synthetic=train.synthetic and validation.synthetic)
    objective = build_objective(config).to(device)
    apply_training_policy(canonical, objective)
    model = VariableMRICachedMCMINet(canonical).to(device)
    offline = canonical.wsi_branch.patch_encoder
    if state_fingerprint(offline) != inputs.cache_identity['encoder_state_sha256']:
        raise ValueError('Factory offline encoder/cache identity mismatch')
    from mcminet.training.checkpointing import training_identity
    training_identity(inputs.cache_identity, model, config)
    optimizer = build_optimizer(model, objective, offline, config)
    audit_optimizer(optimizer, model, objective, offline, config)
    if any(g['lr'] != initial_group_lr(config, g) for g in optimizer.param_groups):
        raise ValueError('Initial optimizer group LR differs from locked protocol')
    scheduler = ValidationScheduler(optimizer, config)
    checkpoint_early = EarlyStopping.from_config(config)
    history = TrainingHistory()
    early = scientific_early_stopping()
    metadata = dict(protocol=protocol.to_dict(), protocol_fingerprint=protocol.fingerprint(),
                    role=role, seed=seed, primary_seed=protocol.primary_seed, native_policy_sha256=policy_fingerprint(),
                    training_roster=[list(r) for r in train.patient_labels],
                    validation_roster=[list(r) for r in validation.patient_labels],
                    data_kind='synthetic' if train.synthetic and validation.synthetic else 'real',
                    execution_mode='synthetic_dry_run' if dry_run_cap is not None else 'scientific',
                    external_test_access=False, score_convention='raw_logit')
    args = dict(model=model, objective=objective, optimizer=optimizer, scheduler=scheduler,
                early_stopping=checkpoint_early, history=history, config=config,
                offline_encoder=offline, cache_identity=inputs.cache_identity,
                device=device, generator=generator)
    rows, start, best = [], 0, None
    if resume_checkpoint is not None:
        start, rows, early, best = load_bundle(resume_checkpoint, metadata=metadata, checkpoint_args=args)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / 'run_metadata.json', metadata)
    end = protocol.max_epochs if dry_run_cap is None else min(protocol.max_epochs, dry_run_cap)
    for epoch in range(start, end):
        if early.should_stop:
            break
        training = train_native_one_epoch(model, objective, train.batches(), optimizer, offline, config, device=device)
        val, auc, records = evaluate_with_scores(model, objective, validation.batches(), offline, config, device=device)
        complete_epoch(epoch, training, val, auc, records, optimizer, scheduler, checkpoint_early, history, early, rows)
        audit_mri_batchnorm_policy(model.mri_branch)
        best = save_bundle(root, epoch=epoch, metadata=metadata, rows=rows,
                           scientific_early=early, previous_best=best, checkpoint_args=args)
        write_json(root / 'scientific_history.json', rows)
    # Repair convenience artifacts from authoritative LAST state after interruption.
    if rows:
        write_json(root / 'best.json', dict(best=best))
        write_json(root / 'scientific_history.json', rows)
    return RunResult(rows, model, objective, optimizer, scheduler, early, metadata, early.should_stop)
