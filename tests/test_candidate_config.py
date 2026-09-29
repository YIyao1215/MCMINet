"""Synthetic consistency and replay tests; no clinical performance evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from mcminet.config import load_config, validate_config, build_model, default, require_candidate
from mcminet.training.training_config import TrainingConfig, build_objective
from mcminet.training.training_policy import apply_training_policy
from mcminet.training.optimizer_builder import build_optimizer, audit_optimizer, initial_group_lr
from mcminet.models.variable_mri_batching import VariableMRICachedMCMINet
from mcminet.training.scheduler import ValidationScheduler, ValidationMetric
from mcminet.training.early_stopping import EarlyStopping
from mcminet.training.scientific_protocol import ScientificTrainingProtocol, validate_role
from mcminet.training.scientific_cohorts import CohortLoader, validate_cohorts
from mcminet.training.model_contract import validate_model_contract
from mcminet.training.epoch_runner import EpochResult, LOSS_NAMES
from mcminet.training.training_history import TrainingHistory
from mcminet.training.scientific_run_history import complete_epoch, scientific_early_stopping, validate_scientific_history

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def pipeline():
    config = load_config(ROOT / 'configs/default.yaml')
    training = TrainingConfig.from_config(config)
    canonical = build_model(config, smoke_test=True)
    objective = build_objective(training)
    apply_training_policy(canonical, objective)
    model = VariableMRICachedMCMINet(canonical)
    offline = canonical.wsi_branch.patch_encoder
    optimizer = build_optimizer(model, objective, offline, training)
    return config, training, canonical, model, objective, offline, optimizer


def test_yaml_protocol_and_constructor_defaults():
    config = load_config(ROOT / 'configs/default.yaml')
    example = load_config(ROOT / 'configs/example.yaml')
    assert {k:v for k,v in example.items() if k != 'data'} == {k:v for k,v in config.items() if k != 'data'}
    assert TrainingConfig.from_config(config) == TrainingConfig()
    protocol = ScientificTrainingProtocol(config)
    assert protocol.batch_size == 8 and protocol.max_epochs == 100
    assert protocol.robustness_seeds == [602, 603, 604, 605]
    for seed in protocol.robustness_seeds:
        validate_role('ROBUSTNESS', seed)
    with pytest.raises(ValueError):
        validate_role('PRIMARY', 602)
    from mcminet.losses import MCMINetObjective, ProxyMetricLearning
    for objective in (MCMINetObjective(), build_objective(TrainingConfig())):
        assert [getattr(objective, 'lambda_'+n) for n in ('cls','intra','cross','hybrid','specific')] == [1., .2, .3, .3, .2]
        assert [getattr(objective.proxy_metric, 'tau_'+n) for n in ('intra','cross','hybrid','specific')] == [.1,.07,.07,.1]
    assert ProxyMetricLearning().tau_cross == .07


@pytest.mark.parametrize('path,value', [
    ('graph.k', True), ('graph.k', 0), ('graph.max_distance', -1),
    ('WSI.gat_dropout', 1.0), ('objective.tau_cross', 0), ('optimizer.lr_proxy', float('nan')),
    ('optimizer.betas', [.9]), ('optimizer.betas', [.9, True]), ('training.batch_size', 8.5),
    ('early_stopping.patience', -1), ('reproducibility.robustness_seeds', [601]),
    ('evaluation.external_test_allowed', True), ('evaluation.threshold_optimization_allowed', True),
    ('scheduler.mode', 'max'), ('model.concatenated_dim', 123), ('optimizer.proxy_weight_decay', .01),
])
def test_config_rejects_invalid_values(path, value):
    config = load_config()
    node = config
    parts = path.split('.')
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_config_rejects_missing_unknown_and_scientific_drift(tmp_path):
    config = load_config()
    del config['graph']['k']
    with pytest.raises(ValueError):
        validate_config(config)
    config = load_config(); config['objective']['single_proxy'] = .2
    with pytest.raises(ValueError):
        validate_config(config)
    config = load_config(); config['graph']['k'] = 3
    validate_config(config)
    with pytest.raises(ValueError, match='candidate configuration drift'):
        require_candidate(config)
    (tmp_path/'cycle.yaml').write_text('extends: cycle.yaml\n')
    with pytest.raises(ValueError, match='cycle'):
        load_config(tmp_path/'cycle.yaml')
    (tmp_path/'duplicate.yaml').write_text('extends: default.yaml\nextends: default.yaml\n')
    with pytest.raises(ValueError, match='Duplicate'):
        load_config(tmp_path/'duplicate.yaml')


def test_model_losses_optimizer_bn_and_native_shapes(pipeline):
    config, t, canonical, model, objective, offline, optimizer = pipeline
    validate_model_contract(canonical, config, synthetic=True)
    assert model.mri_branch.roi_encoder.projection.out_features == 256
    assert model.wsi_branch.gat_encoder.projection.out_features == 256
    assert (model.classifier.fc1.in_features, model.classifier.fc1.out_features) == (512,256)
    assert model.wsi_branch.gat_encoder.gat1.heads == model.wsi_branch.gat_encoder.gat2.heads == 2
    assert model.wsi_branch.gat_encoder.gat1.dropout == .1
    assert model.wsi_branch.graph_config == dict(k=8,max_distance=None,bidirectional=True)
    assert len(optimizer.param_groups) == 9
    ids = [id(p) for g in optimizer.param_groups for p in g['params']]
    assert len(ids) == len(set(ids))
    assert set(ids) == {id(p) for module in (model,objective) for p in module.parameters() if p.requires_grad}
    for g in optimizer.param_groups:
        assert g['lr'] == initial_group_lr(t,g)
        if g['ownership'] == 'PROXY_BANKS':
            assert g['lr'] == 5e-5 and g['weight_decay'] == 0
    by_id = {id(p):g for g in optimizer.param_groups for p in g['params']}
    for module in model.modules():
        for name, p in module.named_parameters(recurse=False):
            if name == 'bias' or isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
                assert by_id[id(p)]['weight_decay'] == 0
            else:
                assert by_id[id(p)]['weight_decay'] == .001
    bn = [m for m in model.mri_branch.modules() if isinstance(m,torch.nn.BatchNorm2d)]
    before = [(m.running_mean.clone(),m.running_var.clone()) for m in bn]
    rois = [[torch.randn(1,32+i*4,28+j*4) for i in range(2)] for j in range(3)]
    seen=[]
    hook=model.mri_branch.roi_encoder.register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape[-2:])))
    model.train()
    output=model(*rois, [torch.randn(3,512),torch.randn(4,512)],
        [torch.tensor([[0,0],[448,0],[0,448]]),torch.tensor([[0,0],[448,0],[0,448],[448,448]])],return_embeddings=True)
    hook.remove()
    assert seen == [tuple(rois[j][i].shape[-2:]) for i in range(2) for j in range(3)]
    assert output['z_mri'].shape == output['z_wsi'].shape == (2,256)
    losses=objective(output['logits'],output['z_mri'],output['z_wsi'],torch.tensor([0,1]))
    assert set(losses) == set(LOSS_NAMES) and all(torch.isfinite(v) for v in losses.values())
    expected=losses['classification']+.2*losses['intra_modal']+.3*losses['cross_modal']+.3*losses['hybrid_proxy']+.2*losses['specific_proxy']
    torch.testing.assert_close(losses['total'],expected)
    losses['total'].backward();optimizer.step()
    for m,(mean,var) in zip(bn,before):
        assert not m.training and m.weight.requires_grad and m.bias.requires_grad
        assert torch.equal(m.running_mean,mean) and torch.equal(m.running_var,var)
    assert all(not p.requires_grad and p.grad is None for p in offline.parameters())
    audit_optimizer(optimizer,model,objective,offline,t)
    optimizer.param_groups[0]['params'].append(optimizer.param_groups[0]['params'][0])
    with pytest.raises(ValueError):
        audit_optimizer(optimizer,model,objective,offline,t)


def test_cached_graph_preserves_actual_custom_settings():
    config=load_config();config['graph']=dict(k=1,max_distance=448.,bidirectional=False)
    canonical=build_model(config,smoke_test=True)
    from mcminet.models.cached_batched_mcminet import CachedBatchedMCMINet
    model=CachedBatchedMCMINet(canonical).eval()
    branch=model.wsi_branch
    assert branch.graph_config==config['graph']
    out=branch([torch.randn(3,512)],[torch.tensor([[0,0],[448,0],[4480,0]])],return_details=True)
    assert out['edge_index'].tolist()==[[0,1],[1,0]]
    with pytest.raises(ValueError,match='identity mismatch'):
        validate_model_contract(model,load_config(),synthetic=True)


def test_scheduler_stopper_external_test_and_patient_isolation():
    t=TrainingConfig()
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=t.classifier_lr)
    scheduler=ValidationScheduler(optimizer,t)
    assert scheduler.scheduler.mode=='min'
    assert scheduler.scheduler.factor==.5 and scheduler.scheduler.patience==5
    assert scheduler.scheduler.threshold==1e-4 and scheduler.scheduler.min_lrs==[1e-7]
    early=EarlyStopping.from_config(t)
    assert early.metric_name=='validation_auroc' and early.mode=='max'
    assert early.patience==15 and early.min_delta==.001
    for _ in range(7):scheduler.step(ValidationMetric('total',2.))
    assert optimizer.param_groups[0]['lr']==t.classifier_lr*.5
    with pytest.raises(ValueError):scheduler.step(ValidationMetric('validation_auroc',.6))
    with pytest.raises(ValueError):ValidationMetric('total',1.,source='external_test')
    external=CohortLoader([], 'EXTERNAL_TEST', (('external',0),('e2',1)))
    with pytest.raises(ValueError,match='EXTERNAL_TEST'):external.validate('INTERNAL_VALIDATION')
    train=CohortLoader([], 'TRAIN', (('p1',0),('p2',1)))
    validation=CohortLoader([], 'INTERNAL_VALIDATION', (('p1',0),('v2',1)))
    with pytest.raises(ValueError,match='overlap'):validate_cohorts(train,validation)
    with pytest.raises(ValueError,match='Incomplete'):list(train.batches())


def test_best_checkpoint_small_improvement_tie_and_stopper():
    # Construct synthetic rankings at finer resolution than the stopping delta.
    from mcminet.training.scientific_metrics import patient_auroc
    t=TrainingConfig();optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=1e-4)
    optimizer.param_groups[0]['name']='test'
    scheduler=ValidationScheduler(optimizer,t)
    engineering=TrainingHistory();early=scientific_early_stopping();mirror=EarlyStopping.from_config(t);rows=[]
    training=EpochResult('training',200,25,{n:1. for n in LOSS_NAMES})
    validation=EpochResult('validation',200,25,{n:1. for n in LOSS_NAMES})
    records=[dict(patient_id=f'n{i}',label=0,raw_logit=float(i+1)) for i in range(100)]
    records += [dict(patient_id=f'p{i}',label=1,raw_logit=70.5) for i in range(100)]
    for epoch,score in enumerate([70.5,75.5,75.5,81.5]):
        records[-1]['raw_logit']=score
        auc=patient_auroc(records)
        complete_epoch(epoch,training,validation,auc,records,optimizer,scheduler,mirror,engineering,early,rows)
    assert [r['best_epoch'] for r in rows]==[0,1,1,3]
    assert [r['early_stopping']['best_epoch'] for r in rows]==[0,0,0,3]
    assert [r['selected_best'] for r in rows]==[True,True,False,True]
    replay=validate_scientific_history(rows,early.state_dict(),engineering.state_dict())
    assert replay.state_dict()==early.state_dict()
    for epoch in range(4,19):early.update(ValidationMetric('validation_auroc',.7011),epoch)
    assert early.should_stop


def test_preprocessing_identity_and_thresholds():
    from mcminet.data.wsi_preprocessing import NATIVE_SIZE, STRIDE, MODEL_SIZE
    from mcminet.data.roi_constrained_wsi_extractor import ROI_THRESHOLD
    from mcminet.data.final_wsi_preprocessor import BACKGROUND_S_MAX, BACKGROUND_V_MIN, BACKGROUND_FRACTION_MAX
    from mcminet.data.wsi_feature_cache import preprocessing_identity
    assert (NATIVE_SIZE,STRIDE,MODEL_SIZE)==(448,448,224)
    assert (ROI_THRESHOLD,BACKGROUND_S_MAX,BACKGROUND_V_MIN,BACKGROUND_FRACTION_MAX)==(.7,.05,.8,.9)
    identity=preprocessing_identity()
    assert identity['policy']['version']==default('WSI.preprocessing.version')
    assert len(identity['sha256'])==64


def test_synthetic_training_resume_and_metadata(tmp_path):
    from mcminet.training.formal_training_driver import _run, ModelInputs
    from mcminet.utils.state_fingerprint import state_fingerprint
    from mcminet.data.wsi_feature_cache import preprocessing_identity
    from mcminet.training.checkpointing import load_checkpoint
    from mcminet.training.reproducibility import seeded_generator
    torch.set_num_threads(1)
    gen=torch.Generator().manual_seed(12)  # Fixture seed, not a scientific run seed.
    def cohort(prefix,role):
        batch=dict(patient_ids=[prefix+'0',prefix+'1'],labels=torch.tensor([0,1]),
            wsi_features_list=[torch.randn(3,512,generator=gen) for _ in range(2)],
            wsi_coordinates_list=[torch.tensor([[0,0],[448,0],[0,448]]) for _ in range(2)])
        for field in ('tumor_roi','peritumoral_roi','lymph_node_roi'):
            batch[field]=[torch.randn(1,24+i*4,28,generator=gen) for i in range(2)]
        return CohortLoader([batch],role,((prefix+'0',0),(prefix+'1',1)),synthetic=True)
    train=cohort('t','TRAIN');val=cohort('v','INTERNAL_VALIDATION')
    identities=[];offline_encoders=[]
    def factory():
        canonical=build_model(smoke_test=True)
        offline=canonical.wsi_branch.patch_encoder
        identity=dict(encoder_state_sha256=state_fingerprint(offline),
                      checkpoint_sha256='0'*64,  # Explicit synthetic provenance only.
                      preprocessing_sha256=preprocessing_identity()['sha256'])
        identities.append(identity);offline_encoders.append(offline)
        return ModelInputs(canonical,identity)
    def run(folder,cap,resume=None):
        return _run(model_factory=factory,train=train,validation=val,output_directory=folder,
            role='PRIMARY',seed=601,device='cpu',protocol=None,config=None,
            resume_checkpoint=resume,generator=None,dry_run_cap=cap)
    uninterrupted=run(tmp_path/'full',2)
    run(tmp_path/'resume',1)
    resumed=run(tmp_path/'resume',2,tmp_path/'resume'/'last.json')
    assert resumed.history==uninterrupted.history
    assert resumed.scheduler.state_dict()==uninterrupted.scheduler.state_dict()
    for name,value in uninterrupted.model.state_dict().items():
        assert torch.equal(value,resumed.model.state_dict()[name]),name
    for name,value in uninterrupted.objective.state_dict().items():
        assert torch.equal(value,resumed.objective.state_dict()[name]),name
    pointers=json.loads((tmp_path/'resume'/'last.json').read_text())
    folder=tmp_path/'resume'/pointers['last']['bundle']
    payload=torch.load(folder/'state.pt',weights_only=True)
    identity=payload['identity']
    assert identity['graph']==load_config()['graph']
    assert identity['candidate_configuration']['model']['concatenated_dim']==512
    assert identity['candidate_configuration']['objective']==load_config()['objective']
    assert identity['candidate_configuration']['scheduler']==load_config()['scheduler']
    assert identity['training_configuration']['seed']==601
    assert identity['encoder_initialization']==dict(mri_pretrained=False,wsi_pretrained=False)
    assert identity['smoke_test'] is True
    assert payload['schema_version']==2
    early=EarlyStopping.from_config(TrainingConfig());early.load_state_dict(payload['early_stopping'])
    history=TrainingHistory();history.load_state_dict(payload['history'])
    kwargs=dict(model=resumed.model,objective=resumed.objective,optimizer=resumed.optimizer,
        scheduler=resumed.scheduler,early_stopping=early,history=history,config=TrainingConfig(),
        offline_encoder=offline_encoders[-1],cache_identity=identities[-1],generator=seeded_generator(601))
    snapshot=state_fingerprint(resumed.model)
    payload['schema_version']=1
    torch.save(payload,tmp_path/'legacy.pt')
    with pytest.raises(ValueError,match='legacy checkpoints cannot resume'):
        load_checkpoint(tmp_path/'legacy.pt',**kwargs)
    assert state_fingerprint(resumed.model)==snapshot
    payload['schema_version']=2;payload['identity']['graph']['k']=1
    torch.save(payload,tmp_path/'tampered.pt')
    with pytest.raises(ValueError,match='identity mismatch'):
        load_checkpoint(tmp_path/'tampered.pt',**kwargs)
    assert state_fingerprint(resumed.model)==snapshot


def test_cache_provenance_round_trip_and_graph_guard(tmp_path):
    from mcminet.models.wsi_patch_encoder import WSIPatchEncoder
    from mcminet.data.wsi_feature_cache import expected_provenance, make_record, save_cache, load_cache
    encoder=WSIPatchEncoder(pretrained=False,trainable=False).eval()
    (tmp_path/'synthetic_wsi').write_bytes(b'synthetic source identity fixture')
    (tmp_path/'synthetic_geojson').write_text('{}')
    expected=expected_provenance('fixture',tmp_path/'synthetic_wsi',tmp_path/'synthetic_geojson',encoder,'0'*64)
    features=torch.randn(2,512);coordinates=torch.tensor([[0,0],[448,0]])
    record=make_record(features,coordinates,expected)
    save_cache(tmp_path/'cache.pt',record,expected)
    loaded=load_cache(tmp_path/'cache.pt',expected)
    assert torch.equal(loaded['features'],features)
    expected['graph_policy']['k']=1
    with pytest.raises(ValueError,match='provenance'):
        load_cache(tmp_path/'cache.pt',expected)


def test_external_cohort_rejected_before_model_factory(tmp_path):
    from mcminet.training.formal_training_driver import run_training
    called=[]
    def factory():
        called.append(True)
        raise AssertionError('External cohort reached model construction')
    train=CohortLoader([], 'TRAIN', (('t0',0),('t1',1)))
    external=CohortLoader([], 'EXTERNAL_TEST', (('e0',0),('e1',1)))
    with pytest.raises(ValueError,match='EXTERNAL_TEST'):
        run_training(model_factory=factory,train=train,validation=external,output_directory=tmp_path/'never')
    assert not called and not (tmp_path/'never').exists()


def test_actual_model_drift_and_unpretrained_formal_run_rejected(pipeline):
    config,t,canonical,model,objective,offline,optimizer=pipeline
    with pytest.raises(ValueError,match='pretrained'):
        validate_model_contract(canonical,config,synthetic=False)
    canonical.wsi_branch.gat_encoder.gat1.dropout=0.0
    with pytest.raises(ValueError,match='GAT dropout'):
        validate_model_contract(canonical,config,synthetic=True)
