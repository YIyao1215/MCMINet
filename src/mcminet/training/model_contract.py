"""Verify actual model settings before attaching candidate checkpoint metadata."""
from mcminet.config import require_candidate, scientific_settings


def validate_model_contract(model, config, *, synthetic=False):
    require_candidate(config)
    attached = getattr(model, 'candidate_config', None)
    if attached is not None and scientific_settings(attached) != scientific_settings(config):
        raise ValueError('Model/config candidate identity mismatch')
    r, w, m, g = (config[k] for k in ('MRI', 'WSI', 'model', 'graph'))
    roi = model.mri_branch.roi_encoder
    fusion = model.mri_branch.fusion
    gat = model.wsi_branch.gat_encoder
    classifier = model.classifier
    graph = getattr(model.wsi_branch, 'graph_config', None)
    if graph is None:
        graph = dict(k=model.wsi_branch.graph_k, max_distance=model.wsi_branch.graph_max_distance,
                     bidirectional=model.wsi_branch.graph_bidirectional)
    checks = {
        'MRI ROI dimension': roi.projection.out_features == r['roi_embedding_dim'],
        'MRI fusion input': fusion[0].in_features == 3 * r['roi_embedding_dim'],
        'MRI fusion hidden': fusion[0].out_features == r['fusion_hidden_dim'],
        'MRI patient dimension': fusion[3].out_features == r['patient_embedding_dim'],
        'MRI dropout': fusion[2].p == r['fusion_dropout'],
        'WSI features': gat.input_dim == w['node_feature_dim'],
        'GAT layer 1': gat.gat1.out_channels == w['gat1_dim'],
        'GAT layer 2': gat.gat2.out_channels == w['gat2_dim'],
        'GAT heads': gat.gat1.heads == gat.gat2.heads == w['gat_heads'],
        'GAT dropout': gat.gat1.dropout == gat.gat2.dropout == w['gat_dropout'],
        'WSI patient dimension': gat.projection.out_features == w['patient_embedding_dim'],
        'Classifier input': classifier.fc1.in_features == m['concatenated_dim'],
        'Classifier hidden': classifier.fc1.out_features == m['classifier_hidden_dim'],
        'Classifier output': classifier.fc2.out_features == 1,
        'Classifier dropout': classifier.dropout.p == m['classifier_dropout'],
        'Graph': graph == g,
    }
    if not synthetic:
        checks['MRI pretrained'] = roi.pretrained == r['pretrained']
        offline = getattr(model.wsi_branch, 'patch_encoder', None)
        if offline is not None:
            checks['WSI pretrained'] = offline.pretrained == w['pretrained']
        checks['smoke mode disabled'] = not getattr(model, 'smoke_test', False)
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError('Model/config mismatch: ' + ', '.join(failed))
