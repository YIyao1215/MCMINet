import torch
from mcminet.models import MCMINet
from mcminet.data import PatientDataset, PatientSample
from mcminet.losses import MCMINetObjective
from mcminet.training import Trainer, TrainingConfig

def test_public_imports_and_sample_validation():
    assert MCMINet.__name__ == 'BatchedMCMINet'
    sample=PatientSample('example',torch.zeros(1,8,8),torch.zeros(1,8,8),torch.zeros(1,8,8),torch.zeros(2,3,8,8),torch.zeros(2,2,dtype=torch.long),0)
    assert len(PatientDataset([sample]))==1

def test_model_output_shape():
    model=MCMINet(mri_pretrained=False,wsi_pretrained=False)
    rois=[torch.randn(2,1,32,28),torch.randn(2,1,36,32),torch.randn(2,1,24,20)]
    features=[torch.randn(3,3,32,32),torch.randn(4,3,32,32)]
    coords=[torch.tensor([[0,0],[448,0],[0,448]]),torch.tensor([[0,0],[448,0],[0,448],[448,448]])]
    out=model(*rois,features,coords,return_embeddings=True)
    assert out['z_mri'].shape==out['z_wsi'].shape==(2,256) and out['logits'].shape==(2,)

def test_loss_finite():
    objective=MCMINetObjective()
    z=torch.nn.functional.normalize(torch.randn(4,256),dim=1)
    losses=objective(torch.randn(4),z,z,torch.tensor([0,1,0,1]))
    assert set(losses)=={'total','classification','intra_modal','cross_modal','hybrid_proxy','specific_proxy'}
    assert all(torch.isfinite(v) for v in losses.values())
