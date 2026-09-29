"""Public model constructors."""
from mcminet.models.batched_mcminet import BatchedMCMINet
MCMINet = BatchedMCMINet
def __getattr__(name):
    if name == 'VariableMRICachedMCMINet':
        from mcminet.models.variable_mri_batching import VariableMRICachedMCMINet
        return VariableMRICachedMCMINet
    raise AttributeError(name)
__all__=['MCMINet','BatchedMCMINet','VariableMRICachedMCMINet']
