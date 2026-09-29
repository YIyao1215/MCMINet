"""Public data records and collation helpers. Heavy optional readers load lazily."""
from mcminet.data.patient_dataset import MCMINetPatientDataset, PatientSample, mcminet_collate_fn
PatientDataset = MCMINetPatientDataset

def variable_mri_collate(*args, **kwargs):
    from mcminet.data.variable_mri_collate import variable_mri_collate as fn
    return fn(*args, **kwargs)

def __getattr__(name):
    if name in {'CachedPatientDataset','CachedPatientRecord','CachedPatientSample','cached_collate_fn'}:
        from mcminet.data.cached_patient_dataset import CachedPatientDataset, CachedPatientRecord, CachedPatientSample, cached_collate_fn
        return locals()[name]
    raise AttributeError(name)
__all__=['PatientDataset','PatientSample','mcminet_collate_fn','variable_mri_collate']
