"""Public training interface with lazy loading of optional data readers."""
from mcminet.training.training_config import TrainingConfig
from mcminet.training.scientific_protocol import ScientificTrainingProtocol
class Trainer:
    def __init__(self, **kwargs): self._kwargs=dict(kwargs)
    def run(self):
        from mcminet.training.formal_training_driver import run_training
        return run_training(**self._kwargs)
def run_training(*args, **kwargs):
    from mcminet.training.formal_training_driver import run_training as fn
    return fn(*args, **kwargs)
__all__=['Trainer','run_training','TrainingConfig','ScientificTrainingProtocol']
