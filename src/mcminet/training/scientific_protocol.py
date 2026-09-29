"""Prespecified candidate protocol; never a claim of optimized hyperparameters."""
import hashlib
import json
from mcminet.config import default, load_config, require_candidate, scientific_settings
from mcminet.training.training_config import TrainingConfig


class ScientificTrainingProtocol:
    def __init__(self, config=None):
        self.config = load_config() if config is None else config
        require_candidate(self.config)
        self.training = TrainingConfig.from_config(self.config)

    def __getattr__(self, name):
        aliases = {'primary_seed': ('reproducibility', 'primary_seed'),
                   'robustness_seeds': ('reproducibility', 'robustness_seeds'),
                   'score_convention': ('evaluation', 'score_convention'),
                   'tie_policy': ('evaluation', 'tie_policy'),
                   'external_test_allowed': ('evaluation', 'external_test_allowed'),
                   'threshold_optimization_allowed': ('evaluation', 'threshold_optimization_allowed')}
        if name in aliases:
            section, key = aliases[name]
            return self.config[section][key]
        return getattr(self.training, name)

    def to_dict(self):
        return scientific_settings(self.config)

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                              separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def validate(self):
        require_candidate(self.config)


def validate_role(role, seed):
    primary = default('reproducibility.primary_seed')
    robustness = default('reproducibility.robustness_seeds')
    if type(seed) is not int or not ((role == 'PRIMARY' and seed == primary) or
                                    (role == 'ROBUSTNESS' and seed in robustness)):
        raise ValueError(f'PRIMARY requires seed {primary}; ROBUSTNESS requires one of {robustness}')


def engineering_config(seed=None):
    """Compatibility name for the canonical training adapter."""
    return TrainingConfig.from_yaml(seed=seed)


def validate_config(config, seed):
    expected = engineering_config(seed).to_dict()
    actual = config.to_dict()
    changed = [k for k in expected if actual[k] != expected[k] or type(actual[k]) != type(expected[k])]
    if changed:
        raise ValueError('Formal candidate training configuration drift: ' + ', '.join(changed))
