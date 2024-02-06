from __future__ import absolute_import

from .random_sampling import RandomSampling
from .entropy_sampling import EntropySampling
from .badge_sampling import BadgeSampling
from .coreset_sampling import CoresetSampling
from .llal_sampling import LLALSampling
from .montecarlo_sampling import MonteCarloSampling
from .confidence_sampling import ConfidenceSampling
from .crb_sampling import CRBSampling
from .temp_crb_sampling import tCRBSampling
from .crb_v2_sampling import v2CRBSampling
from .weighted_crb_sampling import wCRBSampling

__factory = {
    'random': RandomSampling,
    'entropy': EntropySampling,
    'badge': BadgeSampling,
    'coreset': CoresetSampling,
    'llal': LLALSampling,
    'montecarlo': MonteCarloSampling,
    'confidence': ConfidenceSampling,
    'crb': CRBSampling,
    'tcrb': tCRBSampling,
    'wcrb': wCRBSampling,
    'v2crb': v2CRBSampling
}

def names():
    return sorted(__factory.keys())

def build_strategy(method, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):
    if method not in __factory:
        raise KeyError("Unknown query strategy:", method)
    return __factory[method](model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)