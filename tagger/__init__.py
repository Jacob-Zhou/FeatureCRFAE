# -*- coding: utf-8 -*-

from .builders.builder import Builder
from .builders.crf_ae_builder import CRFAEBuilder
from .builders.hmm_builder import FeatureHMMBuilder

__all__ = ['Builder', 'CRFAEBuilder', 'FeatureHMMBuilder']
__version__ = '0.0.1'

BUILDER = {
    builder.NAME: builder
    for builder in [Builder, CRFAEBuilder, FeatureHMMBuilder]
}
