"""
Tumor-from-EEG: a reproducible pipeline for quantitative EEG feature
extraction and subject-disjoint machine-learning evaluation.

See docs/METHODS.md for the analysis protocol and docs/LIMITATIONS.md for what
this software can and cannot support scientifically.
"""

__version__ = "2.0.0"

from .config import AnalysisConfig                      # noqa: F401
from .io import Recording, Cohort, load_cohort          # noqa: F401
