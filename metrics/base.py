"""Base classes for pluggable metric analysis.

Each metric module provides:
1. A compute function (pure data in, data out)
2. An Analyzer subclass (formats results for LLM prompt injection)

The Analyzer produces an AnalysisSection that the prompt builder
composes into the diversity feedback prompt.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class DiversityAnalyzer(ABC):
    """Base class for metric-specific diversity analysis.

    Each subclass knows how to interpret one type of metric result
    and produce a human-readable explanation for the LLM.
    """

    @property
    @abstractmethod
    def section_title(self) -> str:
        """Title for this analysis section in the prompt."""
        ...

    @abstractmethod
    def analyze(self) -> str:
        """Produce a human-readable analysis string.

        Returns empty string if no analysis is applicable.
        """
        ...

    def to_section(self) -> "AnalysisSection":
        """Convenience: run analyze() and wrap in an AnalysisSection."""
        body = self.analyze()
        return AnalysisSection(
            title=self.section_title,
            body=body,
            source_metric=self.__class__.__name__,
        )


@dataclass
class AnalysisSection:
    """A rendered analysis section ready for prompt injection.

    Attributes:
        title: Section header (e.g., "SPATIAL PATH ANALYSIS")
        body: Multi-line analysis text
        source_metric: Which metric/analyzer produced this (for logging)
    """
    title: str
    body: str
    source_metric: str = ""
