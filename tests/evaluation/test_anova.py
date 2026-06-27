"""Full two-way ANOVA via the optional statsmodels extra."""

import importlib.util

import pytest

from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.evaluation.topic import TwoWayAnovaReport, two_way_anova_report

_HAS_SM = importlib.util.find_spec("statsmodels") is not None


@pytest.mark.skipif(_HAS_SM, reason="gate test only meaningful when statsmodels is absent")
def test_anova_fails_fast_without_extra() -> None:
    with pytest.raises(OptionalDependencyError, match="statsmodels"):
        two_way_anova_report(values=[1.0, 2.0, 3.0, 4.0], authors=["a", "a", "b", "b"], topics=["x", "y", "x", "y"])


@pytest.mark.skipif(not _HAS_SM, reason="requires evaluation-stats extra")
def test_anova_returns_p_values_in_unit_interval() -> None:
    values = [1.0, 1.2, 5.0, 5.3, 1.1, 1.3, 5.2, 5.1]
    authors = ["a", "a", "b", "b", "a", "a", "b", "b"]
    topics = ["x", "y", "x", "y", "x", "y", "x", "y"]
    report = two_way_anova_report(values=values, authors=authors, topics=topics)
    assert isinstance(report, TwoWayAnovaReport)
    assert 0.0 <= report.author_p_value <= 1.0
    assert 0.0 <= report.topic_p_value <= 1.0
    assert 0.0 <= report.interaction_p_value <= 1.0
