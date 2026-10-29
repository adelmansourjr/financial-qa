"""Secondary scoring must keep abstentions and numeric zero in their correct roles."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "score_v5_under_test", Path(__file__).parents[1] / "scripts/score_v5_official.py"
)
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


@pytest.mark.parametrize("all_abstain", [False, True])
def test_secondary_v5_scoring_preserves_denominators_and_zero(all_abstain):
    calls = []

    class Metric:
        def __init__(self):
            self.rows = []
            calls.append(self.rows)

        def __call__(self, **kwargs):
            self.rows.append(kwargs)

        def get_overall_metric(self):
            return (0.5, 0.5, 1.0, 0.0)

    rows = [
        {
            "id": "zero",
            "predicted": None if all_abstain else "0",
            "predicted_scale": "million",
            "abstained": all_abstain,
        },
        {"id": "missing", "predicted": None, "predicted_scale": "", "abstained": True},
    ]
    report = {
        "predictions": rows,
        "evaluated": 2,
        "answered": int(not all_abstain),
        "coverage": 0 if all_abstain else 0.5,
        "accuracy": 0 if all_abstain else 0.5,
    }
    questions = {name: {"answer_type": "arithmetic"} for name in ("zero", "missing")}
    result = SCORER.score_method(report, questions, Metric)
    assert len(calls[0]) == 2
    assert len(calls[1]) == int(not all_abstain)
    assert calls[0][1]["prediction"] is None
    if all_abstain:
        assert result["official_em_among_answered"] is None
    else:
        assert calls[0][0]["prediction"] == "0"
        assert calls[0][0]["pred_scale"] == "million"
        assert result["coverage"] == 0.5


def test_secondary_v5_scoring_rejects_non_arithmetic_labels():
    class Metric:
        pass

    with pytest.raises(ValueError, match="restricted to arithmetic"):
        SCORER.score_method(
            {"predictions": [{"id": "text"}]}, {"text": {"answer_type": "span"}}, Metric
        )
