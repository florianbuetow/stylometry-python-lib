"""File-format exporters for evaluation reports and human review packets."""

import json

from stylometry_python_lib.evaluation.export import (
    evaluation_report_to_json,
    human_review_packet_to_csv,
    human_review_packet_to_json,
)
from stylometry_python_lib.evaluation.report import EvaluationReport, SplitDiagnostics
from stylometry_python_lib.evaluation.review import HumanReviewItem, HumanReviewPacket


def _packet() -> HumanReviewPacket:
    return HumanReviewPacket(
        schema_version="v1",
        items=(HumanReviewItem(document_id="d0", text_excerpt="hi", feature_values=(("f0", 1.0),)),),
        feature_names=("f0",),
        max_text_characters=200,
        validation_claim="no automated expert validation performed",
    )


def test_review_packet_json_round_trips() -> None:
    payload = json.loads(human_review_packet_to_json(_packet()))
    assert payload["schema_version"] == "v1"
    assert payload["items"][0]["document_id"] == "d0"


def test_review_packet_csv_has_header_and_row() -> None:
    csv_text = human_review_packet_to_csv(_packet())
    lines = csv_text.strip().splitlines()
    assert lines[0] == "document_id,text_excerpt,f0"
    assert lines[1].startswith("d0,")


def test_evaluation_report_json_includes_split_diagnostics() -> None:
    report = EvaluationReport(
        schema_version="v1",
        topic_leakage_score=0.3,
        ablation_scores=(("lex", 0.8),),
        length_sensitivity=(("f0", 0.1),),
        split_diagnostics=SplitDiagnostics(sample_count=10, topic_count=2, min_topic_count=3, underpowered=False, warning=""),
        family_robustness=(("lex", 0.7),),
    )
    payload = json.loads(evaluation_report_to_json(report))
    assert payload["split_diagnostics"]["sample_count"] == 10
