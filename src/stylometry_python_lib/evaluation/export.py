"""Deterministic file-format exporters for evaluation artifacts."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict

from stylometry_python_lib.evaluation.report import EvaluationReport
from stylometry_python_lib.evaluation.review import HumanReviewPacket


def evaluation_report_to_json(report: EvaluationReport) -> str:
    """Serialize a composite evaluation report to deterministic JSON."""
    return json.dumps(asdict(report), indent=2, sort_keys=True, ensure_ascii=False)


def human_review_packet_to_json(packet: HumanReviewPacket) -> str:
    """Serialize a human review packet to deterministic JSON."""
    return json.dumps(asdict(packet), indent=2, sort_keys=True, ensure_ascii=False)


def human_review_packet_to_csv(packet: HumanReviewPacket) -> str:
    """Serialize a human review packet to CSV with one row per document."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["document_id", "text_excerpt", *packet.feature_names])
    for item in packet.items:
        values_by_name = dict(item.feature_values)
        row: list[str | float] = [item.document_id, item.text_excerpt]
        for name in packet.feature_names:
            if name not in values_by_name:
                raise ValueError(f"Review item {item.document_id} missing feature value: {name}")
            row.append(values_by_name[name])
        writer.writerow(row)
    return buffer.getvalue()
