"""The OpenAPI document (source of the web's TypeScript types) covers the SSE event contract and
the error format (docs/architecture.md §6)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from vramforge_api.app import EVENTS_PATH, create_app
from vramforge_estimator.schemas import AnalysisEvent, Branch, EventType

EVENT_SCHEMAS = ("AnalysisEvent", "EventType", "JobProgress", "ShardProgress")


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return create_app().openapi()


def _refs(node: Any) -> set[str]:
    if isinstance(node, dict):
        found = {node["$ref"]} if isinstance(node.get("$ref"), str) else set()
        return found.union(*(_refs(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_refs(v) for v in node))
    return set()


def test_event_contract_is_in_the_components(spec: dict[str, Any]) -> None:
    schemas = spec["components"]["schemas"]
    for name in EVENT_SCHEMAS:
        assert name in schemas, name
    event = schemas["AnalysisEvent"]
    assert set(event["properties"]) == set(AnalysisEvent.model_fields)
    assert event["properties"]["type"] == {"$ref": "#/components/schemas/EventType"}
    assert schemas["EventType"]["enum"] == [t.value for t in EventType]
    assert "default" not in json.dumps(event["properties"]["progress"])  # like FastAPI's own


def test_partial_keys_are_documented(spec: dict[str, Any]) -> None:
    text = spec["components"]["schemas"]["AnalysisEvent"]["properties"]["partial"]["description"]
    for key in ("`status`", "`rows_ok`", "`rows_failed`", *(f"`max_{b.value}`" for b in Branch)):
        assert key in text, key


def test_events_route_points_at_the_event_schema(spec: dict[str, Any]) -> None:
    route = spec["paths"][EVENTS_PATH]["get"]
    assert "#/components/schemas/AnalysisEvent" in route["description"]
    item = route["responses"]["200"]["content"]["text/event-stream"]["itemSchema"]
    assert item["properties"]["data"]["contentSchema"] == {
        "$ref": "#/components/schemas/AnalysisEvent"
    }


def test_every_reference_resolves(spec: dict[str, Any]) -> None:
    schemas = spec["components"]["schemas"]
    prefix = "#/components/schemas/"
    missing = {r for r in _refs(spec) if not (r.startswith(prefix) and r[len(prefix) :] in schemas)}
    assert missing == set()


def test_errors_are_documented_as_error_response(spec: dict[str, Any]) -> None:
    assert "HTTPValidationError" not in spec["components"]["schemas"]
    for path, methods in spec["paths"].items():
        for method, operation in methods.items():
            for status, response in operation.get("responses", {}).items():
                if not status.startswith(("4", "5")):
                    continue
                schema = response["content"]["application/json"]["schema"]
                assert schema == {"$ref": "#/components/schemas/ErrorResponse"}, (
                    method,
                    path,
                    status,
                )
