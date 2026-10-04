"""Owner cookie, CSRF header, optional access token and error format (plan §18)."""

from __future__ import annotations

from collections.abc import Callable

from fastapi.testclient import TestClient

from vramforge_api.settings import Settings


def test_owner_cookie_is_issued_with_strict_attributes(client: TestClient) -> None:
    resp = client.get("/api/v1/local-roots")
    assert resp.status_code == 200
    cookie = resp.headers["set-cookie"]
    assert cookie.startswith("vf_owner=")
    value = cookie.split(";")[0].split("=", 1)[1]
    assert len(value) == 43  # 256-bit urlsafe token
    assert "HttpOnly" in cookie and "SameSite=Strict" in cookie and "Path=/" in cookie
    assert "Secure" not in cookie
    again = client.get("/api/v1/local-roots")
    assert "set-cookie" not in again.headers  # the browser keeps its owner


def test_secure_cookie_flag_follows_settings(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    client = client_factory(settings_factory(cookie_secure=True))
    cookie = client.get("/api/v1/local-roots").headers["set-cookie"]
    assert "Secure" in cookie


def test_health_does_not_issue_owner_cookie(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert "set-cookie" not in resp.headers


def test_forged_owner_cookie_is_replaced(client: TestClient) -> None:
    client.cookies.set("vf_owner", "../../etc/passwd")
    resp = client.get("/api/v1/local-roots")
    assert resp.headers["set-cookie"].startswith("vf_owner=")


def test_state_changing_requests_require_csrf_header(
    settings: Settings, client_factory: Callable[..., TestClient]
) -> None:
    bare = client_factory(settings, csrf=False)
    for method, url in (
        ("POST", "/api/v1/analyses"),
        ("POST", "/api/v1/uploads"),
        ("POST", "/api/v1/sources/inspect"),
        ("DELETE", "/api/v1/analyses/" + "a" * 32),
        ("POST", "/api/v1/analyses/" + "a" * 32 + "/cancel"),
        ("POST", "/api/v1/analyses/" + "a" * 32 + "/scenarios/export"),
    ):
        resp = bare.request(method, url, json={})
        assert resp.status_code == 403, url
        body = resp.json()
        assert body["error"]["code"] == "FORBIDDEN"
        assert body["error"]["details"]["reason"] == "csrf_header_missing"
    wrong = bare.post("/api/v1/analyses", json={}, headers={"X-VramForge-Request": "0"})
    assert wrong.status_code == 403
    assert bare.get("/api/v1/local-roots").status_code == 200  # safe methods pass


def test_access_token_mode(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    client = client_factory(settings_factory(access_token="open-sesame-123"))
    assert client.get("/api/v1/health").status_code == 200
    denied = client.get("/api/v1/local-roots")
    assert denied.status_code == 401
    assert denied.json()["error"]["code"] == "UNAUTHORIZED"
    assert "set-cookie" not in denied.headers  # no owner is created for anonymous callers
    assert client.get("/api/v1/openapi.json").status_code == 401

    status = client.get("/api/v1/session").json()
    assert status == {"auth_required": True, "authenticated": False}

    bad = client.post("/api/v1/session", json={"token": "nope"})
    assert bad.status_code == 401

    ok = client.post("/api/v1/session", json={"token": "open-sesame-123"})
    assert ok.status_code == 200
    assert ok.json() == {"auth_required": True, "authenticated": True}
    cookie = ok.headers["set-cookie"]
    assert cookie.startswith("vf_access=") and "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "open-sesame-123" not in cookie  # derived value, not the raw token
    assert client.get("/api/v1/local-roots").status_code == 200
    assert client.get("/api/v1/session").json()["authenticated"] is True

    client.delete("/api/v1/session")
    client.cookies.clear()
    bearer = client.get("/api/v1/local-roots", headers={"Authorization": "Bearer open-sesame-123"})
    assert bearer.status_code == 200
    wrong = client.get("/api/v1/local-roots", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401


def test_session_without_access_token(client: TestClient) -> None:
    assert client.get("/api/v1/session").json() == {
        "auth_required": False,
        "authenticated": True,
    }


def test_oversized_json_body_is_rejected(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    client = client_factory(settings_factory(max_json_body_bytes=2048))
    resp = client.post("/api/v1/analyses", content=b"{" + b" " * 4096 + b"}")
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_validation_errors_use_error_response_without_echoing_input(client: TestClient) -> None:
    secret = "hf_" + "Z" * 30
    resp = client.post(
        "/api/v1/analyses", json={"model": {"reference": secret}, "training": "bogus"}
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"]["code"] == "INVALID_REQUEST"
    assert body["error"]["details"]["fields"]
    assert secret not in resp.text


def test_unknown_route_and_method_use_error_response(client: TestClient) -> None:
    resp = client.get("/api/v1/nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "NOT_FOUND"
    resp = client.put("/api/v1/health")
    assert resp.status_code == 405
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"


def test_unhandled_errors_hide_internals(
    settings: Settings, client_factory: Callable[..., TestClient], monkeypatch
) -> None:
    from vramforge_estimator import compatibility

    def boom() -> None:
        raise RuntimeError("/srv/secret/path hf_" + "Q" * 30)

    monkeypatch.setattr(compatibility, "backend_profiles", boom)
    client = client_factory(settings, raise_server_exceptions=False)
    resp = client.get("/api/v1/backend-profiles")
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "/srv/secret" not in resp.text and "Traceback" not in resp.text


def test_owner_survives_malformed_foreign_cookies(client: TestClient) -> None:
    """Cookies are host-scoped, not port-scoped: other local apps' cookies must not break ours."""
    first = client.get("/api/v1/local-roots")
    owner = first.headers["set-cookie"].split(";")[0].split("=", 1)[1]
    client.cookies.clear()
    resp = client.get(
        "/api/v1/local-roots",
        headers={"Cookie": f'other=a b; broken="unterminated; vf_owner={owner}; tail=1'},
    )
    assert resp.status_code == 200
    assert "set-cookie" not in resp.headers  # the existing owner was recognized


def test_chunked_json_body_over_the_cap_is_413(
    settings_factory: Callable[..., Settings], client_factory: Callable[..., TestClient]
) -> None:
    """Without Content-Length the cap is enforced while reading; FastAPI's body parsing must not
    turn that into a generic 400."""
    client = client_factory(settings_factory(max_json_body_bytes=2048))
    body = b'{"schema_version": "1.0"' + b" " * 4096 + b"}"

    def chunks():
        for i in range(0, len(body), 512):
            yield body[i : i + 512]

    resp = client.post(
        "/api/v1/analyses", content=chunks(), headers={"Content-Type": "application/json"}
    )
    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "INVALID_REQUEST"
    assert resp.json()["error"]["user_message"] == "요청 본문이 너무 큽니다."
