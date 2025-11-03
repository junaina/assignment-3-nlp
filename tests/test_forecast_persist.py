import json

def test_forecasts_api_populated(client):
    # First call chart to ensure forecasts are generated/saved
    client.get("/chart?symbol=TEST&horizon=24")
    # Now read back
    r = client.get("/api/forecasts/TEST")
    assert r.status_code == 200
    data = r.get_json()
    assert data["symbol"] == "TEST"
    assert isinstance(data["items"], list)
    assert len(data["items"]) > 0  # should have saved future points
    # spot-check one item
    item = data["items"][0]
    for k in ("created_at", "horizon_h", "model", "target_ts", "pred_close"):
        assert k in item
