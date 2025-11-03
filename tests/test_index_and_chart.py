# tests/test_index_and_chart.py
def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200
    # Has a select for instruments
    assert b"<select" in r.data

def test_chart_renders_and_saves(client):
    # Render a 24h chart for TEST (this also writes forecasts to DB)
    r = client.get("/chart?symbol=TEST&horizon=24")
    assert r.status_code == 200
    # The template includes a div#chart and metrics
    assert b'id="chart"' in r.data
    assert b"Validation" in r.data
