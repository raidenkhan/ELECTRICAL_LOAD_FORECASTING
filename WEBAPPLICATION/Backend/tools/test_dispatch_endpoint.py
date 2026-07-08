"""Test Phase 2 dispatch forecast endpoint."""
import sys, os, httpx, asyncio, json

async def test():
    port = 8001
    base = f"http://localhost:{port}/api/v1"
    c = httpx.AsyncClient()

    r = await c.get(f"http://localhost:{port}/openapi.json", timeout=10)
    paths = r.json().get("paths", {})
    for path, methods in sorted(paths.items()):
        if "dispatch" in path or "forecast" in path:
            print(f"{path}: {list(methods.keys())}")

    # POST dispatch
    print("\n--- POST /forecast/dispatch ---")
    r2 = await c.post(f"{base}/forecast/dispatch", json={}, timeout=30)
    print(f"Status: {r2.status_code}")
    if r2.status_code == 200:
        data = r2.json()
        print(f"Date: {data['forecast_date']}")
        mw = data["forecast_mw"]
        print(f"MW (24h): min={min(mw):.0f}, max={max(mw):.0f}, mean={sum(mw)/len(mw):.0f}")
        if data.get("components"):
            print(f"Components: {list(data['components'].keys())}")
    else:
        print(f"Error: {r2.text[:500]}")

    # GET dispatch/tomorrow
    print("\n--- GET /forecast/dispatch/tomorrow ---")
    r3 = await c.get(f"{base}/forecast/dispatch/tomorrow", timeout=30)
    print(f"Status: {r3.status_code}")
    if r3.status_code == 200:
        data = r3.json()
        print(f"Date: {data['forecast_date']}")
        mw = data["forecast_mw"]
        print(f"MW (24h): min={min(mw):.0f}, max={max(mw):.0f}, mean={sum(mw)/len(mw):.0f}")
    else:
        print(f"Error: {r3.text[:500]}")

    await c.aclose()

asyncio.run(test())
