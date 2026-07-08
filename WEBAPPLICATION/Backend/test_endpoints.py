import asyncio
import httpx

async def test_all_endpoints():
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000', timeout=30) as client:
        print("=== Testing All Frontend Endpoints ===\n")
        
        # 1. STLF
        try:
            resp = await client.post('/api/v1/forecast/stlf', json={'horizon_hours': 24})
            print(f"1. /forecast/stlf: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"   MW: {data.get('forecast_mw', [])[:3]}")
        except Exception as e:
            print(f"1. /forecast/stlf: ERROR - {e}")
        
        # 2. LTLF
        try:
            resp = await client.post('/api/v1/forecast/ltlf', json={'horizon_hours': 168})
            print(f"2. /forecast/ltlf: {resp.status_code}")
        except Exception as e:
            print(f"2. /forecast/ltlf: ERROR - {e}")
        
        # 3. Current
        try:
            resp = await client.get('/api/v1/forecast/current')
            print(f"3. /forecast/current: {resp.status_code}")
        except Exception as e:
            print(f"3. /forecast/current: ERROR - {e}")
        
        # 4. Simulate
        try:
            resp = await client.post('/api/v1/forecast/simulate', json={'horizon_hours': 24, 'temp_offset': 0})
            print(f"4. /forecast/simulate: {resp.status_code}")
        except Exception as e:
            print(f"4. /forecast/simulate: ERROR - {e}")
        
        # 5. Peak Decomposition
        try:
            resp = await client.get('/api/v1/explain/peak-decomposition')
            print(f"5. /explain/peak-decomposition: {resp.status_code}")
        except Exception as e:
            print(f"5. /explain/peak-decomposition: ERROR - {e}")
        
        # 6. Model Metrics
        try:
            resp = await client.get('/api/v1/models/metrics')
            print(f"6. /models/metrics: {resp.status_code}")
        except Exception as e:
            print(f"6. /models/metrics: ERROR - {e}")
        
        # 7. Data Latest
        try:
            resp = await client.get('/api/v1/data/latest')
            print(f"7. /data/latest: {resp.status_code}")
        except Exception as e:
            print(f"7. /data/latest: ERROR - {e}")
        
        # 8. Alerts
        try:
            resp = await client.get('/api/v1/alerts/')
            print(f"8. /alerts/: {resp.status_code}")
        except Exception as e:
            print(f"8. /alerts/: ERROR - {e}")
        
        print("\n=== All endpoint tests complete ===")

asyncio.run(test_all_endpoints())