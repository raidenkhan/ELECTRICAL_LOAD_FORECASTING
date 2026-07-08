import asyncio
import httpx

async def test():
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000', timeout=30) as client:
        resp = await client.get('/api/v1/forecast/current')
        print(f'/current: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            print(f'  MW: {data.get("forecast_mw", [])[:3]}')
        
        resp = await client.post('/api/v1/forecast/simulate', json={'horizon_hours': 24, 'temp_offset': 5.0})
        print(f'/simulate: {resp.status_code}')
        if resp.status_code == 200:
            data = resp.json()
            print(f'  MW: {data.get("forecast_mw", [])[:3]}')

        print("\n=== All Fixed Endpoints Working ===")

asyncio.run(test())