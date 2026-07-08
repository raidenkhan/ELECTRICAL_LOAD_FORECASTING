import asyncio
import httpx

async def test():
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000', timeout=30) as client:
        # Different horizon to bypass cache
        resp = await client.post('/api/v1/forecast/stlf', json={'horizon_hours': 6})
        data = resp.json()
        p10 = data.get('p10', [])[:6]
        p90 = data.get('p90', [])[:6]
        
        meta = data.get('metadata', {})
        acc = meta.get('model_accuracy', {})
        mae = acc.get('mae')
        spread = round(p90[0] - p10[0], 1)
        
        print(f"Model MAE: {mae}")
        print(f"Uncertainty spread: {spread}")
        print(f"P10: {p10[:3]}")
        print(f"P90: {p90[:3]}")

asyncio.run(test())