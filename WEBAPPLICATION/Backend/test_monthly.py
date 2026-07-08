import asyncio
import httpx

async def test():
    async with httpx.AsyncClient(base_url='http://127.0.0.1:8000', timeout=30) as client:
        resp = await client.get('/api/v1/explain/monthly-insights')
        data = resp.json()
        
        print('=== MONTHLY INSIGHTS ===')
        summary = data.get('summary', {})
        print(f"Overall Mean: {summary.get('overall_mean')} MW")
        
        monthly = data.get('monthly_patterns', {})
        print(f"\nMonthly patterns:")
        for month, stats in monthly.items():
            print(f"  {month}: mean={stats.get('mean')} MW, max={stats.get('max')} MW")
        
        drivers = data.get('drivers', [])
        if drivers:
            print(f"\nTop drivers:")
            for d in drivers:
                print(f"  - {d.get('name')}: {d.get('description')}")
        
        recs = data.get('recommendations', [])
        if recs:
            print(f"\nRecommendations:")
            for r in recs:
                print(f"  - {r}")

asyncio.run(test())