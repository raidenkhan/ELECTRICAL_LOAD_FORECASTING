
import asyncio
import asyncpg
import json

async def check():
    try:
        conn = await asyncpg.connect('postgresql://postgres:postgres@localhost:5432/loadforecast')
        row = await conn.fetchrow('SELECT error_messages, validation_summary FROM validation_reports ORDER BY id DESC LIMIT 1')
        print(f"Error Messages: {row['error_messages']}")
        print(f"Validation Summary: {json.dumps(json.loads(row['validation_summary']), indent=2)}")
        await conn.close()
    except Exception as e:
        print(f"Query failed: {e}")

if __name__ == "__main__":
    asyncio.run(check())
