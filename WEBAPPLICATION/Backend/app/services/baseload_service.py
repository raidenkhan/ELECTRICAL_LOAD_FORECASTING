from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from app.db.models.baseload import BaseloadPlant
from app.core.logging import get_logger

logger = get_logger(__name__)

BASELOAD_SEED_DATA: list[dict] = [
    # Hydro
    {"plant_name": "AKOSOMBO", "unit_name": "1G2", "constant_mw": 140, "category": "hydro"},
    {"plant_name": "AKOSOMBO", "unit_name": "1G4", "constant_mw": 120, "category": "hydro"},
    {"plant_name": "KPONG", "unit_name": "19G1", "constant_mw": 35, "category": "hydro"},
    {"plant_name": "KPONG", "unit_name": "19G2", "constant_mw": 35, "category": "hydro"},
    # Thermal baseload
    {"plant_name": "TAPCO", "unit_name": "32G1", "constant_mw": 95, "category": "thermal"},
    {"plant_name": "TAPCO", "unit_name": "32G2", "constant_mw": 0, "category": "thermal"},
    {"plant_name": "TAPCO", "unit_name": "32G3", "constant_mw": 45, "category": "thermal"},
    {"plant_name": "TEMA THERMAL", "unit_name": "47G1", "constant_mw": 105, "category": "thermal"},
    {"plant_name": "TEMA THERMAL", "unit_name": "SIEMENS", "constant_mw": 0, "category": "thermal"},
    {"plant_name": "TEMA THERMAL", "unit_name": "MRP", "constant_mw": 0, "category": "thermal"},
    {"plant_name": "CENIT", "unit_name": "47G2", "constant_mw": 105, "category": "thermal"},
    {"plant_name": "ASOGLI", "unit_name": "GROUP 1", "constant_mw": 90, "category": "thermal"},
    {"plant_name": "ASOGLI", "unit_name": "GROUP 2", "constant_mw": 45, "category": "thermal"},
    {"plant_name": "ASOGLI", "unit_name": "GROUP 3 Gas", "constant_mw": 0, "category": "thermal"},
    {"plant_name": "ASOGLI", "unit_name": "LCO/DFO", "constant_mw": 150, "category": "thermal"},
    {"plant_name": "ASOGLI", "unit_name": "GROUP 4", "constant_mw": 180, "category": "thermal"},
    {"plant_name": "TICO", "unit_name": "32G4", "constant_mw": 110, "category": "thermal"},
    {"plant_name": "TICO", "unit_name": "32G5", "constant_mw": 110, "category": "thermal"},
    {"plant_name": "TICO", "unit_name": "32G6", "constant_mw": 110, "category": "thermal"},
    {"plant_name": "T36", "unit_name": "6G1-5", "constant_mw": 0, "category": "thermal"},
    {"plant_name": "AKSAGAS", "unit_name": None, "constant_mw": 0, "category": "thermal"},
    {"plant_name": "CENPOWER", "unit_name": "GAS", "constant_mw": 109, "category": "thermal"},
    {"plant_name": "CENPOWER", "unit_name": "LCO/DFO", "constant_mw": 0, "category": "thermal"},
    {"plant_name": "CENPOWER", "unit_name": "STEAM", "constant_mw": 62, "category": "thermal"},
    {"plant_name": "KPONE THERMAL", "unit_name": "GAS", "constant_mw": 101, "category": "thermal"},
    {"plant_name": "KPONE THERMAL", "unit_name": "DFO", "constant_mw": 0, "category": "thermal"},
    {"plant_name": "ANWOMASO THERMAL", "unit_name": "91G1-10", "constant_mw": 138, "category": "thermal"},
    {"plant_name": "TWIN CITY", "unit_name": "54G1", "constant_mw": 130, "category": "thermal"},
    {"plant_name": "TWIN CITY", "unit_name": "54G2", "constant_mw": 60, "category": "thermal"},
    {"plant_name": "AKSA Anwomaso", "unit_name": "95G1", "constant_mw": 41, "category": "thermal"},
    {"plant_name": "AKSA Anwomaso", "unit_name": "95G2", "constant_mw": 41, "category": "thermal"},
    {"plant_name": "AKSA Anwomaso", "unit_name": "95G3", "constant_mw": 41, "category": "thermal"},
    {"plant_name": "Bridge Power", "unit_name": "85G1-5", "constant_mw": 150, "category": "thermal"},
    # Interconnections
    {"plant_name": "SONABEL", "unit_name": "export", "constant_mw": 150, "category": "interconnection"},
    {"plant_name": "SONABEL", "unit_name": "import", "constant_mw": 0, "category": "interconnection"},
    {"plant_name": "CIEGenser", "unit_name": "export", "constant_mw": 89, "category": "interconnection"},
    {"plant_name": "CIEGenser", "unit_name": "import", "constant_mw": 0, "category": "interconnection"},
    {"plant_name": "CIEGenser", "unit_name": "Load Reduction", "constant_mw": 0, "category": "interconnection"},
    {"plant_name": "VRA", "unit_name": "export", "constant_mw": 0, "category": "interconnection"},
    {"plant_name": "VRA", "unit_name": "import", "constant_mw": 0, "category": "interconnection"},
    {"plant_name": "VRA", "unit_name": "Load Reduction", "constant_mw": 0, "category": "interconnection"},
    {"plant_name": "CEB", "unit_name": "wheeled", "constant_mw": 0, "category": "interconnection"},
]


class BaseloadService:

    async def get_all_active(self, db: AsyncSession) -> list[BaseloadPlant]:
        result = await db.execute(
            select(BaseloadPlant).where(BaseloadPlant.is_active == True)
        )
        return list(result.scalars().all())

    async def seed_if_empty(self, db: AsyncSession) -> int:
        """Seed baseload plants if table is empty. Returns count seeded."""
        result = await db.execute(select(BaseloadPlant).limit(1))
        existing = result.scalar_one_or_none()
        if existing:
            return 0

        for entry in BASELOAD_SEED_DATA:
            plant = BaseloadPlant(**entry)
            db.add(plant)
        await db.commit()

        result = await db.execute(select(BaseloadPlant))
        count = len(result.scalars().all())
        logger.info(f"Seeded {count} baseload plants")
        return count

    def get_plant_label(self, plant: BaseloadPlant) -> str:
        if plant.unit_name:
            return f"{plant.plant_name} - {plant.unit_name}"
        return plant.plant_name
