from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.deps import get_database
from app.db.models.baseload import BaseloadPlant
from app.services.baseload_service import BaseloadService
from app.core.logging import get_logger
from pydantic import BaseModel

router = APIRouter()
logger = get_logger(__name__)
baseload_service = BaseloadService()


class BaseloadPlantOut(BaseModel):
    id: int
    plant_name: str
    unit_name: str | None = None
    constant_mw: float
    category: str
    is_active: bool

    class Config:
        from_attributes = True


class BaseloadPlantUpdate(BaseModel):
    constant_mw: float


@router.get("/plants", response_model=list[BaseloadPlantOut])
async def list_baseload_plants(
    db: AsyncSession = Depends(get_database),
):
    """List all registered baseload plants."""
    plants = await baseload_service.get_all_active(db)
    return plants


@router.patch("/plants/{plant_id}", response_model=BaseloadPlantOut)
async def update_baseload_plant(
    plant_id: int,
    body: BaseloadPlantUpdate,
    db: AsyncSession = Depends(get_database),
):
    """Update a baseload plant's constant MW value."""
    plant = await db.get(BaseloadPlant, plant_id)
    if not plant:
        raise HTTPException(status_code=404, detail="Plant not found")
    plant.constant_mw = body.constant_mw
    await db.commit()
    await db.refresh(plant)
    return plant
