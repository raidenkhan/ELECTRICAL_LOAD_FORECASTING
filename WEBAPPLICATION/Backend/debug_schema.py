
from app.schemas.forecast import ForecastRequest
import pydantic
print(f"Pydantic version: {pydantic.__version__}")
field = ForecastRequest.model_fields['horizon_hours']
print(f"Horizon hours constraints: {field.metadata}")
