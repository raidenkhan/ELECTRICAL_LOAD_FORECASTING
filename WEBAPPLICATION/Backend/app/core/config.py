from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=(),
        extra="ignore"
    )
    
    # API Settings
    PROJECT_NAME: str = "Load Forecasting API"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/loadforecast"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Model Paths
    MODEL_DIR: str = "./models"
    LTLF_RECURSIVE_PATH: str = "./models/ltlf_recursive.pkl"
    DECOMP_MODEL_PATH: str = "./models/decomp_engine.joblib"
    
    # Decomposition Model Constants
    TEMP_KNOT: float = 24.0
    MIN_LOAD_MW: float = 5.0
    
    # Feature Engineering
    LAG_FEATURES: str = "1,4,96,672"  # 15m, 1h, 24h, 7d
    ROLLING_WINDOWS: str = "96"       # 24h window
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    @property
    def lag_features_list(self) -> list[int]:
        """Convert LAG_FEATURES string to list of integers."""
        return [int(x.strip()) for x in self.LAG_FEATURES.split(",")]
    
    @property
    def rolling_windows_list(self) -> list[int]:
        """Convert ROLLING_WINDOWS string to list of integers."""
        return [int(x.strip()) for x in self.ROLLING_WINDOWS.split(",")]
    
    @property
    def cors_origins(self) -> list[str]:
        """Convert ALLOWED_ORIGINS string to list."""
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]


# Global settings instance
settings = Settings()
