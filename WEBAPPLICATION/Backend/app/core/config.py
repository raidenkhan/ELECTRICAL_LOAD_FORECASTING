import sys
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    _base_dir: Optional[Path] = None

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
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://localhost:8000"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/loadforecast.db"

    # Model Paths
    MODEL_DIR: str = "./models"
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

    @property
    def BASE_DIR(self) -> Path:
        """Project root (dev) or PyInstaller bundle root (sys._MEIPASS)."""
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS)
        return Path(__file__).resolve().parent.parent.parent

    @property
    def DATA_DIR(self) -> Path:
        """Writable data directory — next to exe when frozen, else BASE_DIR/data."""
        if getattr(sys, 'frozen', False):
            return Path(sys.executable).parent / "data"
        return self.BASE_DIR / "data"

    @property
    def FRONTEND_DIR(self) -> Path:
        """Frontend static build directory.
        
        PyInstaller: bundled at BASE_DIR/frontend/out/
        Dev:         WEBAPPLICATION/frontend/out/ (sibling of Backend/)
        """
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            return Path(sys._MEIPASS) / "frontend" / "out"
        return self.BASE_DIR.parent / "frontend" / "out"


# Global settings instance
settings = Settings()
