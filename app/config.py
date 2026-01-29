# pharmacy_deserts/app/config.py
"""
Centralized configuration management for the Pharmacy Desert Explorer.
Loads settings from environment variables with sensible defaults.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, rely on system environment


@dataclass
class Config:
    """Application configuration loaded from environment variables."""
    
    # Environment: 'development' or 'production'
    environment: str = 'development'
    
    # AWS S3 Configuration
    aws_s3_bucket: Optional[str] = None
    aws_region: str = 'us-east-1'
    
    # Authentication
    app_password: Optional[str] = None
    require_auth: bool = False
    
    # Data paths (relative paths, will be prefixed with S3 bucket in production)
    data_dir: str = 'raw_data'
    results_dir: str = 'results'
    
    # Dataset configuration
    active_dataset_id: Optional[str] = None  # Load from this dataset config instead of default files
    
    # Streamlit settings
    server_port: int = 8501
    server_headless: bool = True
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return not self.is_production
    
    def get_data_path(self, relative_path: str) -> str:
        """
        Get full data path based on environment.
        
        In production, returns S3 URI.
        In development, returns local path.
        """
        if self.is_production and self.aws_s3_bucket:
            return f"s3://{self.aws_s3_bucket}/{relative_path}"
        return relative_path
    
    def get_financial_data_path(self) -> str:
        return self.get_data_path(f"{self.data_dir}/financial_data.csv")
    
    def get_health_data_path(self) -> str:
        return self.get_data_path(f"{self.data_dir}/health_data.csv")
    
    def get_pharmacy_data_path(self) -> str:
        return self.get_data_path(f"{self.data_dir}/Pharmacy_list_ZIP_fixed_final")
    
    def get_population_data_path(self) -> str:
        return self.get_data_path(f"{self.data_dir}/population_data.csv")
    
    def get_hhi_data_path(self) -> str:
        return self.get_data_path(f"{self.data_dir}/HHI_data.xlsx")
    
    def get_hud_crosswalk_path(self) -> str:
        return self.get_data_path(f"{self.data_dir}/zip_county_cross.xlsx")
    
    def get_county_desert_path(self) -> str:
        return self.get_data_path(f"{self.data_dir}/driving-time-desert.csv")
    
    def get_glm_results_path(self) -> str:
        return self.get_data_path(f"{self.results_dir}/national_ifae_rank.csv")


def load_config() -> Config:
    """
    Load configuration from environment variables.
    
    Returns:
        Config object with all settings
    """
    return Config(
        environment=os.getenv('ENVIRONMENT', 'development'),
        aws_s3_bucket=os.getenv('AWS_S3_BUCKET'),
        aws_region=os.getenv('AWS_REGION', 'us-east-1'),
        app_password=os.getenv('APP_PASSWORD'),
        require_auth=os.getenv('REQUIRE_AUTH', 'false').lower() == 'true',
        data_dir=os.getenv('DATA_DIR', 'raw_data'),
        results_dir=os.getenv('RESULTS_DIR', 'results'),
        active_dataset_id=os.getenv('ACTIVE_DATASET_ID'),
        server_port=int(os.getenv('STREAMLIT_SERVER_PORT', '8501')),
        server_headless=os.getenv('STREAMLIT_SERVER_HEADLESS', 'true').lower() == 'true',
    )


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    Creates it on first access.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> Config:
    """
    Force reload configuration from environment.
    Useful after changing environment variables.
    """
    global _config
    _config = load_config()
    return _config

