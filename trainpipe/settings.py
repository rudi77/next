from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRAINPIPE_",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    api_key: str = "dev-key-change-me"
    host: str = "0.0.0.0"
    port: int = 8080

    mlflow_tracking_uri: str = "http://localhost:5000"

    visible_gpus: list[int] | None = None

    poll_interval_sec: float = 1.0
    heartbeat_interval_sec: float = 5.0

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "trainpipe.sqlite3"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def output_base_dir(self) -> Path:
        return self.data_dir / "outputs"


settings = Settings()
