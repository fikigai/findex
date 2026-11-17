from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    index_path: Path = Path("index.bin")
    corpus_dir: Path = Path("data/corpus")
    app_title: str = "Findex Web Search"

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()