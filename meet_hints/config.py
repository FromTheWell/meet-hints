from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Anthropic (optional until you have credentials)
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    claude_model: str = Field("claude-sonnet-4-6", alias="CLAUDE_MODEL")

    # Notion (optional until you have credentials)
    notion_api_key: str | None = Field(None, alias="NOTION_API_KEY")
    notion_parent_page_id: str | None = Field(None, alias="NOTION_PARENT_PAGE_ID")

    # Transcription
    whisper_model_size: str = Field("base", alias="WHISPER_MODEL_SIZE")
    whisper_device: str = Field("cpu", alias="WHISPER_DEVICE")
    whisper_compute_type: str = Field("int8", alias="WHISPER_COMPUTE_TYPE")

    # Recording
    output_dir: str = Field("./outputs", alias="OUTPUT_DIR")
    audio_sample_rate: int = Field(16000, alias="AUDIO_SAMPLE_RATE")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "populate_by_name": True}


settings = Settings()
