from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"
    
    class Config:
        env_file = ".env"

settings = Settings() 