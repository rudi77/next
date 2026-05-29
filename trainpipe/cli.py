def main() -> None:
    import uvicorn

    from .settings import settings

    uvicorn.run(
        "trainpipe.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
