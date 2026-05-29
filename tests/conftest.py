import pytest_asyncio

from trainpipe.core.db import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    instance = Database(tmp_path / "test.sqlite3")
    await instance.init()
    return instance
