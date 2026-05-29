async def test_init_creates_expected_tables(db):
    async with db.connect() as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cur.fetchall()
    names = [r[0] for r in rows]
    for table in ("experiments", "studies", "gpu_leases", "events", "schema_version"):
        assert table in names, f"missing table: {table}"


async def test_init_is_idempotent(db):
    # Running init() again must not raise or duplicate migrations.
    await db.init()
    async with db.connect() as conn:
        cur = await conn.execute("SELECT MAX(version) FROM schema_version")
        row = await cur.fetchone()
    assert row[0] == 1


async def test_wal_mode_enabled(db):
    async with db.connect() as conn:
        cur = await conn.execute("PRAGMA journal_mode")
        mode = (await cur.fetchone())[0]
    assert mode.lower() == "wal"
