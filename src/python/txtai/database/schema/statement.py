"""
Statement module
"""


class Statement:
    """
    Standard database schema SQL statements.
    """

    # Temporary table for working with id batches
    CREATE_BATCH = """
        CREATE TEMP TABLE IF NOT EXISTS batch (
            indexid INTEGER,
            id TEXT,
            batch INTEGER
        )
    """

    DELETE_BATCH = "DELETE FROM batch"
    INSERT_BATCH_INDEXID = "INSERT INTO batch (indexid, batch) VALUES (?, ?)"
    INSERT_BATCH_ID = "INSERT INTO batch (id, batch) VALUES (?, ?)"

    # Temporary table for joining similarity scores
    CREATE_SCORES = """
        CREATE TEMP TABLE IF NOT EXISTS scores (
            indexid INTEGER PRIMARY KEY,
            score REAL
        )
    """

    DELETE_SCORES = "DELETE FROM scores"
    INSERT_SCORE = "INSERT INTO scores VALUES (?, ?)"

    # Documents - stores full content
    CREATE_DOCUMENTS = """
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            data JSON,
            tags TEXT,
            entry DATETIME
        )
    """

    INSERT_DOCUMENT = "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?)"
    DELETE_DOCUMENTS = "DELETE FROM documents WHERE id IN (SELECT id FROM batch)"

    # Objects - stores binary content
    CREATE_OBJECTS = """
        CREATE TABLE IF NOT EXISTS objects (
            id TEXT PRIMARY KEY,
            object BLOB,
            tags TEXT,
            entry DATETIME
        )
    """

    INSERT_OBJECT = "INSERT OR REPLACE INTO objects VALUES (?, ?, ?, ?)"
    DELETE_OBJECTS = "DELETE FROM objects WHERE id IN (SELECT id FROM batch)"

    # Sections - stores section text
    CREATE_SECTIONS = """
        CREATE TABLE IF NOT EXISTS %s (
            indexid INTEGER PRIMARY KEY,
            id TEXT,
            text TEXT,
            tags TEXT,
            entry DATETIME
        )
    """

    CREATE_SECTIONS_INDEX = "CREATE INDEX section_id ON sections(id)"
    INSERT_SECTION = "INSERT INTO sections VALUES (?, ?, ?, ?, ?)"
    DELETE_SECTIONS = "DELETE FROM sections WHERE id IN (SELECT id FROM batch)"
    COPY_SECTIONS = (
        "INSERT INTO %s SELECT (select count(*) - 1 from sections s1 where s.indexid >= s1.indexid) indexid, "
        + "s.id, %s AS text, s.tags, s.entry FROM sections s LEFT JOIN documents d ON s.id = d.id ORDER BY indexid"
    )
    STREAM_SECTIONS = (
        "SELECT s.id, s.text, data, object, s.tags FROM %s s "
        + "LEFT JOIN documents d ON s.id = d.id "
        + "LEFT JOIN objects o ON s.id = o.id ORDER BY indexid"
    )
    DROP_SECTIONS = "DROP TABLE sections"
    RENAME_SECTIONS = "ALTER TABLE %s RENAME TO sections"

    # Queries
    SELECT_IDS = "SELECT indexid, id FROM sections WHERE id in (SELECT id FROM batch)"
    COUNT_IDS = "SELECT count(indexid) FROM sections"

    # Partial sql clauses
    TABLE_CLAUSE = (
        "SELECT %s FROM sections s "
        + "LEFT JOIN documents d ON s.id = d.id "
        + "LEFT JOIN objects o ON s.id = o.id "
        + "LEFT JOIN scores sc ON s.indexid = sc.indexid"
    )
    IDS_CLAUSE = "s.indexid in (SELECT indexid from batch WHERE batch=%s)"
