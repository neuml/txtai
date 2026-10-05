"""
ORM Module
"""

# Conditional import
try:
    from sqlalchemy import Column, DateTime, Float, JSON, Integer, LargeBinary, String, Text
    from sqlalchemy.orm import DeclarativeBase

    ORM = True
except ImportError:
    ORM = False


# Standard database schema using object relational mapping (ORM).
if ORM:

    def idcolumn():
        """
        Creates an id column. This method creates an unbounded text field for platforms that support it.

        Returns:
            id column definition
        """

        return String(512).with_variant(Text(), "sqlite", "postgresql")

    class Base(DeclarativeBase):
        """
        Base mapping.
        """

    class Batch(Base):
        """
        Batch temporary table mapping.
        """

        __tablename__ = "batch"
        __table_args__ = {"prefixes": ["TEMPORARY"]}

        autoid = Column(Integer, primary_key=True, autoincrement=True)
        indexid = Column(Integer)
        id = Column(idcolumn())
        batch = Column(Integer)

    class Score(Base):
        """
        Scores temporary table mapping.
        """

        __tablename__ = "scores"
        __table_args__ = {"prefixes": ["TEMPORARY"]}

        indexid = Column(Integer, primary_key=True, autoincrement=False)
        score = Column(Float)

    class Document(Base):
        """
        Documents table mapping.
        """

        __tablename__ = "documents"

        id = Column(idcolumn(), primary_key=True)
        data = Column(JSON)
        tags = Column(Text)
        entry = Column(DateTime(timezone=True))

    class Object(Base):
        """
        Objects table mapping.
        """

        __tablename__ = "objects"

        id = Column(idcolumn(), primary_key=True)
        object = Column(LargeBinary)
        tags = Column(Text)
        entry = Column(DateTime(timezone=True))

    class SectionBase(Base):
        """
        Generic sections table mapping. Allows multiple section table names for reindexing.
        """

        __abstract__ = True

        indexid = Column(Integer, primary_key=True, autoincrement=False)
        id = Column(idcolumn(), index=True)
        text = Column(Text)
        tags = Column(Text)
        entry = Column(DateTime(timezone=True))

    class Section(SectionBase):
        """
        Section table mapping.
        """

        __tablename__ = "sections"
