"""Declarative base for ORM models (no domain models yet)."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
