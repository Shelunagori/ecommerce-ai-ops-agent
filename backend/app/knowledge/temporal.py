"""Effective-date semantics for versioned policies.

A version applies on ``as_of`` when ``effective_from <= as_of < effective_to`` (``effective_to``
NULL = open-ended), compared as UTC calendar dates:

* a ``date`` is used as-is;
* a ``datetime`` must be timezone-aware; it is converted to UTC and its date is used;
* a naive ``datetime`` is rejected (no timezone is ever assumed).
"""

from datetime import UTC, date, datetime


def effective_date(as_of: date | datetime) -> date:
    if isinstance(as_of, datetime):  # check first: datetime is a subclass of date
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of datetime must be timezone-aware")
        return as_of.astimezone(UTC).date()
    if isinstance(as_of, date):
        return as_of
    raise TypeError("as_of must be a date or a timezone-aware datetime")
