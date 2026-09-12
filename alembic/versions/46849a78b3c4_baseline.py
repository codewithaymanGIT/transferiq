"""baseline -- marks the pre-Alembic schema state, no operations

This is a deliberate no-op. The real schema was created by
database/schema.sql (loaded automatically by Postgres on first container
start), not by Alembic -- this migration exists only so Alembic has a
known starting point ("head") to build future real migrations on top of.

Running `alembic upgrade head` on a fresh database will NOT create any
tables; you still need schema.sql for that (see docker-compose.yml).
This migration is meant to be applied via `alembic stamp head` on a
database that was already created from schema.sql, as documented in
README.md.

Known gap surfaced while adopting Alembic: autogenerate detected that
`player_comparisons`, `watchlist_players`, and `player_comparison_players`
exist in schema.sql and the live database, but have no corresponding ORM
class in backend/app/models.py. These are the blueprint's optional
auth/watchlist tables (never marked core) -- left as-is here rather than
guessing whether to add ORM classes or drop the tables; flagged in
docs/model-comparison.md's follow-up notes instead.
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "46849a78b3c4"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Deliberate no-op -- see module docstring."""
    pass


def downgrade() -> None:
    """Deliberate no-op -- see module docstring."""
    pass
