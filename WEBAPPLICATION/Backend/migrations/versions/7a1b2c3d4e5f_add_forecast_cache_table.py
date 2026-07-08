"""add forecast cache table

Revision ID: 7a1b2c3d4e5f
Revises: 568e0b942d8a
Create Date: 2026-05-26 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7a1b2c3d4e5f'
down_revision = '568e0b942d8a'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('forecast_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cache_key', sa.String(length=100), nullable=False),
        sa.Column('horizon', sa.String(length=10), nullable=False),
        sa.Column('forecast_date', sa.Date(), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('cache_key'),
    )
    op.create_index(op.f('ix_forecast_cache_id'), 'forecast_cache', ['id'], unique=False)
    op.create_index(op.f('ix_forecast_cache_cache_key'), 'forecast_cache', ['cache_key'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_forecast_cache_cache_key'), table_name='forecast_cache')
    op.drop_index(op.f('ix_forecast_cache_id'), table_name='forecast_cache')
    op.drop_table('forecast_cache')
