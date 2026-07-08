"""Add ecg_historical_demand table

Revision ID: 004
Revises: 003
Create Date: 2026-05-25

"""
from alembic import op
import sqlalchemy as sa


revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'ecg_historical_demand',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('hour', sa.Integer(), nullable=False),
        sa.Column('demand_mw', sa.Float(), nullable=False),
        sa.Column('temperature_c', sa.Float(), nullable=True),
        sa.Column('is_holiday', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_ecg_historical_demand')),
    )
    op.create_index(
        op.f('ix_ecg_historical_demand_date'),
        'ecg_historical_demand', ['date'], unique=False,
    )
    op.create_index(
        op.f('ix_ecg_historical_demand_id'),
        'ecg_historical_demand', ['id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_ecg_historical_demand_id'), table_name='ecg_historical_demand')
    op.drop_index(op.f('ix_ecg_historical_demand_date'), table_name='ecg_historical_demand')
    op.drop_table('ecg_historical_demand')
