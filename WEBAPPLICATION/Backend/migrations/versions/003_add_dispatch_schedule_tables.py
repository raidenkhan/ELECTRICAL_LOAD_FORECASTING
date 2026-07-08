"""Add dispatch schedule tables

Revision ID: 003
Revises: 9cc320bfb605
Create Date: 2026-05-25

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers
revision = '003'
down_revision = '9cc320bfb605'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'daily_dispatch_schedules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('source_filename', sa.String(length=255), nullable=False),
        sa.Column('operator_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_daily_dispatch_schedules')),
    )
    op.create_index(
        op.f('ix_daily_dispatch_schedules_date'),
        'daily_dispatch_schedules', ['date'], unique=True,
    )
    op.create_index(
        op.f('ix_daily_dispatch_schedules_id'),
        'daily_dispatch_schedules', ['id'], unique=False,
    )

    op.create_table(
        'hourly_demand',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('schedule_id', sa.Integer(), nullable=False),
        sa.Column('hour', sa.Integer(), nullable=False),
        sa.Column('entity_name', sa.String(length=100), nullable=False),
        sa.Column('demand_mw', sa.Float(), nullable=False),
        sa.Column('is_forecasted', sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_hourly_demand')),
    )
    op.create_index(
        op.f('ix_hourly_demand_id'),
        'hourly_demand', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_hourly_demand_schedule_id'),
        'hourly_demand', ['schedule_id'], unique=False,
    )

    op.create_table(
        'hourly_supply',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('schedule_id', sa.Integer(), nullable=False),
        sa.Column('hour', sa.Integer(), nullable=False),
        sa.Column('plant_name', sa.String(length=100), nullable=False),
        sa.Column('supply_mw', sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_hourly_supply')),
    )
    op.create_index(
        op.f('ix_hourly_supply_id'),
        'hourly_supply', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_hourly_supply_schedule_id'),
        'hourly_supply', ['schedule_id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_hourly_supply_schedule_id'), table_name='hourly_supply')
    op.drop_index(op.f('ix_hourly_supply_id'), table_name='hourly_supply')
    op.drop_table('hourly_supply')

    op.drop_index(op.f('ix_hourly_demand_schedule_id'), table_name='hourly_demand')
    op.drop_index(op.f('ix_hourly_demand_id'), table_name='hourly_demand')
    op.drop_table('hourly_demand')

    op.drop_index(op.f('ix_daily_dispatch_schedules_id'), table_name='daily_dispatch_schedules')
    op.drop_index(op.f('ix_daily_dispatch_schedules_date'), table_name='daily_dispatch_schedules')
    op.drop_table('daily_dispatch_schedules')
