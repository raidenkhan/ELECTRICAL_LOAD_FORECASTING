"""add baseload plants table

Revision ID: e401205b9ef0
Revises: 004
Create Date: 2026-05-25 18:52:39.752316

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e401205b9ef0'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('baseload_plants',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('plant_name', sa.String(length=100), nullable=False),
        sa.Column('unit_name', sa.String(length=100), nullable=True),
        sa.Column('constant_mw', sa.Float(), nullable=False),
        sa.Column('category', sa.String(length=50), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_baseload_plants'))
    )
    op.create_index(op.f('ix_baseload_plants_id'), 'baseload_plants', ['id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_baseload_plants_id'), table_name='baseload_plants')
    op.drop_table('baseload_plants')
