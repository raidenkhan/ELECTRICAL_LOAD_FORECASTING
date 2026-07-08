"""add audit logs table

Revision ID: 568e0b942d8a
Revises: e401205b9ef0
Create Date: 2026-05-26 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '568e0b942d8a'
down_revision = 'e401205b9ef0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('schedule_id', sa.Integer(), nullable=False),
        sa.Column('action', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('hash', sa.String(length=64), nullable=False),
        sa.Column('previous_hash', sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(['schedule_id'], ['daily_dispatch_schedules.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_audit_logs_id'), 'audit_logs', ['id'], unique=False)
    op.create_index(op.f('ix_audit_logs_schedule_id'), 'audit_logs', ['schedule_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_audit_logs_schedule_id'), table_name='audit_logs')
    op.drop_index(op.f('ix_audit_logs_id'), table_name='audit_logs')
    op.drop_table('audit_logs')
