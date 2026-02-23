"""Initial migration - create data tables

Revision ID: 001
Revises: 
Create Date: 2026-02-04

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create raw_data_uploads table
    op.create_table(
        'raw_data_uploads',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('filename', sa.String(length=255), nullable=False),
        sa.Column('upload_timestamp', sa.DateTime(), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('row_count', sa.Integer(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('validation_report_id', sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_raw_data_uploads'))
    )
    op.create_index(op.f('ix_raw_data_uploads_id'), 'raw_data_uploads', ['id'], unique=False)

    # Create validation_reports table
    op.create_table(
        'validation_reports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('upload_id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('total_rows', sa.Integer(), nullable=False),
        sa.Column('valid_rows', sa.Integer(), nullable=False),
        sa.Column('invalid_rows', sa.Integer(), nullable=False),
        sa.Column('anomaly_count', sa.Integer(), nullable=True),
        sa.Column('validation_summary', sa.JSON(), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column('error_messages', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_validation_reports'))
    )
    op.create_index(op.f('ix_validation_reports_id'), 'validation_reports', ['id'], unique=False)
    op.create_index(op.f('ix_validation_reports_upload_id'), 'validation_reports', ['upload_id'], unique=False)

    # Create validated_data table
    op.create_table(
        'validated_data',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('upload_id', sa.Integer(), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('total_load_mw', sa.Float(), nullable=False),
        sa.Column('line1_mw', sa.Float(), nullable=True),
        sa.Column('line2_mw', sa.Float(), nullable=True),
        sa.Column('line3_mw', sa.Float(), nullable=True),
        sa.Column('voltage_kv', sa.Float(), nullable=True),
        sa.Column('current_a', sa.Float(), nullable=True),
        sa.Column('temperature_c', sa.Float(), nullable=True),
        sa.Column('frequency_hz', sa.Float(), nullable=True),
        sa.Column('is_anomaly', sa.Boolean(), nullable=True),
        sa.Column('validation_flags', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_validated_data'))
    )
    op.create_index(op.f('ix_validated_data_id'), 'validated_data', ['id'], unique=False)
    op.create_index(op.f('ix_validated_data_upload_id'), 'validated_data', ['upload_id'], unique=False)
    op.create_index(op.f('ix_validated_data_timestamp'), 'validated_data', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_validated_data_timestamp'), table_name='validated_data')
    op.drop_index(op.f('ix_validated_data_upload_id'), table_name='validated_data')
    op.drop_index(op.f('ix_validated_data_id'), table_name='validated_data')
    op.drop_table('validated_data')
    
    op.drop_index(op.f('ix_validation_reports_upload_id'), table_name='validation_reports')
    op.drop_index(op.f('ix_validation_reports_id'), table_name='validation_reports')
    op.drop_table('validation_reports')
    
    op.drop_index(op.f('ix_raw_data_uploads_id'), table_name='raw_data_uploads')
    op.drop_table('raw_data_uploads')
