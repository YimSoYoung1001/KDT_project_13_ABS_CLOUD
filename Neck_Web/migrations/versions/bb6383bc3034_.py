"""empty message

Revision ID: bb6383bc3034
Revises: 
Create Date: 2024-06-16 15:39:42.326802

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = 'bb6383bc3034'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('NewTable',
    sa.Column('id', sa.String(length=100), nullable=False),
    sa.Column('occur', sa.String(length=100), nullable=True),
    sa.Column('date', sa.String(length=100), nullable=True),
    sa.Column('time', sa.String(length=100), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.drop_table('newtable')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('newtable',
    sa.Column('id', mysql.VARCHAR(length=100), nullable=False),
    sa.Column('occur', mysql.VARCHAR(length=100), nullable=True),
    sa.Column('date', mysql.VARCHAR(length=100), nullable=True),
    sa.Column('time', mysql.VARCHAR(length=100), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    mysql_collate='utf8mb4_0900_ai_ci',
    mysql_default_charset='utf8mb4',
    mysql_engine='InnoDB'
    )
    op.drop_table('NewTable')
    # ### end Alembic commands ###
