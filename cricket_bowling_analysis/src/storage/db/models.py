"""
src/storage/db/models.py
--------------------------
FUTURE — activate when scaling to 1000+ bowlers.
SQLAlchemy ORM definitions for 3 PostgreSQL tables:
    Bowler   — bowler profiles
    Session  — recording sessions
    Delivery — all biomechanics KPIs per delivery

To activate:
    1. docker-compose up -d  (start PostgreSQL)
    2. pip install sqlalchemy psycopg2-binary alembic
    3. alembic init alembic
    4. alembic revision --autogenerate -m "init"
    5. alembic upgrade head
    6. Change storage.backend to "db" in config.yaml
"""

from sqlalchemy import (Column, String, Float, Integer,
                         DateTime, Text, ForeignKey, func)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship
import uuid

Base = declarative_base()


class Bowler(Base):
    __tablename__ = "bowlers"

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name          = Column(String(100), nullable=False)
    dob           = Column(DateTime, nullable=True)
    dominant_hand = Column(String(5))
    bowling_style = Column(String(50))
    height_cm     = Column(Float)
    weight_kg     = Column(Float)
    created_at    = Column(DateTime, server_default=func.now())
    sessions      = relationship("Session", back_populates="bowler")


class Session(Base):
    __tablename__ = "sessions"

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    bowler_id      = Column(UUID(as_uuid=True), ForeignKey("bowlers.id"), nullable=False)
    recorded_at    = Column(DateTime)
    location       = Column(String(100))
    surface        = Column(String(50))
    cam_fps        = Column(Integer)
    cam_resolution = Column(String(20))
    sync_method    = Column(String(20))
    raw_path       = Column(Text)
    notes          = Column(Text)
    created_at     = Column(DateTime, server_default=func.now())
    bowler         = relationship("Bowler", back_populates="sessions")
    deliveries     = relationship("Delivery", back_populates="session")


class Delivery(Base):
    __tablename__ = "deliveries"

    id                          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id                  = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False)
    delivery_number             = Column(Integer)
    release_frame               = Column(Integer)
    release_height_px           = Column(Float)
    release_angle_deg           = Column(Float)
    elbow_angle_at_release      = Column(Float)
    shoulder_angle_at_release   = Column(Float)
    front_knee_angle_at_release = Column(Float)
    back_knee_angle_at_release  = Column(Float)
    hip_shoulder_sep_at_release = Column(Float)
    trunk_lean_at_release       = Column(Float)
    elbow_angle_max             = Column(Float)
    hip_shoulder_sep_max        = Column(Float)
    trunk_lean_max              = Column(Float)
    arm_velocity_max            = Column(Float)
    arm_velocity_mean           = Column(Float)
    runup_speed_mean            = Column(Float)
    ball_speed_ms               = Column(Float)
    parquet_path                = Column(Text)
    created_at                  = Column(DateTime, server_default=func.now())
    session                     = relationship("Session", back_populates="deliveries")