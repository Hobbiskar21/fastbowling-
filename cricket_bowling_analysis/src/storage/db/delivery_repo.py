"""
src/storage/db/delivery_repo.py
---------------------------------
FUTURE — CRUD operations for the Delivery table.
Same save_delivery() interface as csv_writer — one import line change to switch.
"""

from src.storage.db.models import Delivery, Session as SessionModel
import uuid


def save_delivery(record: dict, session_id: str, db_session) -> str:
    delivery = Delivery(
        id=uuid.uuid4(),
        session_id=session_id,
        delivery_number=record.get("delivery_number"),
        release_frame=record.get("release_frame"),
        release_height_px=record.get("release_height_px"),
        release_angle_deg=record.get("release_angle_deg"),
        elbow_angle_at_release=record.get("elbow_angle_at_release"),
        shoulder_angle_at_release=record.get("shoulder_angle_at_release"),
        front_knee_angle_at_release=record.get("front_knee_angle_at_release"),
        back_knee_angle_at_release=record.get("back_knee_angle_at_release"),
        hip_shoulder_sep_at_release=record.get("hip_shoulder_sep_at_release"),
        trunk_lean_at_release=record.get("trunk_lean_at_release"),
        elbow_angle_max=record.get("elbow_angle_max"),
        hip_shoulder_sep_max=record.get("hip_shoulder_sep_max"),
        trunk_lean_max=record.get("trunk_lean_max"),
        arm_velocity_max=record.get("arm_velocity_max"),
        arm_velocity_mean=record.get("arm_velocity_mean"),
        runup_speed_mean=record.get("runup_speed_mean"),
        ball_speed_ms=record.get("ball_speed_ms"),
        parquet_path=record.get("parquet_path"),
    )
    db_session.add(delivery)
    db_session.commit()
    return str(delivery.id)


def get_by_session(session_id: str, db_session) -> list:
    return (db_session.query(Delivery)
            .filter(Delivery.session_id == session_id)
            .order_by(Delivery.delivery_number)
            .all())


def get_by_bowler(bowler_id: str, db_session) -> list:
    return (db_session.query(Delivery)
            .join(SessionModel, Delivery.session_id == SessionModel.id)
            .filter(SessionModel.bowler_id == bowler_id)
            .order_by(Delivery.created_at)
            .all())