"""Teacher-model interfaces used by the main research pipeline."""

from src.teacher.base import TeacherCall, TeacherLLM, build_teacher, teacher_exchange_log

__all__ = ["TeacherCall", "TeacherLLM", "build_teacher", "teacher_exchange_log"]
