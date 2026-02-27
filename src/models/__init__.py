"""
Module models — Architectures Teacher et Student pour Depth Anything V2.
"""

from .backbone import DINOv2Backbone
from .decoder import DPTDecoder
from .teacher import TeacherModel
from .student import StudentModel

__all__ = [
    "DINOv2Backbone",
    "DPTDecoder",
    "TeacherModel",
    "StudentModel",
]
