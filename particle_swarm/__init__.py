"""
Neural Particle Swarm - HoloForge
Real-time particle system with hand gesture control
"""

__version__ = "1.0.0"
__author__ = "HoloForge Project"

from .particle_system import ParticleSystem
from .shape_library import ShapeLibrary
from .gesture_recognizer import GestureRecognizer
from .transition_manager import TransitionManager
from .camera_capture import CameraCapture

__all__ = [
    'ParticleSystem',
    'ShapeLibrary',
    'GestureRecognizer',
    'TransitionManager',
    'CameraCapture',
]
