"""
Configuration parameters for Neural Particle Swarm
All tunable parameters in one place for easy optimization
"""

# Particle System
PARTICLE_COUNT = 6000
PARTICLE_SIZE = 3
PARTICLE_DAMPING = 0.92
MAX_VELOCITY = 15.0

# Physics
SPRING_STRENGTH = 0.15
HAND_FORCE_STRENGTH = 8.0
HAND_FORCE_RADIUS = 150.0
BOID_SEPARATION_RADIUS = 15.0
BOID_SEPARATION_STRENGTH = 0.8
ENABLE_BOID_SEPARATION = False  # Disable for performance if needed

# Shape Cycling
SHAPE_HOLD_TIME = 8.0  # seconds
DISSOLVE_TIME = 1.0
SWIRL_TIME = 0.5
FORM_TIME = 1.5

# Colors (depth-based gradient)
COLOR_NEAR = (0, 255, 255)    # Cyan
COLOR_MID = (255, 255, 255)    # White
COLOR_FAR = (255, 0, 102)      # Magenta
Z_NEAR_THRESHOLD = 30
Z_FAR_THRESHOLD = -30

# Gestures
GESTURE_HOLD_TIME = 0.3  # seconds to confirm gesture
SKIP_COOLDOWN = 2.0      # seconds between skips

# Display
DISPLAY_WIDTH = 720
DISPLAY_HEIGHT = 720
FRAME_RATE = 60
BACKGROUND_COLOR = (0, 0, 0)

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Hand Tracking
HAND_MIN_DETECTION_CONFIDENCE = 0.5
HAND_MIN_TRACKING_CONFIDENCE = 0.5

# 3D Camera Settings
CAMERA_EYE_Z = 500  # Distance from origin
AUTO_ROTATE_SPEED = 0.1  # Radians per second
