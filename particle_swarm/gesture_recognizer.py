"""
Gesture Recognizer - Hand gesture detection using MediaPipe
Converts hand landmarks to actionable gestures with hold times
"""

import numpy as np
from config import *


class GestureRecognizer:
    """
    Recognizes hand gestures from MediaPipe landmarks
    Implements hold times and cooldowns for robust detection
    """

    def __init__(self):
        self.current_gesture = None
        self.gesture_start_time = 0
        self.last_skip_time = 0
        self.last_capture_time = 0
        self.freeze_active = False

        # MediaPipe hand tracking (initialized externally)
        self.hand_tracker = None

    def recognize_gesture(self, hand_landmarks):
        """
        Analyze hand landmarks to determine gesture

        Args:
            hand_landmarks: MediaPipe hand landmarks object

        Returns:
            Gesture name string or None
        """
        if hand_landmarks is None:
            return None

        # Extract finger states
        fingers_extended = self.get_fingers_extended(hand_landmarks)
        thumb_index_distance = self.get_thumb_index_distance(hand_landmarks)

        # Gesture logic (priority order)
        # 1. Scatter - Open palm (all 5 fingers extended)
        if sum(fingers_extended) == 5:
            return 'scatter'

        # 2. Attract - Pinch (thumb and index close together)
        if thumb_index_distance < 0.05:  # Normalized distance
            return 'attract'

        # 3. Freeze toggle - Fist (no fingers extended)
        if sum(fingers_extended) == 0:
            return 'freeze_toggle'

        # 4. Skip - Peace sign (index and middle extended only)
        if fingers_extended[1] and fingers_extended[2] and sum(fingers_extended) == 2:
            return 'skip'

        # 5. Capture face - Thumbs up (thumb extended only)
        if fingers_extended[0] and sum(fingers_extended[1:]) == 0:
            return 'capture_face'

        return 'neutral'  # Hand present but no clear gesture

    def get_fingers_extended(self, landmarks):
        """
        Determine which fingers are extended

        Args:
            landmarks: MediaPipe hand landmarks

        Returns:
            List of 5 booleans [thumb, index, middle, ring, pinky]
        """
        # MediaPipe hand landmark indices
        # Finger tips: [4, 8, 12, 16, 20]
        # Finger PIPs: [2, 6, 10, 14, 18]

        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [2, 6, 10, 14, 18]

        extended = []
        for tip_idx, pip_idx in zip(finger_tips, finger_pips):
            tip = landmarks.landmark[tip_idx]
            pip = landmarks.landmark[pip_idx]

            # For most fingers, check if tip is above PIP (lower y value)
            # For thumb, check horizontal distance instead
            if tip_idx == 4:  # Thumb
                wrist = landmarks.landmark[0]
                # Thumb extended if tip is far from wrist in x
                extended.append(abs(tip.x - wrist.x) > 0.1)
            else:
                # Other fingers extended if tip y < pip y
                extended.append(tip.y < pip.y)

        return extended

    def get_thumb_index_distance(self, landmarks):
        """
        Calculate normalized distance between thumb tip and index tip

        Args:
            landmarks: MediaPipe hand landmarks

        Returns:
            Float distance (normalized 0-1)
        """
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]

        dx = thumb_tip.x - index_tip.x
        dy = thumb_tip.y - index_tip.y
        dz = thumb_tip.z - index_tip.z

        return np.sqrt(dx**2 + dy**2 + dz**2)

    def update(self, hand_landmarks, current_time):
        """
        Process hand landmarks and return confirmed gesture
        Requires hold time for some gestures

        Args:
            hand_landmarks: MediaPipe hand landmarks or None
            current_time: Current time in seconds

        Returns:
            Confirmed gesture string or None
        """
        detected_gesture = self.recognize_gesture(hand_landmarks)

        # Handle instant gestures (no hold required)

        # Freeze toggle - instant but stateful
        if detected_gesture == 'freeze_toggle':
            if self.current_gesture != 'freeze_toggle':
                self.freeze_active = not self.freeze_active
                self.current_gesture = 'freeze_toggle'
            return 'freeze' if self.freeze_active else None

        # Skip - instant with cooldown
        if detected_gesture == 'skip':
            if current_time - self.last_skip_time > SKIP_COOLDOWN:
                self.last_skip_time = current_time
                self.current_gesture = 'skip'
                return 'skip'
            return None

        # Capture face - instant with cooldown
        if detected_gesture == 'capture_face':
            if current_time - self.last_capture_time > SKIP_COOLDOWN:
                self.last_capture_time = current_time
                self.current_gesture = 'capture_face'
                return 'capture_face'
            return None

        # Handle hold gestures (scatter, attract)
        if detected_gesture in ['scatter', 'attract']:
            if self.current_gesture != detected_gesture:
                self.current_gesture = detected_gesture
                self.gesture_start_time = current_time

            # Check if held long enough
            if current_time - self.gesture_start_time >= GESTURE_HOLD_TIME:
                return detected_gesture
            return None

        # Neutral or no gesture - reset
        if detected_gesture == 'neutral' or detected_gesture is None:
            # Don't reset freeze toggle state
            if self.current_gesture not in ['freeze_toggle']:
                self.current_gesture = None

        return None

    def get_hand_position_3d(self, landmarks, display_width, display_height):
        """
        Convert 2D hand landmarks to 3D position in particle space
        Uses palm center as hand position

        Args:
            landmarks: MediaPipe hand landmarks
            display_width: Display width for scaling
            display_height: Display height for scaling

        Returns:
            Tuple (x, y, z) in particle coordinate space
        """
        if landmarks is None:
            return (0, 0, 0)

        # Use palm center (average of wrist and finger bases)
        palm_landmarks = [0, 1, 5, 9, 13, 17]
        palm_x = np.mean([landmarks.landmark[i].x for i in palm_landmarks])
        palm_y = np.mean([landmarks.landmark[i].y for i in palm_landmarks])
        palm_z = np.mean([landmarks.landmark[i].z for i in palm_landmarks])

        # CRITICAL ENHANCEMENT #2: Map to actual particle coordinate space
        # Camera: x [0,1], y [0,1], z [-0.1, 0.1] (relative depth)
        # Particle space: ±40 to ±100 depending on shape (from shape_library.py)
        #
        # Actual shape ranges:
        #   - DNA helix: height=150 (±75 in y), radius=40 (±40 in x,z)
        #   - Torus knot: R=60, r=20 (approximately ±80)
        #   - Sphere: radius=70 (±70)
        #   - Most shapes fit within ±100 cube
        #
        # Hand should cover particle volume, not extend beyond it
        # Using ±100 for X,Y ensures hand covers all shapes
        # Using ±75 for Z matches typical shape depth range

        x = (palm_x - 0.5) * 200  # Range: -100 to +100
        y = (palm_y - 0.5) * 200  # Range: -100 to +100
        z = palm_z * 150          # Depth range: -75 to +75

        return (x, y, z)

    def reset(self):
        """Reset all gesture state"""
        self.current_gesture = None
        self.gesture_start_time = 0
        self.freeze_active = False
