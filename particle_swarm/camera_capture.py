"""
Camera Capture - Integration with Picamera2 and MediaPipe
Handles camera initialization and hand tracking
"""

import numpy as np
from config import *


class CameraCapture:
    """
    Manages camera capture and MediaPipe hand tracking
    Designed for Raspberry Pi Camera Module 3
    """

    def __init__(self):
        self.camera = None
        self.mp_hands = None
        self.hands = None
        self.current_frame = None
        self.current_landmarks = None

        self.initialize_camera()
        self.initialize_hand_tracking()

    def initialize_camera(self):
        """Initialize Picamera2 for Raspberry Pi Camera Module 3"""
        try:
            from picamera2 import Picamera2

            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            print("Camera initialized successfully")
        except PermissionError as e:
            # ENHANCEMENT #3: Better error messages for common issues
            print(f"ERROR: Permission denied accessing camera")
            print(f"  → Run: sudo usermod -aG video $USER")
            print(f"  → Then logout and login again")
            print(f"Running in demo mode without camera")
            self.camera = None
        except FileNotFoundError as e:
            print(f"ERROR: Camera device not found")
            print(f"  → Enable camera in raspi-config:")
            print(f"    sudo raspi-config")
            print(f"    Interface Options → Camera → Enable")
            print(f"  → Then reboot")
            print(f"Running in demo mode without camera")
            self.camera = None
        except Exception as e:
            print(f"ERROR: Camera initialization failed: {e}")
            print(f"  → Check camera cable connection")
            print(f"  → Verify camera is enabled: vcgencmd get_camera")
            print(f"  → Full error: {type(e).__name__}: {str(e)}")
            print(f"Running in demo mode without camera")
            self.camera = None

    def initialize_hand_tracking(self):
        """Initialize MediaPipe hand tracking"""
        try:
            import mediapipe as mp

            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=HAND_MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=HAND_MIN_TRACKING_CONFIDENCE
            )
            print("MediaPipe hand tracking initialized")
        except Exception as e:
            print(f"Warning: MediaPipe initialization failed: {e}")
            print("Hand tracking will be unavailable")
            self.hands = None

    def get_hand_landmarks(self):
        """
        Capture frame and detect hand landmarks

        Returns:
            MediaPipe hand landmarks object or None
        """
        if self.camera is None or self.hands is None:
            return None

        try:
            # ENHANCEMENT #3: Robust error handling for frame capture
            # Capture frame from camera
            self.current_frame = self.camera.capture_array()

            # MediaPipe expects RGB (already in RGB888 from config)
            results = self.hands.process(self.current_frame)

            if results.multi_hand_landmarks:
                self.current_landmarks = results.multi_hand_landmarks[0]
                return self.current_landmarks

            self.current_landmarks = None
            return None

        except RuntimeError as e:
            # Camera might have disconnected
            print(f"ERROR: Camera runtime error: {e}")
            print(f"  → Check camera connection")
            return None
        except Exception as e:
            # Generic error - log and continue
            print(f"WARNING: Frame capture error: {type(e).__name__}: {e}")
            return None

    def get_current_frame(self):
        """
        Get the most recent camera frame

        Returns:
            NumPy array (H, W, 3) RGB image or None
        """
        return self.current_frame

    def capture_for_face(self):
        """
        Capture a frame specifically for face detection

        Returns:
            NumPy array (H, W, 3) RGB image or None
        """
        if self.camera is None:
            print("Camera not available for face capture")
            return None

        try:
            frame = self.camera.capture_array()
            print("Frame captured for face detection")
            return frame
        except Exception as e:
            print(f"Error capturing frame for face: {e}")
            return None

    def cleanup(self):
        """Cleanup camera and MediaPipe resources"""
        if self.camera is not None:
            try:
                self.camera.stop()
                print("Camera stopped")
            except:
                pass

        if self.hands is not None:
            try:
                self.hands.close()
                print("MediaPipe closed")
            except:
                pass


class MockCamera:
    """
    Mock camera for testing without hardware
    Simulates hand gestures for development
    """

    def __init__(self):
        self.frame_count = 0
        print("Mock camera initialized (testing mode)")

    def get_hand_landmarks(self):
        """Return None to simulate no hand detected"""
        self.frame_count += 1
        return None

    def get_current_frame(self):
        """Return a blank frame"""
        return np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)

    def capture_for_face(self):
        """Return None - no face capture in mock mode"""
        return None

    def cleanup(self):
        """No cleanup needed for mock"""
        pass
