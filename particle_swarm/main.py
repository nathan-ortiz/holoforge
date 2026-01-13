"""
Neural Particle Swarm - Main Application
Real-time particle system with hand gesture control
"""

import py5
import time
from particle_system import ParticleSystem
from shape_library import ShapeLibrary, capture_face_pointcloud
from gesture_recognizer import GestureRecognizer
from transition_manager import TransitionManager
from camera_capture import CameraCapture, MockCamera
from config import *


class NeuralParticleSwarm(py5.Sketch):
    """
    Main application class for particle swarm visualization
    Integrates all components into a cohesive py5 sketch
    Uses py5 class mode by inheriting from py5.Sketch
    """

    def __init__(self):
        super().__init__()
        # Core systems
        self.particle_system = None
        self.shape_library = None
        self.gesture_recognizer = None
        self.transition_manager = None
        self.camera = None

        # Timing
        self.start_time = 0
        self.frame_times = []
        self.show_fps = True
        self.last_fps_log = 0

        # Rotation
        self.auto_rotation = 0

    def settings(self):
        """py5 settings - called first to set size and renderer"""
        self.size(DISPLAY_WIDTH, DISPLAY_HEIGHT, self.P3D)

    def setup(self):
        """py5 setup function - called once at start"""
        print("=" * 60)
        print("NEURAL PARTICLE SWARM - HoloForge")
        print("=" * 60)

        # Set frame rate
        self.frame_rate(FRAME_RATE)

        # Camera setup for 3D
        self.camera(0, 0, CAMERA_EYE_Z, 0, 0, 0, 0, 1, 0)

        # Smooth rendering
        self.smooth()

        print("\nInitializing systems...")

        # Initialize shape library (pre-compute all shapes)
        print("1/5 Loading shape library...")
        self.shape_library = ShapeLibrary(PARTICLE_COUNT)

        # Initialize particle system
        print("2/5 Creating particle system...")
        self.particle_system = ParticleSystem(PARTICLE_COUNT)

        # Set initial shape
        first_shape = self.shape_library.get_shape('dna_helix')
        self.particle_system.set_targets(first_shape)

        # Initialize transition manager
        print("3/5 Setting up transitions...")
        self.transition_manager = TransitionManager(self.shape_library)
        self.transition_manager.current_targets = first_shape.copy()

        # Initialize camera and hand tracking
        print("4/5 Initializing camera...")
        try:
            self.camera = CameraCapture()
        except Exception as e:
            print(f"Camera failed, using mock: {e}")
            self.camera = MockCamera()

        # Initialize gesture recognizer
        print("5/5 Setting up gesture recognition...")
        self.gesture_recognizer = GestureRecognizer()

        self.start_time = time.time()

        print("\n" + "=" * 60)
        print("READY! Particle swarm active.")
        print("=" * 60)
        print("\nGestures:")
        print("  Open Palm (5 fingers)  -> Scatter particles")
        print("  Pinch (thumb+index)    -> Attract particles")
        print("  Fist (closed hand)     -> Freeze/unfreeze cycling")
        print("  Peace Sign (2 fingers) -> Skip to next shape")
        print("  Thumbs Up             -> Capture face")
        print("\nPress 'f' to toggle FPS display")
        print("Press 'q' to quit")
        print("=" * 60 + "\n")

    def draw(self):
        """py5 draw function - called every frame"""
        frame_start = time.time()

        # Clear background
        self.background(*BACKGROUND_COLOR)

        # Calculate time
        current_time = time.time() - self.start_time
        dt = 1.0 / FRAME_RATE

        # 1. Get hand tracking data
        hand_landmarks = self.camera.get_hand_landmarks()

        # 2. Recognize gesture
        gesture = self.gesture_recognizer.update(hand_landmarks, current_time)

        # 3. Handle gesture actions
        hand_force = None
        if gesture == 'scatter':
            hand_pos = self.gesture_recognizer.get_hand_position_3d(
                hand_landmarks, DISPLAY_WIDTH, DISPLAY_HEIGHT
            )
            hand_force = {
                'position': hand_pos,
                'gesture': 'scatter',
                'strength': HAND_FORCE_STRENGTH
            }

        elif gesture == 'attract':
            hand_pos = self.gesture_recognizer.get_hand_position_3d(
                hand_landmarks, DISPLAY_WIDTH, DISPLAY_HEIGHT
            )
            hand_force = {
                'position': hand_pos,
                'gesture': 'attract',
                'strength': HAND_FORCE_STRENGTH
            }

        elif gesture == 'freeze':
            self.transition_manager.set_frozen(True)

        elif gesture == 'skip':
            self.transition_manager.skip_to_next(current_time)

        elif gesture == 'capture_face':
            print("Capturing face...")
            face_frame = self.camera.capture_for_face()
            if face_frame is not None:
                face_points = capture_face_pointcloud(face_frame, PARTICLE_COUNT)
                if face_points is not None:
                    self.shape_library.set_face(face_points)
                else:
                    print("No face detected in frame")
            else:
                print("Could not capture frame")

        # Handle unfreeze
        if not self.gesture_recognizer.freeze_active:
            self.transition_manager.set_frozen(False)

        # 4. Update transition state
        current_targets = self.transition_manager.update(current_time)
        if current_targets is not None:
            self.particle_system.set_targets(current_targets)

        # 5. Update particle physics
        self.particle_system.update(dt, hand_force)

        # 6. Render particles
        self.push_matrix()
        self.translate(DISPLAY_WIDTH/2, DISPLAY_HEIGHT/2, 0)

        # Add gentle rotation to shapes (except during gestures)
        if hand_force is None and not self.transition_manager.frozen:
            self.auto_rotation = (current_time * AUTO_ROTATE_SPEED) % (2 * self.PI)
            self.rotate_y(self.auto_rotation)

        self.particle_system.render(self)
        self.pop_matrix()

        # 7. Display FPS and info
        if self.show_fps:
            self.draw_ui(current_time)

        # Track frame time and optional performance logging
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

        # ENHANCEMENT #2: FPS performance logging
        if VERBOSE_LOGGING and current_time - self.last_fps_log > 5.0:
            if self.frame_times:
                avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
                min_fps = 1.0 / max(self.frame_times)
                max_fps = 1.0 / min(self.frame_times)
                print(f"Avg FPS: {avg_fps:.1f} | Min: {min_fps:.1f} | Max: {max_fps:.1f}")
                self.last_fps_log = current_time

    def draw_ui(self, current_time):
        """Draw UI overlay with FPS and status"""
        # Switch to 2D rendering for UI
        self.camera()
        self.reset_matrix()

        # FPS counter
        fps = self.get_frame_rate()
        self.fill(255, 255, 0)
        self.text_size(16)
        self.text(f"FPS: {fps:.1f}", 10, 20)

        # Current shape
        shape_name = self.transition_manager.get_current_shape_name()
        state = self.transition_manager.state
        self.text(f"Shape: {shape_name}", 10, 40)
        self.text(f"State: {state}", 10, 60)

        # Frozen indicator
        if self.transition_manager.frozen:
            self.fill(255, 100, 100)
            self.text("FROZEN", 10, 80)

        # Particle count
        self.fill(255, 255, 0)
        self.text(f"Particles: {PARTICLE_COUNT}", 10, 100)

        # Restore 3D camera
        self.camera(0, 0, CAMERA_EYE_Z, 0, 0, 0, 0, 1, 0)

    def key_pressed(self):
        """Handle keyboard input"""
        if self.key == 'f' or self.key == 'F':
            self.show_fps = not self.show_fps
            print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")

        elif self.key == 'q' or self.key == 'Q':
            print("\nShutting down...")
            self.cleanup()
            self.exit()

        elif self.key == 's' or self.key == 'S':
            # Manual skip
            self.transition_manager.skip_to_next(time.time() - self.start_time)

        elif self.key == ' ':
            # Spacebar - freeze toggle
            self.gesture_recognizer.freeze_active = not self.gesture_recognizer.freeze_active

    def cleanup(self):
        """Cleanup resources before exit"""
        if self.camera:
            self.camera.cleanup()
        print("Cleanup complete")


def main():
    """Main entry point"""
    print("\nStarting Neural Particle Swarm...")
    print("Please wait while systems initialize...\n")

    # Create and run py5 sketch in class mode
    sketch = NeuralParticleSwarm()
    py5.run_sketch(sketch_class=sketch, sketch_path=None)


if __name__ == '__main__':
    main()
