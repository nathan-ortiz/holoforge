"""
Neural Particle Swarm - Main Application
Real-time particle system with hand gesture control
Pygame + PyOpenGL rendering (migrated from py5)
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import time
import numpy as np
import math

from particle_system import ParticleSystem
from shape_library import ShapeLibrary, capture_face_pointcloud
from gesture_recognizer import GestureRecognizer
from transition_manager import TransitionManager
from camera_capture import CameraCapture, MockCamera
from config import *


class NeuralParticleSwarm:
    """
    Main application class for particle swarm visualization
    Integrates all components into a cohesive Pygame+OpenGL application
    """

    def __init__(self):
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
        self.clock = None
        self.current_fps = 0.0

        # Rotation
        self.auto_rotation = 0.0

        # Pygame/OpenGL
        self.running = False
        self.screen = None
        self.font = None

    def setup(self):
        """Initialize Pygame, OpenGL, and all subsystems"""
        print("=" * 60)
        print("NEURAL PARTICLE SWARM - HoloForge")
        print("=" * 60)

        # Initialize Pygame
        pygame.init()
        pygame.display.set_mode(
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption("Neural Particle Swarm - HoloForge")

        # Initialize clock for FPS control
        self.clock = pygame.time.Clock()

        # Initialize font for FPS display (rendered to texture)
        pygame.font.init()
        self.font = pygame.font.SysFont('monospace', 16)

        # Setup OpenGL
        self.setup_opengl()

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
        self.running = True

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

    def setup_opengl(self):
        """Configure OpenGL for 3D particle rendering"""
        # Set background color
        glClearColor(
            BACKGROUND_COLOR[0] / 255.0,
            BACKGROUND_COLOR[1] / 255.0,
            BACKGROUND_COLOR[2] / 255.0,
            1.0
        )

        # Enable depth testing for proper 3D layering
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        # Enable point smoothing for better particle appearance
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # Enable blending for smooth points
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set point size
        glPointSize(PARTICLE_SIZE)

        # Setup perspective projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, DISPLAY_WIDTH / DISPLAY_HEIGHT, 0.1, 5000.0)
        glMatrixMode(GL_MODELVIEW)

    def handle_events(self):
        """Process pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.show_fps = not self.show_fps
                    print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")

                elif event.key == pygame.K_q:
                    print("\nShutting down...")
                    self.running = False

                elif event.key == pygame.K_s:
                    # Manual skip
                    self.transition_manager.skip_to_next(
                        time.time() - self.start_time
                    )

                elif event.key == pygame.K_SPACE:
                    # Spacebar - freeze toggle
                    self.gesture_recognizer.freeze_active = (
                        not self.gesture_recognizer.freeze_active
                    )

    def draw(self):
        """Main render loop - called every frame"""
        frame_start = time.time()

        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

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

        # 6. Render particles in 3D
        self.render_particles(current_time, hand_force)

        # 7. Display FPS and info overlay
        if self.show_fps:
            self.draw_ui(current_time)

        # Swap buffers
        pygame.display.flip()

        # Track frame time and optional performance logging
        frame_time = time.time() - frame_start
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

        # Calculate current FPS
        if self.frame_times:
            self.current_fps = 1.0 / (
                sum(self.frame_times) / len(self.frame_times)
            )

        # ENHANCEMENT #2: FPS performance logging
        if VERBOSE_LOGGING and current_time - self.last_fps_log > 5.0:
            if self.frame_times:
                avg_fps = self.current_fps
                min_fps = 1.0 / max(self.frame_times)
                max_fps = 1.0 / min(self.frame_times)
                print(
                    f"Avg FPS: {avg_fps:.1f} | "
                    f"Min: {min_fps:.1f} | Max: {max_fps:.1f}"
                )
                self.last_fps_log = current_time

    def render_particles(self, current_time, hand_force):
        """Render all particles using OpenGL points"""
        # Setup modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Position camera (looking at origin from positive Z)
        glTranslatef(0, 0, -CAMERA_EYE_Z)

        # Add gentle rotation (except during gestures or when frozen)
        if hand_force is None and not self.transition_manager.frozen:
            self.auto_rotation = (
                current_time * AUTO_ROTATE_SPEED * 180.0 / math.pi
            ) % 360.0
            glRotatef(self.auto_rotation, 0, 1, 0)

        # Get particle data
        positions = self.particle_system.positions
        colors = self.particle_system.colors

        # Render particles as GL_POINTS
        glBegin(GL_POINTS)
        for i in range(self.particle_system.count):
            # Normalize color from 0-255 to 0-1
            r = colors[i, 0] / 255.0
            g = colors[i, 1] / 255.0
            b = colors[i, 2] / 255.0
            glColor3f(r, g, b)
            glVertex3f(positions[i, 0], positions[i, 1], positions[i, 2])
        glEnd()

    def draw_ui(self, current_time):
        """Draw 2D UI overlay with FPS and status"""
        # Save current matrices
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluOrtho2D(0, DISPLAY_WIDTH, 0, DISPLAY_HEIGHT)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Disable depth test for 2D overlay
        glDisable(GL_DEPTH_TEST)

        # Render text as textured quads
        self.render_text(f"FPS: {self.current_fps:.1f}", 10, DISPLAY_HEIGHT - 20)

        shape_name = self.transition_manager.get_current_shape_name()
        state = self.transition_manager.state
        self.render_text(f"Shape: {shape_name}", 10, DISPLAY_HEIGHT - 40)
        self.render_text(f"State: {state}", 10, DISPLAY_HEIGHT - 60)

        if self.transition_manager.frozen:
            self.render_text("FROZEN", 10, DISPLAY_HEIGHT - 80, color=(255, 100, 100))

        self.render_text(f"Particles: {PARTICLE_COUNT}", 10, DISPLAY_HEIGHT - 100)

        # Re-enable depth test
        glEnable(GL_DEPTH_TEST)

        # Restore matrices
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()

    def render_text(self, text, x, y, color=(255, 255, 0)):
        """Render text using pygame font and OpenGL texture"""
        # Create text surface
        text_surface = self.font.render(text, True, color, (0, 0, 0, 0))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        width, height = text_surface.get_size()

        # Create texture
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, text_data
        )

        # Enable texturing
        glEnable(GL_TEXTURE_2D)

        # Draw textured quad
        glColor3f(1, 1, 1)  # White to show texture colors
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + width, y)
        glTexCoord2f(1, 1); glVertex2f(x + width, y + height)
        glTexCoord2f(0, 1); glVertex2f(x, y + height)
        glEnd()

        # Disable texturing and cleanup
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([texture_id])

    def run(self):
        """Main application loop"""
        self.setup()

        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(FRAME_RATE)

        self.cleanup()

    def cleanup(self):
        """Cleanup resources before exit"""
        if self.camera:
            self.camera.cleanup()
        pygame.quit()
        print("Cleanup complete")


def main():
    """Main entry point"""
    print("\nStarting Neural Particle Swarm...")
    print("Please wait while systems initialize...\n")

    app = NeuralParticleSwarm()
    app.run()


if __name__ == '__main__':
    main()
