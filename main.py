#!/usr/bin/env python3
"""
HoloForge - Neural Particle Swarm Holographic Display
Main application using Pygame + PyOpenGL for rendering

Designed for:
- Raspberry Pi 5 (ARM64)
- 720x720 HyperPixel display
- Camera Module 3
- Beam splitter holographic display
"""

import sys
import time
import numpy as np

# Add particle_swarm to path
sys.path.insert(0, 'particle_swarm')

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from config import *
from particle_system import ParticleSystem
from shape_library import ShapeLibrary, capture_face_pointcloud
from transition_manager import TransitionManager
from gesture_recognizer import GestureRecognizer
from camera_capture import CameraCapture, MockCamera


class HoloForgeApp:
    """
    Main application class for HoloForge holographic display.
    Uses Pygame for window management and PyOpenGL for 3D rendering.
    """

    def __init__(self):
        self.running = False
        self.clock = None
        self.start_time = 0
        self.frame_count = 0
        self.last_fps_time = 0

        # Core components
        self.particles = None
        self.shapes = None
        self.transitions = None
        self.gestures = None
        self.camera = None

        # Camera rotation for 3D effect
        self.camera_angle = 0.0

    def init_pygame(self):
        """Initialize Pygame and OpenGL context."""
        pygame.init()
        pygame.display.set_caption("HoloForge - Holographic Display")

        # Set OpenGL attributes before creating display
        pygame.display.gl_set_attribute(GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(GL_MULTISAMPLESAMPLES, 4)

        # Create OpenGL window
        flags = DOUBLEBUF | OPENGL
        try:
            # Try fullscreen on Pi
            self.screen = pygame.display.set_mode(
                (DISPLAY_WIDTH, DISPLAY_HEIGHT), flags | FULLSCREEN
            )
        except pygame.error:
            # Fallback to windowed mode
            self.screen = pygame.display.set_mode(
                (DISPLAY_WIDTH, DISPLAY_HEIGHT), flags
            )

        self.clock = pygame.time.Clock()

        # Hide mouse cursor for clean display
        pygame.mouse.set_visible(False)

        print(f"Display initialized: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")

    def init_opengl(self):
        """Configure OpenGL for particle rendering."""
        # Set viewport
        glViewport(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT)

        # Set up perspective projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, DISPLAY_WIDTH / DISPLAY_HEIGHT, 1.0, 1000.0)

        # Switch to modelview for rendering
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Enable depth testing for proper 3D layering
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)

        # Enable point smoothing for nicer particles
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)

        # Enable blending for transparency effects
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Set clear color to black
        glClearColor(0.0, 0.0, 0.0, 1.0)

        # Set point size
        glPointSize(PARTICLE_SIZE)

        print("OpenGL initialized")

    def init_components(self):
        """Initialize particle system and related components."""
        print("Initializing components...")

        # Particle system
        self.particles = ParticleSystem(PARTICLE_COUNT)

        # Shape library (pre-generates all shapes)
        self.shapes = ShapeLibrary(PARTICLE_COUNT)

        # Transition manager
        self.transitions = TransitionManager(self.shapes)

        # Gesture recognizer
        self.gestures = GestureRecognizer()

        # Camera capture (with fallback to mock)
        try:
            self.camera = CameraCapture()
            if self.camera.camera is None:
                print("Using mock camera")
                self.camera = MockCamera()
        except Exception as e:
            print(f"Camera init failed: {e}, using mock")
            self.camera = MockCamera()

        # Set initial targets
        initial_targets = self.transitions.update(0)
        if initial_targets is not None:
            self.particles.set_targets(initial_targets)

        print("Components initialized")

    def handle_events(self):
        """Process Pygame events."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE or event.key == K_q:
                    self.running = False
                elif event.key == K_SPACE:
                    # Manual skip to next shape
                    current_time = time.time() - self.start_time
                    self.transitions.skip_to_next(current_time)
                elif event.key == K_f:
                    # Toggle freeze
                    self.transitions.set_frozen(not self.transitions.frozen)
                elif event.key == K_c:
                    # Capture face
                    self.capture_face()

    def handle_gestures(self, current_time):
        """Process hand gestures from camera."""
        # Get hand landmarks from camera
        landmarks = self.camera.get_hand_landmarks()

        # Recognize gesture
        gesture = self.gestures.update(landmarks, current_time)

        if gesture == 'skip':
            self.transitions.skip_to_next(current_time)
        elif gesture == 'freeze':
            self.transitions.set_frozen(True)
        elif gesture == 'capture_face':
            self.capture_face()
        elif gesture in ['scatter', 'attract']:
            # Apply hand force to particles
            hand_pos = self.gestures.get_hand_position_3d(
                landmarks, DISPLAY_WIDTH, DISPLAY_HEIGHT
            )
            return {
                'position': hand_pos,
                'gesture': gesture,
                'strength': HAND_FORCE_STRENGTH
            }

        return None

    def capture_face(self):
        """Capture face from camera and add to shapes."""
        frame = self.camera.capture_for_face()
        if frame is not None:
            face_points = capture_face_pointcloud(frame, PARTICLE_COUNT)
            if face_points is not None:
                self.shapes.set_face(face_points)
                print("Face captured and added to shape library")
            else:
                print("No face detected in frame")
        else:
            print("Could not capture frame for face")

    def update(self, dt, current_time):
        """Update particle system and transitions."""
        # Handle gestures and get hand force
        hand_force = self.handle_gestures(current_time)

        # Update transition manager
        targets = self.transitions.update(current_time)
        if targets is not None:
            self.particles.set_targets(targets)

        # Update particle physics
        self.particles.update(dt, hand_force)

        # Update camera rotation for 3D effect
        self.camera_angle += AUTO_ROTATE_SPEED * dt

    def render(self):
        """Render particles using OpenGL."""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up camera view
        glLoadIdentity()
        gluLookAt(
            CAMERA_EYE_Z * np.sin(self.camera_angle), 0, CAMERA_EYE_Z * np.cos(self.camera_angle),  # Eye position (rotating)
            0, 0, 0,  # Look at center
            0, 1, 0   # Up vector
        )

        # Render particles as points
        glBegin(GL_POINTS)
        for i in range(self.particles.count):
            # Set color (normalized to 0-1 for OpenGL)
            r, g, b = self.particles.colors[i] / 255.0
            glColor3f(r, g, b)

            # Set vertex position
            x, y, z = self.particles.positions[i]
            glVertex3f(x, y, z)
        glEnd()

        # Swap buffers
        pygame.display.flip()

    def render_optimized(self):
        """Optimized rendering using vertex arrays."""
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set up camera view
        glLoadIdentity()
        gluLookAt(
            CAMERA_EYE_Z * np.sin(self.camera_angle), 0, CAMERA_EYE_Z * np.cos(self.camera_angle),
            0, 0, 0,
            0, 1, 0
        )

        # Enable vertex arrays
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)

        # Prepare data - ensure contiguous arrays with correct type
        positions = np.ascontiguousarray(self.particles.positions, dtype=np.float32)
        colors = np.ascontiguousarray(self.particles.colors / 255.0, dtype=np.float32)

        # Set vertex and color pointers
        glVertexPointer(3, GL_FLOAT, 0, positions)
        glColorPointer(3, GL_FLOAT, 0, colors)

        # Draw all particles at once
        glDrawArrays(GL_POINTS, 0, self.particles.count)

        # Disable vertex arrays
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_COLOR_ARRAY)

        # Swap buffers
        pygame.display.flip()

    def log_fps(self, current_time):
        """Log FPS periodically."""
        self.frame_count += 1

        if current_time - self.last_fps_time >= 5.0:
            fps = self.frame_count / (current_time - self.last_fps_time)
            if VERBOSE_LOGGING:
                print(f"FPS: {fps:.1f} | State: {self.transitions.state} | Shape: {self.transitions.get_current_shape_name()}")
            self.frame_count = 0
            self.last_fps_time = current_time

    def run(self):
        """Main application loop."""
        print("Starting HoloForge...")

        # Initialize all components
        self.init_pygame()
        self.init_opengl()
        self.init_components()

        self.running = True
        self.start_time = time.time()
        self.last_fps_time = 0

        print("Entering main loop...")

        try:
            while self.running:
                # Calculate timing
                current_time = time.time() - self.start_time
                dt = 1.0 / FRAME_RATE  # Fixed timestep for consistent physics

                # Process events
                self.handle_events()

                # Update simulation
                self.update(dt, current_time)

                # Render (use optimized version for better performance)
                self.render_optimized()

                # Log FPS
                self.log_fps(current_time)

                # Cap frame rate
                self.clock.tick(FRAME_RATE)

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        print("Cleaning up...")

        if self.camera is not None:
            self.camera.cleanup()

        pygame.quit()
        print("Goodbye!")


def main():
    """Entry point."""
    app = HoloForgeApp()
    app.run()


if __name__ == "__main__":
    main()
