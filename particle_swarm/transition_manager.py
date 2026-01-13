"""
Transition Manager - State machine for smooth shape morphing
Handles auto-cycling and manual transitions between shapes
"""

import numpy as np
from config import *


class TransitionManager:
    """
    Manages transitions between shapes with dissolve/swirl/form phases
    States: HOLDING -> DISSOLVING -> SWIRLING -> FORMING -> HOLDING

    All target arrays are guaranteed to have exactly particle_count points.
    """

    STATES = ['HOLDING', 'DISSOLVING', 'SWIRLING', 'FORMING']

    def __init__(self, shape_library):
        self.shape_library = shape_library
        self.particle_count = shape_library.particle_count
        self.current_shape_index = 0
        self.state = 'HOLDING'
        self.state_start_time = 0
        self.frozen = False

        self.current_targets = None
        self.next_targets = None
        self.swirl_center = np.zeros(3)

        # Pre-generated random targets for dissolve phase (to avoid jitter)
        self.dissolve_targets = None

        # Initialize with first shape
        shape_order = self.shape_library.get_shape_order()
        first_shape = self._get_validated_shape(shape_order[0])
        self.current_targets = first_shape.copy()

    def _get_validated_shape(self, shape_name):
        """
        Get a shape and validate/resample it to match particle_count.

        Args:
            shape_name: Name of the shape to get

        Returns:
            NumPy array (particle_count, 3) or None if shape unavailable
        """
        shape = self.shape_library.get_shape(shape_name)
        if shape is None:
            return None

        # Resample if needed to ensure exact count
        if len(shape) != self.particle_count:
            if len(shape) > self.particle_count:
                indices = np.random.choice(len(shape), self.particle_count, replace=False)
            else:
                indices = np.random.choice(len(shape), self.particle_count, replace=True)
            shape = shape[indices]

        return shape

    def update(self, current_time):
        """
        Update transition state and return current target positions

        Args:
            current_time: Current time in seconds

        Returns:
            NumPy array (N, 3) of target positions for particles
        """
        if self.frozen:
            return self.current_targets

        elapsed = current_time - self.state_start_time

        if self.state == 'HOLDING':
            if elapsed > SHAPE_HOLD_TIME:
                self.start_dissolve(current_time)
            return self.current_targets

        elif self.state == 'DISSOLVING':
            progress = min(elapsed / DISSOLVE_TIME, 1.0)
            targets = self.interpolate_dissolve(progress)

            if elapsed > DISSOLVE_TIME:
                self.start_swirl(current_time)

            return targets

        elif self.state == 'SWIRLING':
            targets = self.generate_swirl_targets(elapsed)

            if elapsed > SWIRL_TIME:
                self.start_forming(current_time)

            return targets

        elif self.state == 'FORMING':
            progress = min(elapsed / FORM_TIME, 1.0)
            targets = self.interpolate_forming(progress)

            if elapsed > FORM_TIME:
                self.start_holding(current_time)

            return targets

        return self.current_targets

    def start_dissolve(self, current_time):
        """Begin dissolve phase - particles scatter randomly"""
        self.state = 'DISSOLVING'
        self.state_start_time = current_time
        # Pre-generate random targets once to avoid jitter during dissolve
        self.dissolve_targets = np.random.uniform(-150, 150, (self.particle_count, 3))
        print(f"Dissolving from {self.get_current_shape_name()}")

    def start_swirl(self, current_time):
        """Begin swirl phase - particles rotate in space"""
        self.state = 'SWIRLING'
        self.state_start_time = current_time

        # Load next shape, skipping any None shapes (like uncaptured face)
        shape_order = self.shape_library.get_shape_order()
        attempts = 0
        max_attempts = len(shape_order)

        while attempts < max_attempts:
            self.current_shape_index = (self.current_shape_index + 1) % len(shape_order)
            next_shape_name = shape_order[self.current_shape_index]
            self.next_targets = self._get_validated_shape(next_shape_name)

            if self.next_targets is not None:
                break
            attempts += 1

        # Fallback: if all shapes are None, use current targets
        if self.next_targets is None:
            print("Warning: No valid next shape found, staying on current shape")
            self.next_targets = self.current_targets.copy()

        # Pre-generate swirl base positions for consistent animation
        self.swirl_base = np.random.uniform(-100, 100, (self.particle_count, 3))

        print(f"Swirling...")

    def start_forming(self, current_time):
        """Begin forming phase - particles converge to next shape"""
        self.state = 'FORMING'
        self.state_start_time = current_time
        print(f"Forming {self.get_current_shape_name()}")

    def start_holding(self, current_time):
        """Begin holding phase - particles hold current shape"""
        self.state = 'HOLDING'
        self.state_start_time = current_time
        self.current_targets = self.next_targets.copy()
        print(f"Holding {self.get_current_shape_name()}")

    def interpolate_dissolve(self, progress):
        """
        Particles scatter randomly from current shape

        Args:
            progress: 0.0 to 1.0

        Returns:
            NumPy array (particle_count, 3) of interpolated positions
        """
        # Ease-in cubic for acceleration
        eased_progress = progress ** 3

        # Use pre-generated dissolve targets (not random each frame)
        if self.dissolve_targets is None:
            self.dissolve_targets = np.random.uniform(-150, 150, (self.particle_count, 3))

        return self.current_targets * (1 - eased_progress) + self.dissolve_targets * eased_progress

    def generate_swirl_targets(self, elapsed):
        """
        Particles swirl in circular motion

        Args:
            elapsed: Time elapsed in swirl phase

        Returns:
            NumPy array (particle_count, 3) of swirl positions
        """
        # Multiple rotations during swirl
        angle = elapsed * 4 * np.pi  # Two full rotations
        rotation_matrix = self.rotation_matrix_y(angle)

        # Use pre-generated swirl base positions (not random each frame)
        if not hasattr(self, 'swirl_base') or self.swirl_base is None:
            self.swirl_base = np.random.uniform(-100, 100, (self.particle_count, 3))

        rotated = self.swirl_base @ rotation_matrix.T

        return rotated

    def interpolate_forming(self, progress):
        """
        Ease particles from swirl into next shape

        Args:
            progress: 0.0 to 1.0

        Returns:
            NumPy array (particle_count, 3) of interpolated positions
        """
        # Ease-out cubic for smooth arrival
        eased_progress = 1 - (1 - progress) ** 3

        # Use swirl_base at elapsed=0 for starting position
        swirl_targets = self.generate_swirl_targets(0)

        # Safety: ensure arrays match in size
        if self.next_targets is None:
            return swirl_targets

        if len(swirl_targets) != len(self.next_targets):
            # This should never happen with validated shapes, but be safe
            print(f"Warning: Size mismatch in interpolate_forming: {len(swirl_targets)} vs {len(self.next_targets)}")
            if len(self.next_targets) > self.particle_count:
                indices = np.random.choice(len(self.next_targets), self.particle_count, replace=False)
            else:
                indices = np.random.choice(len(self.next_targets), self.particle_count, replace=True)
            self.next_targets = self.next_targets[indices]

        return swirl_targets * (1 - eased_progress) + self.next_targets * eased_progress

    def rotation_matrix_y(self, angle):
        """
        3D rotation matrix around Y axis

        Args:
            angle: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    def rotation_matrix_z(self, angle):
        """
        3D rotation matrix around Z axis

        Args:
            angle: Rotation angle in radians

        Returns:
            3x3 rotation matrix
        """
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    def skip_to_next(self, current_time):
        """
        Manually trigger transition to next shape

        Args:
            current_time: Current time in seconds
        """
        if self.state == 'HOLDING':
            print("Skipping to next shape...")
            self.start_dissolve(current_time)

    def set_frozen(self, frozen):
        """
        Freeze or unfreeze shape cycling

        Args:
            frozen: Boolean freeze state
        """
        self.frozen = frozen
        if frozen:
            print("Shape cycling frozen")
        else:
            print("Shape cycling resumed")

    def get_current_shape_name(self):
        """Get name of current shape"""
        shape_order = self.shape_library.get_shape_order()
        return shape_order[self.current_shape_index]

    def get_state_info(self):
        """
        Get current state information for debugging

        Returns:
            Dict with state details
        """
        return {
            'state': self.state,
            'shape': self.get_current_shape_name(),
            'frozen': self.frozen,
            'elapsed': 0  # Would need current_time to calculate
        }
