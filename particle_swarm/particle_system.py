"""
Particle System with vectorized NumPy operations
Handles physics simulation and rendering for thousands of particles
"""

import numpy as np
from config import *


class ParticleSystem:
    """
    High-performance particle system using vectorized NumPy operations
    All particle data stored as arrays for maximum performance
    """

    def __init__(self, count):
        self.count = count

        # All particle data as NumPy arrays (N, 3) shape
        self.positions = np.random.uniform(-100, 100, (count, 3))
        self.velocities = np.zeros((count, 3))
        self.targets = np.zeros((count, 3))
        self.colors = np.zeros((count, 3))

        # Initialize colors
        self.update_colors()

    def update(self, dt, hand_force=None):
        """
        Update particle physics using vectorized operations

        Args:
            dt: Delta time in seconds
            hand_force: Optional dict with 'position', 'gesture', 'strength'
        """
        # 1. Calculate forces toward targets (spring attraction)
        target_forces = (self.targets - self.positions) * SPRING_STRENGTH

        # 2. Add hand force field if active
        if hand_force is not None:
            hand_forces = self.calculate_hand_forces(hand_force)
            target_forces += hand_forces

        # 3. Add boid separation (optional, for organic flow)
        if ENABLE_BOID_SEPARATION:
            separation_forces = self.calculate_separation_forces()
            target_forces += separation_forces

        # 4. Update velocities and positions (vectorized)
        self.velocities += target_forces * dt
        self.velocities *= PARTICLE_DAMPING
        self.velocities = np.clip(self.velocities, -MAX_VELOCITY, MAX_VELOCITY)
        self.positions += self.velocities * dt

        # 5. Update colors based on Z-depth
        self.update_colors()

    def calculate_hand_forces(self, hand_state):
        """
        Calculate force field from hand gesture (vectorized with early culling)

        Args:
            hand_state: dict with 'position' (x,y,z), 'gesture' ('scatter'/'attract'), 'strength'

        Returns:
            NumPy array (N, 3) of force vectors
        """
        hand_pos = np.array(hand_state['position'], dtype=np.float64)

        # Validate hand position - reject NaN or Inf values
        if not np.all(np.isfinite(hand_pos)):
            return np.zeros_like(self.positions)

        # ENHANCEMENT #5: Early distance culling for performance
        # Calculate distances first, then only process nearby particles
        deltas = self.positions - hand_pos
        distances = np.linalg.norm(deltas, axis=1, keepdims=True)

        # Early exit if no particles in range
        in_radius = (distances.flatten() < HAND_FORCE_RADIUS)
        if not np.any(in_radius):
            return np.zeros_like(self.positions)

        # Avoid division by zero
        distances = np.maximum(distances, 1.0)

        # Only calculate forces for particles within radius (saves computation)
        forces = np.zeros_like(self.positions)

        # Inverse square force (like gravity/magnetism)
        force_magnitudes = hand_state['strength'] / (distances ** 2)

        # Direction: away for scatter, toward for attract
        if hand_state['gesture'] == 'scatter':
            directions = deltas / distances  # Normalize, pointing away
        else:  # attract
            directions = -deltas / distances  # Normalize, pointing toward

        # Apply force only to particles within radius
        forces[in_radius] = (directions * force_magnitudes)[in_radius]

        return forces

    def calculate_separation_forces(self):
        """
        Calculate boid-style separation forces (vectorized, but expensive)
        Only use if performance allows

        Returns:
            NumPy array (N, 3) of separation force vectors
        """
        # This is O(N^2) - use only with small particle counts or disable
        forces = np.zeros_like(self.positions)

        for i in range(self.count):
            # Calculate distances to all other particles
            deltas = self.positions - self.positions[i]
            distances = np.linalg.norm(deltas, axis=1, keepdims=True)
            distances = np.maximum(distances, 0.1)  # Avoid self-distance

            # Apply separation force to nearby particles
            too_close = (distances.flatten() < BOID_SEPARATION_RADIUS) & (distances.flatten() > 0)

            if np.any(too_close):
                separation = deltas[too_close] / distances[too_close]
                forces[i] = np.mean(separation, axis=0) * BOID_SEPARATION_STRENGTH

        return forces

    def update_colors(self):
        """
        Update particle colors based on Z-depth (fully vectorized)
        Creates gradient: far (magenta) -> mid (white) -> near (cyan)
        CRITICAL: No Python loops - uses NumPy broadcasting for performance
        """
        z_values = self.positions[:, 2]

        # Normalize Z to [0, 1] for color interpolation
        # Protect against division by zero if thresholds are equal
        z_range = Z_NEAR_THRESHOLD - Z_FAR_THRESHOLD
        if abs(z_range) < 1e-6:
            z_norm = np.full_like(z_values, 0.5)  # All particles at midpoint
        else:
            z_norm = np.clip(
                (z_values - Z_FAR_THRESHOLD) / z_range,
                0, 1
            )

        # CRITICAL FIX #2: Fully vectorized color interpolation
        # Create boolean masks for the two color regions
        is_lower_half = z_norm < 0.5

        # Calculate interpolation parameter t for both regions
        # Lower half (far -> mid): t goes from 0 to 1 as z_norm goes from 0 to 0.5
        t_lower = z_norm * 2
        # Upper half (mid -> near): t goes from 0 to 1 as z_norm goes from 0.5 to 1
        t_upper = (z_norm - 0.5) * 2

        # Vectorized linear interpolation using broadcasting
        # shape (N,) * shape (3,) broadcasts to (N, 3)
        color_far = np.array(COLOR_FAR)
        color_mid = np.array(COLOR_MID)
        color_near = np.array(COLOR_NEAR)

        # Interpolate for lower half (far -> mid)
        colors_lower = color_far[np.newaxis, :] * (1 - t_lower[:, np.newaxis]) + \
                      color_mid[np.newaxis, :] * t_lower[:, np.newaxis]

        # Interpolate for upper half (mid -> near)
        colors_upper = color_mid[np.newaxis, :] * (1 - t_upper[:, np.newaxis]) + \
                      color_near[np.newaxis, :] * t_upper[:, np.newaxis]

        # Combine using np.where with broadcasting
        self.colors = np.where(is_lower_half[:, np.newaxis], colors_lower, colors_upper)

    def render(self, sketch):
        """
        Render all particles using py5's efficient point rendering

        Args:
            sketch: py5 sketch instance
        """
        # MODERATE FIX #5: Performance-critical rendering section
        # stroke_weight is set once outside the loop for efficiency
        sketch.stroke_weight(PARTICLE_SIZE)

        # Per-particle colors require iteration (no way around this without shaders)
        # Using begin_shape(POINTS) is the most efficient non-shader approach
        sketch.begin_shape(sketch.POINTS)
        for i in range(self.count):
            sketch.stroke(*self.colors[i])
            sketch.vertex(*self.positions[i])
        sketch.end_shape()

    def set_targets(self, target_positions):
        """
        Set new target positions for all particles

        Args:
            target_positions: NumPy array (N, 3) of target positions
        """
        # ENHANCEMENT #4: Particle count validation with automatic resampling
        if target_positions is None:
            return

        # Validate array - reject if contains NaN or Inf
        if not np.all(np.isfinite(target_positions)):
            print("WARNING: Target positions contain NaN or Inf values, skipping update")
            return

        original_count = len(target_positions)

        if original_count != self.count:
            # Resample to match particle count
            if original_count < self.count:
                # Upsample: repeat with random selection
                indices = np.random.choice(original_count, self.count, replace=True)
            else:
                # Downsample: random selection without replacement
                indices = np.random.choice(original_count, self.count, replace=False)

            target_positions = target_positions[indices]
            print(f"WARNING: Resampled targets from {original_count} to {self.count} particles")

        self.targets = target_positions.copy()
