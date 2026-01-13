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
        Calculate force field from hand gesture (vectorized)

        Args:
            hand_state: dict with 'position' (x,y,z), 'gesture' ('scatter'/'attract'), 'strength'

        Returns:
            NumPy array (N, 3) of force vectors
        """
        hand_pos = np.array(hand_state['position'])

        # Vectorized distance calculation
        deltas = self.positions - hand_pos
        distances = np.linalg.norm(deltas, axis=1, keepdims=True)
        distances = np.maximum(distances, 1.0)  # Avoid division by zero

        # Inverse square force (like gravity/magnetism)
        force_magnitudes = hand_state['strength'] / (distances ** 2)

        # Direction: away for scatter, toward for attract
        if hand_state['gesture'] == 'scatter':
            directions = deltas / distances  # Normalize, pointing away
        else:  # attract
            directions = -deltas / distances  # Normalize, pointing toward

        # Apply force only within radius
        in_radius = (distances.flatten() < HAND_FORCE_RADIUS)
        forces = directions * force_magnitudes
        forces[~in_radius] = 0

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
        Update particle colors based on Z-depth (vectorized)
        Creates gradient: far (magenta) -> mid (white) -> near (cyan)
        """
        z_values = self.positions[:, 2]

        # Normalize Z to [0, 1] for color interpolation
        z_norm = np.clip(
            (z_values - Z_FAR_THRESHOLD) / (Z_NEAR_THRESHOLD - Z_FAR_THRESHOLD),
            0, 1
        )

        # Interpolate between far → mid → near colors
        # Split at z_norm = 0.5
        colors = np.zeros((self.count, 3))

        for i in range(self.count):
            if z_norm[i] < 0.5:
                # Far to mid
                t = z_norm[i] * 2
                colors[i] = self.lerp_color(COLOR_FAR, COLOR_MID, t)
            else:
                # Mid to near
                t = (z_norm[i] - 0.5) * 2
                colors[i] = self.lerp_color(COLOR_MID, COLOR_NEAR, t)

        self.colors = colors

    def lerp_color(self, c1, c2, t):
        """Linear interpolation between two colors"""
        return np.array(c1) * (1 - t) + np.array(c2) * t

    def render(self, sketch):
        """
        Render all particles using py5's efficient point rendering

        Args:
            sketch: py5 sketch instance
        """
        sketch.stroke_weight(PARTICLE_SIZE)

        # Efficient point rendering with py5
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
        if target_positions is not None and len(target_positions) == self.count:
            self.targets = target_positions.copy()
