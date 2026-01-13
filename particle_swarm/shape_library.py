"""
Shape Library - Pre-computed 3D shapes for particle targets
All shapes return NumPy arrays of (N, 3) positions
"""

import numpy as np
from config import PARTICLE_COUNT


def generate_dna_helix(num_points=6000, height=150, radius=40):
    """
    Two intertwined helices with connecting rungs
    Creates iconic DNA double helix structure
    """
    points_per_strand = num_points // 2 - num_points // 10
    points_rungs = num_points // 10

    # Helix parametric equations
    t = np.linspace(0, 4 * np.pi, points_per_strand)

    # Strand 1
    x1 = radius * np.cos(t)
    y1 = np.linspace(-height/2, height/2, points_per_strand)
    z1 = radius * np.sin(t)

    # Strand 2 (180° offset)
    x2 = radius * np.cos(t + np.pi)
    y2 = y1
    z2 = radius * np.sin(t + np.pi)

    # Combine strands
    strand1 = np.column_stack([x1, y1, z1])
    strand2 = np.column_stack([x2, y2, z2])

    # Add rungs (connecting lines between strands)
    rung_indices = np.linspace(0, points_per_strand-1, points_rungs, dtype=int)
    rungs = []
    for idx in rung_indices:
        # Create line between strand points
        for alpha in np.linspace(0, 1, 5):
            rung_point = strand1[idx] * (1 - alpha) + strand2[idx] * alpha
            rungs.append(rung_point)

    rungs = np.array(rungs)

    return np.vstack([strand1, strand2, rungs])


def generate_torus_knot(num_points=6000, p=3, q=2, R=60, r=20):
    """
    Mathematical torus knot: winds p times around q times
    p=3, q=2 creates trefoil knot
    """
    t = np.linspace(0, 2 * np.pi * q, num_points)

    # Torus knot parametric equations
    x = (R + r * np.cos(p * t)) * np.cos(q * t)
    y = (R + r * np.cos(p * t)) * np.sin(q * t)
    z = r * np.sin(p * t)

    return np.column_stack([x, y, z])


def generate_lorenz_attractor(num_points=6000, scale=3.0):
    """
    Chaotic attractor with signature butterfly shape
    Numerically integrate the Lorenz system
    """
    # Lorenz system parameters
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    dt = 0.01

    # Integrate to generate trajectory
    x, y, z = 0.1, 0.0, 0.0
    points = []

    for _ in range(num_points):
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt

        x += dx
        y += dy
        z += dz

        points.append([x * scale, y * scale, z * scale])

    return np.array(points)


def generate_ellipsoid(num_points, radii, center):
    """
    Helper function: Generate random points within an ellipsoid

    Args:
        num_points: Number of points to generate
        radii: Tuple of (rx, ry, rz) ellipsoid radii
        center: Tuple of (cx, cy, cz) center position
    """
    # Random points on unit sphere, stretched to ellipsoid
    u = np.random.uniform(0, 2*np.pi, num_points)
    v = np.random.uniform(0, np.pi, num_points)

    x = radii[0] * np.sin(v) * np.cos(u) + center[0]
    y = radii[1] * np.sin(v) * np.sin(u) + center[1]
    z = radii[2] * np.cos(v) + center[2]

    return np.column_stack([x, y, z])


def load_stanford_bunny(num_points=6000):
    """
    Load pre-sampled Stanford Bunny point cloud from file
    If file doesn't exist, create simple bunny-like shape from ellipsoids
    """
    try:
        points = np.load('bunny_points.npy')
        # Resample to target count
        indices = np.random.choice(len(points), num_points, replace=True)
        return points[indices] * 150  # Scale to fit cube
    except FileNotFoundError:
        # Fallback: Create simple bunny-like shape (ellipsoids)
        body = generate_ellipsoid(num_points // 2, (40, 50, 30), (0, 0, 0))
        head = generate_ellipsoid(num_points // 4, (25, 30, 25), (0, 60, 0))
        ear1 = generate_ellipsoid(num_points // 8, (8, 25, 8), (-15, 85, 0))
        ear2 = generate_ellipsoid(num_points // 8, (8, 25, 8), (15, 85, 0))
        return np.vstack([body, head, ear1, ear2])


def generate_detailed_sphere(num_points=6000, radius=70):
    """
    Sphere with Fibonacci distribution for even point spacing
    Adds slight noise for organic appearance
    """
    # Fibonacci sphere for even distribution
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    # Base sphere positions
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)

    # Add subtle noise for organic feel
    noise = np.random.uniform(-5, 5, num_points)

    x += noise * np.cos(theta) * np.sin(phi)
    y += noise * np.sin(theta) * np.sin(phi)
    z += noise * np.cos(phi)

    return np.column_stack([x, y, z])


def generate_double_helix_torus(num_points=6000, R=60, helix_radius=10, wraps=6):
    """
    HIGH ENHANCEMENT #4: Double helix wrapped around torus - signature "wow" shape
    Combines DNA helix concept with toroidal geometry for stunning 3D depth

    Creates two intertwined helices that wind around a torus path
    Perfect for beam splitter display - complex 3D geometry from any angle

    Args:
        num_points: Number of particles
        R: Major radius of torus (distance from center to tube center)
        helix_radius: How far helices extend from torus centerline
        wraps: Number of times helices wrap around the torus (6 recommended)
    """
    # Generate parameter for torus path
    t = np.linspace(0, 2 * np.pi, num_points // 2)

    # Calculate torus centerline (circle in XY plane)
    torus_x = R * np.cos(t)
    torus_y = R * np.sin(t)
    torus_z = np.zeros_like(t)

    # First helix: winds around torus with multiple wraps
    helix_angle_1 = t * wraps  # Multiple rotations around torus

    # Calculate helix 1 positions
    # Helix extends perpendicular to torus centerline
    # Use local coordinate system that rotates with torus
    h1_x = torus_x + helix_radius * np.cos(helix_angle_1) * np.cos(t)
    h1_y = torus_y + helix_radius * np.cos(helix_angle_1) * np.sin(t)
    h1_z = helix_radius * np.sin(helix_angle_1)

    # Second helix: offset by π phase (opposite side)
    helix_angle_2 = t * wraps + np.pi

    # Calculate helix 2 positions
    h2_x = torus_x + helix_radius * np.cos(helix_angle_2) * np.cos(t)
    h2_y = torus_y + helix_radius * np.cos(helix_angle_2) * np.sin(t)
    h2_z = helix_radius * np.sin(helix_angle_2)

    # Combine both helices
    helix1 = np.column_stack([h1_x, h1_y, h1_z])
    helix2 = np.column_stack([h2_x, h2_y, h2_z])

    return np.vstack([helix1, helix2])


def generate_cube(num_points=6000, size=100):
    """
    HIGH ENHANCEMENT #3: Wireframe cube with edge emphasis and glowing vertices
    Creates sci-fi aesthetic with clear depth perception for beam splitter display

    Emphasizes:
    - 12 edges with dense point distribution
    - 8 vertices with glowing clusters
    - No solid faces (wireframe only)
    """
    half_size = size / 2

    # Define the 8 vertices of the cube
    vertices = [
        (-half_size, -half_size, -half_size),  # 0: back-bottom-left
        (half_size, -half_size, -half_size),   # 1: back-bottom-right
        (half_size, half_size, -half_size),    # 2: back-top-right
        (-half_size, half_size, -half_size),   # 3: back-top-left
        (-half_size, -half_size, half_size),   # 4: front-bottom-left
        (half_size, -half_size, half_size),    # 5: front-bottom-right
        (half_size, half_size, half_size),     # 6: front-top-right
        (-half_size, half_size, half_size),    # 7: front-top-left
    ]

    # Define the 12 edges as pairs of vertex indices
    edges = [
        # Bottom square (z = -half_size)
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top square (z = +half_size)
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical connecting edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # Allocate points: 70% to edges, 30% to vertex clusters
    points_for_edges = int(num_points * 0.7)
    points_for_vertices = num_points - points_for_edges

    points_per_edge = points_for_edges // 12
    points_per_vertex = points_for_vertices // 8

    all_points = []

    # Generate edge points
    for v1_idx, v2_idx in edges:
        v1 = np.array(vertices[v1_idx])
        v2 = np.array(vertices[v2_idx])

        # Distribute points evenly along this edge
        t = np.linspace(0, 1, points_per_edge)
        for ti in t:
            point = v1 * (1 - ti) + v2 * ti
            all_points.append(point)

    # Generate vertex glow clusters
    for vertex in vertices:
        v = np.array(vertex)

        # Create small point cluster around vertex with random offsets
        # This creates a "glowing vertex" sci-fi effect
        for _ in range(points_per_vertex):
            offset = np.random.uniform(-5, 5, 3)  # Small random offset
            glowing_point = v + offset
            all_points.append(glowing_point)

    # Convert to NumPy array
    points_array = np.array(all_points)

    # Resample to exact target count if needed
    if len(points_array) != num_points:
        indices = np.random.choice(len(points_array), num_points, replace=(len(points_array) < num_points))
        points_array = points_array[indices]

    return points_array


def capture_face_pointcloud(camera_frame, num_points=6000):
    """
    Capture face from camera and convert to 3D point cloud
    CRITICAL ENHANCEMENT #1: Uses gradient-based depth estimation for true 3D structure

    Algorithm:
    - Sobel edge detection finds high-gradient areas (facial features)
    - High gradients (eyes, nose, mouth) → protrude forward (positive Z)
    - Low gradients (flat areas, edges) → recede backward (negative Z)
    - Creates natural 3D facial structure instead of flat extrusion

    Args:
        camera_frame: RGB image from camera
        num_points: Number of points to generate

    Returns:
        NumPy array (N, 3) or None if no face detected
    """
    import cv2

    gray = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2GRAY)

    # Detect face region (use Haar cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None  # No face detected

    # Use first detected face
    (x, y, w, h) = faces[0]
    face_region = gray[y:y+h, x:x+w]

    # Resize to manageable size
    face_region = cv2.resize(face_region, (100, 100))

    # ENHANCEMENT: Calculate gradient-based depth map
    # Step 1: Sobel edge detection to find feature gradients
    grad_x = cv2.Sobel(face_region, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(face_region, cv2.CV_64F, 0, 1, ksize=3)

    # Step 2: Calculate gradient magnitude (edge strength)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Step 3: Normalize gradient to [0, 1] range
    gradient_normalized = gradient_magnitude / (np.max(gradient_magnitude) + 1e-8)

    # Step 4: Normalize brightness to [0, 1] range
    brightness_normalized = face_region.astype(float) / 255.0

    # Step 5: Create depth map combining gradient (70%) and brightness (30%)
    # High gradient areas (features) protrude, flat areas recede
    depth_map = gradient_normalized * 0.7 + brightness_normalized * 0.3

    # Step 6: Generate point cloud from depth map
    points = []
    y_coords, x_coords = np.where(face_region > 50)  # Threshold for face region

    if len(y_coords) == 0:
        return None

    # Sample approximately 60% of valid face pixels for density control
    total_valid_pixels = len(y_coords)
    target_samples = int(total_valid_pixels * 0.6)

    # Ensure we have enough points
    if target_samples < num_points:
        # If not enough valid pixels, sample all with replacement
        indices = np.random.choice(total_valid_pixels, num_points, replace=True)
    else:
        # Sample without replacement for better distribution
        indices = np.random.choice(total_valid_pixels, min(num_points, target_samples), replace=False)

    for idx in indices:
        y_idx = y_coords[idx]
        x_idx = x_coords[idx]

        # Map pixel coordinates to 3D space
        px = (x_idx / 100.0) * 120 - 60  # Center and scale X
        py = (y_idx / 100.0) * 120 - 60  # Center and scale Y

        # CRITICAL: Set Z from depth map (range: -50 to +50)
        # depth_map is [0, 1], map to [-50, +50]
        pz = (depth_map[y_idx, x_idx] * 100) - 50

        points.append([px, py, pz])

    if len(points) == 0:
        return None

    # Resample to exact target count if needed
    points_array = np.array(points)
    if len(points_array) != num_points:
        indices = np.random.choice(len(points_array), num_points, replace=(len(points_array) < num_points))
        points_array = points_array[indices]

    return points_array


class ShapeLibrary:
    """
    Manages pre-computed shapes and cycling through them
    """

    def __init__(self, particle_count):
        self.particle_count = particle_count
        self.shapes = {}
        self.shape_order = [
            'dna_helix',
            'torus_knot',
            'double_helix_torus',  # HIGH ENHANCEMENT #4: Signature wow shape
            'face',  # Will be None until captured
            'lorenz_attractor',
            'stanford_bunny',
            'detailed_sphere',
            'cube'
        ]

        # Pre-generate all shapes (expensive, but only done once)
        print("Generating shapes...")
        self.shapes['dna_helix'] = generate_dna_helix(particle_count)
        self.shapes['torus_knot'] = generate_torus_knot(particle_count)
        self.shapes['double_helix_torus'] = generate_double_helix_torus(particle_count)  # New!
        self.shapes['lorenz_attractor'] = generate_lorenz_attractor(particle_count)
        self.shapes['stanford_bunny'] = load_stanford_bunny(particle_count)
        self.shapes['detailed_sphere'] = generate_detailed_sphere(particle_count)
        self.shapes['cube'] = generate_cube(particle_count)
        self.shapes['face'] = None  # Captured on demand
        print("Shapes generated successfully!")

    def get_shape(self, name):
        """
        Get pre-computed shape by name

        Args:
            name: Shape name from shape_order

        Returns:
            NumPy array (N, 3) or None if not available
        """
        if name == 'face' and self.shapes['face'] is None:
            return None  # Skip if face not captured
        return self.shapes.get(name)

    def set_face(self, points):
        """
        Update the face shape with captured points

        Args:
            points: NumPy array (N, 3) of face point cloud
        """
        if points is not None:
            self.shapes['face'] = points
            print("Face captured successfully!")

    def get_shape_order(self):
        """Get list of shape names in cycling order"""
        return self.shape_order
