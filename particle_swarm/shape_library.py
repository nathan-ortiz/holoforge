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


def generate_cube(num_points=6000, size=100):
    """
    Cube with points on faces and edges
    """
    points_per_face = num_points // 6
    half_size = size / 2

    faces = []

    # Six faces of the cube
    for _ in range(points_per_face):
        # Front and back faces
        faces.append([np.random.uniform(-half_size, half_size),
                     np.random.uniform(-half_size, half_size),
                     half_size])
        faces.append([np.random.uniform(-half_size, half_size),
                     np.random.uniform(-half_size, half_size),
                     -half_size])

        # Left and right faces
        faces.append([half_size,
                     np.random.uniform(-half_size, half_size),
                     np.random.uniform(-half_size, half_size)])
        faces.append([-half_size,
                     np.random.uniform(-half_size, half_size),
                     np.random.uniform(-half_size, half_size)])

        # Top and bottom faces
        faces.append([np.random.uniform(-half_size, half_size),
                     half_size,
                     np.random.uniform(-half_size, half_size)])
        faces.append([np.random.uniform(-half_size, half_size),
                     -half_size,
                     np.random.uniform(-half_size, half_size)])

    return np.array(faces[:num_points])


def capture_face_pointcloud(camera_frame, num_points=6000):
    """
    Capture face from camera and convert to 3D point cloud
    Uses simple extrusion (no depth estimation for speed)

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

    # Create layered point cloud (simple extrusion)
    num_layers = 10
    points_per_layer = num_points // num_layers

    points = []
    for layer in range(num_layers):
        z_depth = np.linspace(50, -50, num_layers)[layer]

        # Sample face pixels in this layer
        y_coords, x_coords = np.where(face_region > 50)  # Threshold for face region
        if len(y_coords) == 0:
            continue

        # Random sample from face pixels
        sample_size = min(points_per_layer, len(y_coords))
        indices = np.random.choice(len(y_coords), sample_size, replace=False)

        for idx in indices:
            # Map pixel coordinates to 3D space
            px = (x_coords[idx] / 100.0) * 120 - 60  # Center and scale
            py = (y_coords[idx] / 100.0) * 120 - 60
            pz = z_depth

            points.append([px, py, pz])

    if len(points) == 0:
        return None

    return np.array(points)


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
