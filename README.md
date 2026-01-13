# Neural Particle Swarm - HoloForge

A visually stunning particle system for Raspberry Pi 5 + HyperPixel 4.0 Square display that forms recognizable 3D shapes from thousands of particles. Features real-time hand gesture control via Raspberry Pi Camera Module 3.

![HoloForge Banner](https://via.placeholder.com/800x200/0a0a0a/00ffff?text=Neural+Particle+Swarm)

## Features

- **5,000-8,000 particles** forming complex 3D shapes
- **60 FPS** target performance on Raspberry Pi 5
- **Hand gesture control** via MediaPipe
- **6+ pre-computed shapes**: DNA helix, torus knot, Lorenz attractor, Stanford bunny, sphere, cube
- **Face capture**: Capture and display your face as particles
- **Smooth transitions** with dissolve/swirl/form phases
- **Depth-based coloring** with cyan-white-magenta gradients

## Hardware Requirements

- **Raspberry Pi 5** (4GB or 8GB RAM)
- **HyperPixel 4.0 Square** (720x720 display)
- **Raspberry Pi Camera Module 3**
- **Beam splitter cube** (for Pepper's Ghost effect)

## Software Requirements

- Python 3.11+
- py5 (Python Processing)
- MediaPipe
- OpenCV
- NumPy 1.26.4
- Picamera2

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/holoforge.git
cd holoforge
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Display (HyperPixel 4.0)

Follow HyperPixel setup instructions for Raspberry Pi OS.

## Usage

### Run the Particle Swarm

```bash
cd particle_swarm
python main.py
```

### Keyboard Controls

- **F** - Toggle FPS display
- **S** - Skip to next shape
- **Spacebar** - Freeze/unfreeze shape cycling
- **Q** - Quit application

### Hand Gestures

- **Open Palm** (5 fingers extended) - Scatter particles away from hand
- **Pinch** (thumb + index together) - Attract particles to hand
- **Fist** (closed hand) - Freeze/unfreeze automatic shape cycling
- **Peace Sign** (index + middle extended) - Skip to next shape
- **Thumbs Up** - Capture face from camera as particle shape

## Configuration

All tunable parameters are in `particle_swarm/config.py`:

```python
# Particle System
PARTICLE_COUNT = 6000          # Number of particles
PARTICLE_SIZE = 3              # Render size
PARTICLE_DAMPING = 0.92        # Velocity damping
MAX_VELOCITY = 15.0            # Speed limit

# Physics
SPRING_STRENGTH = 0.15         # Target attraction force
HAND_FORCE_STRENGTH = 8.0      # Gesture interaction strength
HAND_FORCE_RADIUS = 150.0      # Gesture influence radius

# Shape Cycling
SHAPE_HOLD_TIME = 8.0          # Seconds to hold each shape
DISSOLVE_TIME = 1.0            # Dissolve phase duration
SWIRL_TIME = 0.5               # Swirl phase duration
FORM_TIME = 1.5                # Formation phase duration

# Colors (depth-based gradient)
COLOR_NEAR = (0, 255, 255)     # Cyan
COLOR_MID = (255, 255, 255)    # White
COLOR_FAR = (255, 0, 102)      # Magenta
```

## Performance Optimization

If FPS drops below 60:

1. **Reduce particle count** in config.py
2. **Disable boid separation** (`ENABLE_BOID_SEPARATION = False`)
3. **Increase damping** for faster settling
4. **Reduce particle size** for less rendering overhead
5. **Close other applications** to free up resources

## Architecture

```
particle_swarm/
├── main.py                 # Entry point, py5 sketch
├── particle_system.py      # Vectorized particle physics
├── shape_library.py        # 3D shape generators
├── gesture_recognizer.py   # Hand gesture interpretation
├── transition_manager.py   # Shape cycling state machine
├── camera_capture.py       # Camera and hand tracking
└── config.py              # All tunable parameters
```

## Shapes Library

### Pre-computed Shapes

1. **DNA Double Helix** - Two intertwined helices with rungs
2. **Torus Knot** - Mathematical trefoil knot (p=3, q=2)
3. **Lorenz Attractor** - Chaotic butterfly attractor
4. **Stanford Bunny** - Iconic 3D model (or ellipsoid approximation)
5. **Detailed Sphere** - Fibonacci-distributed sphere with noise
6. **Cube** - Geometric cube with surface points

### Face Capture

Capture your face in real-time using the **thumbs up** gesture. The system uses:
- OpenCV Haar Cascade for face detection
- Simple depth extrusion for 3D effect
- 10 layers for volumetric appearance

## Troubleshooting

### Camera Not Detected

```
Warning: Camera initialization failed
Running in demo mode without camera
```

**Solution**: Check camera cable connection and enable camera in `raspi-config`.

### Low FPS

**Symptoms**: FPS counter shows < 60
**Solutions**:
- Reduce `PARTICLE_COUNT` in config.py
- Disable `ENABLE_BOID_SEPARATION`
- Close background applications

### Hand Tracking Not Working

**Solution**: Ensure good lighting and hand visibility. MediaPipe requires clear view of hand.

### Import Errors

```
ModuleNotFoundError: No module named 'py5'
```

**Solution**: Activate virtual environment and reinstall dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Development

### Adding New Shapes

Create a new shape generator in `shape_library.py`:

```python
def generate_my_shape(num_points=6000):
    """
    Your custom shape description
    """
    # Generate points as NumPy array (N, 3)
    points = np.array([...])
    return points
```

Add to `ShapeLibrary.__init__()`:

```python
self.shapes['my_shape'] = generate_my_shape(particle_count)
self.shape_order.append('my_shape')
```

### Tuning Physics

Experiment with config.py parameters:
- Higher `SPRING_STRENGTH` = faster convergence
- Higher `PARTICLE_DAMPING` = less oscillation
- Higher `MAX_VELOCITY` = more energetic motion

## Credits

- **py5** - Python Processing by [@hx2A](https://github.com/hx2A)
- **MediaPipe** - Hand tracking by Google
- **HyperPixel 4.0** - Display by Pimoroni
- **Stanford Bunny** - Stanford Computer Graphics Laboratory

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue or pull request.

## Links

- [Project Repository](https://github.com/yourusername/holoforge)
- [py5 Documentation](https://py5.ixora.io/)
- [MediaPipe Docs](https://developers.google.com/mediapipe)

---

**Made with ❤️ for the Raspberry Pi community**
