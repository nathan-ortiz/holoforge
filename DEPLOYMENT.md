# Neural Particle Swarm - Deployment Checklist

## Pre-Deployment Tests

Complete all items before deploying to Raspberry Pi 5.

### Installation Validation
- [ ] `python test_installation.py` passes all tests
- [ ] All core modules import successfully
- [ ] NumPy version is 1.26.4 (confirmed compatible)
- [ ] py5 version is >= 0.8.0
- [ ] MediaPipe imports without errors
- [ ] OpenCV imports without errors

### Performance Validation
- [ ] `python test_performance.py` shows <16ms frame time
- [ ] Full frame simulation achieves 60 FPS target
- [ ] No performance bottlenecks exceed 5ms
- [ ] Color update is fully vectorized (no Python loops)
- [ ] Hand force calculation has early culling
- [ ] Memory usage stable over 600-frame test

### Code Quality
- [ ] py5 class mode implemented correctly (inherits from py5.Sketch)
- [ ] All py5 calls use `self.*` instead of `py5.*`
- [ ] Hand position mapping uses particle coordinate space (±150)
- [ ] Boid separation is DISABLED (ENABLE_BOID_SEPARATION = False)
- [ ] VERBOSE_LOGGING is set appropriately (False for production)

### Functionality Tests
- [ ] MockCamera mode runs without errors (for development)
- [ ] All shapes load and render correctly
  - [ ] DNA helix
  - [ ] Torus knot
  - [ ] Lorenz attractor
  - [ ] Stanford bunny
  - [ ] Detailed sphere
  - [ ] Cube
  - [ ] Face capture (skipped if camera unavailable)
- [ ] Transitions are smooth (no jumps or glitches)
- [ ] Auto-cycling works for 5+ complete cycles
- [ ] Shape dissolve phase looks natural
- [ ] Swirl phase has proper rotation
- [ ] Formation phase uses ease-out interpolation

### Configuration
- [ ] PARTICLE_COUNT set for target hardware (6000 for Pi 5)
- [ ] ENABLE_BOID_SEPARATION = False
- [ ] DISPLAY_WIDTH = 720, DISPLAY_HEIGHT = 720
- [ ] FRAME_RATE = 60
- [ ] AUTO_ROTATE_SPEED set appropriately (0.1 rad/s)
- [ ] Color gradients configured (cyan-white-magenta)

### Raspberry Pi Specific
- [ ] Shell script has DISPLAY=:0.0 export
- [ ] Camera initialization handles common errors gracefully
- [ ] PermissionError provides helpful sudo command
- [ ] FileNotFoundError suggests raspi-config steps
- [ ] All error messages point to solutions

### Keyboard Controls
- [ ] F key toggles FPS display
- [ ] S key skips to next shape
- [ ] Spacebar freezes/unfreezes cycling
- [ ] Q key quits cleanly

### Gesture Recognition (if camera available)
- [ ] Open palm (scatter) detected reliably
- [ ] Pinch (attract) detected reliably
- [ ] Fist (freeze toggle) detected reliably
- [ ] Peace sign (skip) detected reliably
- [ ] Thumbs up (capture face) detected reliably
- [ ] Gesture hold time prevents false triggers (0.3s)
- [ ] Skip cooldown prevents spam (2.0s)

### Visual Quality
- [ ] Particles form recognizable shapes
- [ ] Color gradient transitions smoothly with depth
- [ ] Auto-rotation is smooth and gentle
- [ ] No flickering or stuttering
- [ ] Particle size is visible but not overwhelming
- [ ] Background is black (0, 0, 0)

### Stability
- [ ] No crashes during 10-minute run
- [ ] No memory leaks (check with `htop` or similar)
- [ ] Frame rate remains stable over time
- [ ] Transition state machine handles all states correctly
- [ ] No exceptions in console during normal operation

### Error Handling
- [ ] Camera failure gracefully falls back to MockCamera
- [ ] Missing face returns None and logs appropriate message
- [ ] Hand tracking failure doesn't crash application
- [ ] Particle count mismatch triggers resampling warning
- [ ] Invalid targets handled gracefully

### Documentation
- [ ] README.md has accurate installation instructions
- [ ] README.md lists all keyboard controls
- [ ] README.md lists all hand gestures
- [ ] README.md has troubleshooting section
- [ ] Comments explain all performance-critical sections

---

## Deployment Steps for Raspberry Pi 5

Once all checklist items are complete:

### 1. Prepare Repository
```bash
# Ensure all changes are committed
git status
git add -A
git commit -m "Production-ready particle swarm"
git push origin <branch>
```

### 2. Clone on Raspberry Pi
```bash
ssh pi@raspberrypi.local
cd ~
git clone https://github.com/yourusername/holoforge.git
cd holoforge
```

### 3. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Tests
```bash
# Test installation
python test_installation.py

# Test performance
python test_performance.py
```

### 5. Configure System
```bash
# Enable camera
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable
# Reboot: sudo reboot

# Add user to video group
sudo usermod -aG video $USER
# Logout and login again
```

### 6. First Run
```bash
# Run with launcher script
./run_particle_swarm.sh

# Or run directly
cd particle_swarm
python main.py
```

### 7. Verify on HyperPixel
- [ ] Display shows particle swarm at 720x720
- [ ] FPS counter shows 50-60 FPS
- [ ] Auto-cycling through shapes works
- [ ] Keyboard controls respond
- [ ] No error messages in console

### 8. Test with Camera
- [ ] Camera initializes successfully
- [ ] Hand tracking works in good lighting
- [ ] Gestures trigger appropriate actions
- [ ] Scatter/attract forces visible
- [ ] Face capture creates recognizable shape

---

## Performance Targets

| Metric | Target | Acceptable | Action if Below |
|--------|--------|------------|-----------------|
| Frame Rate | 60 FPS | 55+ FPS | Reduce particle count |
| Frame Time | <16.67ms | <18ms | Profile bottlenecks |
| Particle Update | <8ms | <10ms | Check vectorization |
| Color Update | <1ms | <2ms | Verify no Python loops |
| Transition Calc | <2ms | <3ms | Optimize interpolation |
| Gesture Latency | <100ms | <150ms | Check MediaPipe config |

---

## Troubleshooting Deployment Issues

### Low FPS on Pi 5
1. Check `ENABLE_BOID_SEPARATION = False`
2. Reduce `PARTICLE_COUNT` to 5000 or 4000
3. Increase `PARTICLE_DAMPING` to 0.95
4. Run `test_performance.py` to identify bottleneck

### Display Not Showing
1. Check `export DISPLAY=:0.0` in shell script
2. Verify HyperPixel drivers installed: `ls /dev/fb*`
3. Test with simple: `DISPLAY=:0.0 xeyes`

### Camera Not Working
1. Check cable connection
2. Run: `vcgencmd get_camera`
3. Enable in `raspi-config`
4. Add to video group: `sudo usermod -aG video $USER`

### Hand Tracking Unreliable
1. Improve lighting (bright, even light)
2. Reduce `HAND_MIN_DETECTION_CONFIDENCE` to 0.3
3. Reduce `HAND_MIN_TRACKING_CONFIDENCE` to 0.3
4. Check MediaPipe version: `pip show mediapipe`

---

## Sign-Off

Deployment ready when all checklist items are complete and verified.

**Tested by:** _________________
**Date:** _________________
**Raspberry Pi Model:** _________________
**Performance:** _______ FPS average

**Notes:**
