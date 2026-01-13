#!/usr/bin/env python3
"""
Test script to verify Neural Particle Swarm installation
Checks all dependencies and core functionality
"""

import sys
import time


def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    results = []

    # Test NumPy
    try:
        import numpy as np
        version = np.__version__
        results.append(("NumPy", True, version))
    except ImportError as e:
        results.append(("NumPy", False, str(e)))

    # Test py5
    try:
        import py5
        version = py5.__version__ if hasattr(py5, '__version__') else "unknown"
        results.append(("py5", True, version))
    except ImportError as e:
        results.append(("py5", False, str(e)))

    # Test MediaPipe
    try:
        import mediapipe as mp
        version = mp.__version__ if hasattr(mp, '__version__') else "unknown"
        results.append(("MediaPipe", True, version))
    except ImportError as e:
        results.append(("MediaPipe", False, str(e)))

    # Test OpenCV
    try:
        import cv2
        version = cv2.__version__
        results.append(("OpenCV", True, version))
    except ImportError as e:
        results.append(("OpenCV", False, str(e)))

    # Test Picamera2 (optional on non-Pi systems)
    try:
        from picamera2 import Picamera2
        results.append(("Picamera2", True, "installed"))
    except ImportError:
        results.append(("Picamera2", False, "Not available (OK if not on Pi)"))

    # Test SciPy (optional)
    try:
        import scipy
        version = scipy.__version__
        results.append(("SciPy", True, version))
    except ImportError:
        results.append(("SciPy", False, "Optional"))

    return results


def test_particle_swarm_modules():
    """Test particle swarm modules"""
    print("\nTesting particle swarm modules...")
    results = []

    try:
        from particle_swarm import config
        results.append(("config.py", True, "OK"))
    except ImportError as e:
        results.append(("config.py", False, str(e)))

    try:
        from particle_swarm.particle_system import ParticleSystem
        results.append(("particle_system.py", True, "OK"))
    except ImportError as e:
        results.append(("particle_system.py", False, str(e)))

    try:
        from particle_swarm.shape_library import ShapeLibrary
        results.append(("shape_library.py", True, "OK"))
    except ImportError as e:
        results.append(("shape_library.py", False, str(e)))

    try:
        from particle_swarm.gesture_recognizer import GestureRecognizer
        results.append(("gesture_recognizer.py", True, "OK"))
    except ImportError as e:
        results.append(("gesture_recognizer.py", False, str(e)))

    try:
        from particle_swarm.transition_manager import TransitionManager
        results.append(("transition_manager.py", True, "OK"))
    except ImportError as e:
        results.append(("transition_manager.py", False, str(e)))

    try:
        from particle_swarm.camera_capture import CameraCapture
        results.append(("camera_capture.py", True, "OK"))
    except ImportError as e:
        results.append(("camera_capture.py", False, str(e)))

    return results


def test_basic_functionality():
    """Test basic particle system functionality"""
    print("\nTesting basic functionality...")
    results = []

    try:
        import numpy as np
        from particle_swarm.particle_system import ParticleSystem

        # Create particle system
        ps = ParticleSystem(100)
        results.append(("Create ParticleSystem", True, "100 particles"))

        # Test update
        ps.update(0.016, None)
        results.append(("Update physics", True, "OK"))

        # Test color update
        ps.update_colors()
        results.append(("Update colors", True, "OK"))

    except Exception as e:
        results.append(("Particle System", False, str(e)))

    try:
        from particle_swarm.shape_library import generate_dna_helix, generate_torus_knot

        # Test shape generation
        dna = generate_dna_helix(100)
        results.append(("Generate DNA helix", True, f"Shape: {dna.shape}"))

        torus = generate_torus_knot(100)
        results.append(("Generate torus knot", True, f"Shape: {torus.shape}"))

    except Exception as e:
        results.append(("Shape Library", False, str(e)))

    return results


def print_results(title, results):
    """Print test results in a formatted table"""
    print(f"\n{title}")
    print("=" * 70)

    all_passed = True
    for name, passed, info in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"

        print(f"{color}{status}{reset}  {name:<30} {info}")

        if not passed:
            all_passed = False

    return all_passed


def main():
    """Run all tests"""
    print("=" * 70)
    print("NEURAL PARTICLE SWARM - INSTALLATION TEST")
    print("=" * 70)

    # Test imports
    import_results = test_imports()
    imports_ok = print_results("Dependencies", import_results)

    # Test particle swarm modules
    module_results = test_particle_swarm_modules()
    modules_ok = print_results("Particle Swarm Modules", module_results)

    # Test basic functionality
    func_results = test_basic_functionality()
    func_ok = print_results("Basic Functionality", func_results)

    # Summary
    print("\n" + "=" * 70)
    if imports_ok and modules_ok and func_ok:
        print("✓ ALL TESTS PASSED - Installation successful!")
        print("\nYou can now run: ./run_particle_swarm.sh")
    else:
        print("✗ SOME TESTS FAILED - Please check errors above")
        print("\nInstall missing dependencies with: pip install -r requirements.txt")

    print("=" * 70 + "\n")

    return 0 if (imports_ok and modules_ok and func_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
