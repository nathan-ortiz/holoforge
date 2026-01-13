#!/usr/bin/env python3
"""
Performance test script for Neural Particle Swarm
Tests physics simulation performance without rendering
"""

import sys
import time
import numpy as np

# Add particle_swarm to path
sys.path.insert(0, 'particle_swarm')

from particle_system import ParticleSystem
from shape_library import ShapeLibrary
from transition_manager import TransitionManager
from config import PARTICLE_COUNT


def test_particle_update_no_forces():
    """Test basic particle update with no external forces"""
    print("Testing particle update (no forces)...")

    ps = ParticleSystem(PARTICLE_COUNT)
    dt = 1.0 / 60.0  # 60 FPS target

    # Run 100 updates and measure time
    start = time.time()
    for _ in range(100):
        ps.update(dt, None)
    elapsed = time.time() - start

    avg_time_ms = (elapsed / 100) * 1000
    print(f"  Average update time: {avg_time_ms:.2f}ms")
    print(f"  Est. FPS: {1000/avg_time_ms:.1f}")

    return avg_time_ms


def test_particle_update_with_hand_force():
    """Test particle update with hand force applied"""
    print("\nTesting particle update (with hand force)...")

    ps = ParticleSystem(PARTICLE_COUNT)
    dt = 1.0 / 60.0

    # Simulate hand force in center of particle cloud
    hand_force = {
        'position': (0, 0, 0),
        'gesture': 'scatter',
        'strength': 8.0
    }

    # Run 100 updates and measure time
    start = time.time()
    for _ in range(100):
        ps.update(dt, hand_force)
    elapsed = time.time() - start

    avg_time_ms = (elapsed / 100) * 1000
    print(f"  Average update time: {avg_time_ms:.2f}ms")
    print(f"  Est. FPS: {1000/avg_time_ms:.1f}")

    return avg_time_ms


def test_color_update():
    """Test color update performance"""
    print("\nTesting color update...")

    ps = ParticleSystem(PARTICLE_COUNT)

    # Run 1000 color updates and measure time
    start = time.time()
    for _ in range(1000):
        ps.update_colors()
    elapsed = time.time() - start

    avg_time_ms = (elapsed / 1000) * 1000
    print(f"  Average update time: {avg_time_ms:.3f}ms")

    return avg_time_ms


def test_transition_calculations():
    """Test transition manager performance"""
    print("\nTesting transition calculations...")

    shape_library = ShapeLibrary(PARTICLE_COUNT)
    tm = TransitionManager(shape_library)

    # Run 100 transition updates and measure time
    start = time.time()
    current_time = 0.0
    for i in range(100):
        current_time += 1.0 / 60.0
        tm.update(current_time)
    elapsed = time.time() - start

    avg_time_ms = (elapsed / 100) * 1000
    print(f"  Average update time: {avg_time_ms:.2f}ms")

    return avg_time_ms


def test_full_frame_simulation():
    """Test complete frame simulation (600 frames = 10 seconds @ 60 FPS)"""
    print("\nTesting full frame simulation (600 frames)...")

    # Initialize all systems
    shape_library = ShapeLibrary(PARTICLE_COUNT)
    ps = ParticleSystem(PARTICLE_COUNT)
    tm = TransitionManager(shape_library)

    dt = 1.0 / 60.0
    frame_times = []

    # Run 600 frames
    for frame in range(600):
        frame_start = time.time()

        # Update transition
        current_time = frame * dt
        targets = tm.update(current_time)
        if targets is not None:
            ps.set_targets(targets)

        # Update physics
        ps.update(dt, None)

        # Update colors
        ps.update_colors()

        frame_time = time.time() - frame_start
        frame_times.append(frame_time)

        # Progress indicator every 100 frames
        if (frame + 1) % 100 == 0:
            print(f"  Progress: {frame + 1}/600 frames")

    # Calculate statistics
    avg_time = np.mean(frame_times) * 1000
    min_time = np.min(frame_times) * 1000
    max_time = np.max(frame_times) * 1000
    std_time = np.std(frame_times) * 1000

    print(f"\n  Average frame time: {avg_time:.2f}ms")
    print(f"  Min frame time: {min_time:.2f}ms")
    print(f"  Max frame time: {max_time:.2f}ms")
    print(f"  Std deviation: {std_time:.2f}ms")
    print(f"  Est. FPS: {1000/avg_time:.1f}")

    return avg_time


def identify_bottlenecks(update_no_force, update_with_force, color_update, transition, full_frame):
    """Analyze which components are bottlenecks"""
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)

    components = {
        'Particle Update (no force)': update_no_force,
        'Particle Update (with force)': update_with_force,
        'Color Update': color_update,
        'Transition Manager': transition,
    }

    # Sort by time
    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)

    print("\nComponents by time (slowest first):")
    for name, time_ms in sorted_components:
        percentage = (time_ms / full_frame) * 100
        print(f"  {name:<30} {time_ms:6.2f}ms  ({percentage:5.1f}% of frame)")

    print(f"\n  {'FULL FRAME':<30} {full_frame:6.2f}ms  (100.0%)")


def main():
    """Run all performance tests"""
    print("=" * 70)
    print("NEURAL PARTICLE SWARM - PERFORMANCE TEST")
    print("=" * 70)
    print(f"\nParticle count: {PARTICLE_COUNT}")
    print(f"Target: 60 FPS (16.67ms per frame)")
    print("=" * 70)

    # Run individual component tests
    update_no_force = test_particle_update_no_forces()
    update_with_force = test_particle_update_with_hand_force()
    color_update = test_color_update()
    transition = test_transition_calculations()

    # Run full simulation
    full_frame = test_full_frame_simulation()

    # Analyze bottlenecks
    identify_bottlenecks(update_no_force, update_with_force, color_update, transition, full_frame)

    # Final assessment
    print("\n" + "=" * 70)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 70)

    target_ms = 16.67  # 60 FPS
    if full_frame < target_ms:
        margin = target_ms - full_frame
        print(f"✓ PASS: 60 FPS achievable!")
        print(f"  Frame time: {full_frame:.2f}ms (target: {target_ms:.2f}ms)")
        print(f"  Headroom: {margin:.2f}ms ({(margin/target_ms)*100:.1f}%)")
        print(f"\n  This leaves time for:")
        print(f"  - Rendering: ~{margin*0.7:.1f}ms")
        print(f"  - OS overhead: ~{margin*0.3:.1f}ms")
    else:
        deficit = full_frame - target_ms
        print(f"✗ FAIL: 60 FPS NOT achievable")
        print(f"  Frame time: {full_frame:.2f}ms (target: {target_ms:.2f}ms)")
        print(f"  Over budget: {deficit:.2f}ms ({(deficit/target_ms)*100:.1f}%)")
        print(f"  Estimated FPS: {1000/full_frame:.1f}")
        print(f"\n  Recommendations:")
        print(f"  1. Reduce PARTICLE_COUNT in config.py")
        print(f"  2. Increase PARTICLE_DAMPING for faster settling")
        print(f"  3. Review bottlenecks above")

    print("=" * 70 + "\n")

    return 0 if full_frame < target_ms else 1


if __name__ == '__main__':
    sys.exit(main())
