#!/usr/bin/env python3
"""
Test script for Stanford Bunny OBJ parsing
Validates OBJ file loading and point cloud generation
"""

import sys
import os

# Add particle_swarm to path
sys.path.insert(0, 'particle_swarm')

from shape_library import load_stanford_bunny, parse_obj_file
import numpy as np


def test_obj_file_presence():
    """Check if OBJ file exists"""
    print("=" * 70)
    print("TEST 1: OBJ File Presence")
    print("=" * 70)

    obj_paths = [
        'stanford_bunny.obj',
        'stanford-bunny.obj',
    ]

    found = False
    for path in obj_paths:
        if os.path.exists(path):
            print(f"✓ Found: {path}")
            file_size = os.path.getsize(path)
            print(f"  File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")
            found = True

            # Count lines
            with open(path, 'r') as f:
                lines = f.readlines()
                vertex_lines = [l for l in lines if l.strip().startswith('v ')]
                print(f"  Total lines: {len(lines):,}")
                print(f"  Vertex lines: {len(vertex_lines):,}")

    if not found:
        print("✗ No OBJ file found")
        print("\nExpected locations:")
        for path in obj_paths:
            print(f"  - {os.path.abspath(path)}")

    print()
    return found


def test_obj_parsing():
    """Test OBJ file parsing"""
    print("=" * 70)
    print("TEST 2: OBJ File Parsing")
    print("=" * 70)

    obj_paths = [
        'stanford_bunny.obj',
        'stanford-bunny.obj',
    ]

    for path in obj_paths:
        if os.path.exists(path):
            try:
                print(f"Parsing {path}...")
                vertices = parse_obj_file(path)

                print(f"✓ Successfully parsed")
                print(f"  Vertices extracted: {len(vertices):,}")

                if len(vertices) > 0:
                    # Show coordinate ranges
                    min_coords = np.min(vertices, axis=0)
                    max_coords = np.max(vertices, axis=0)
                    mean_coords = np.mean(vertices, axis=0)

                    print(f"  X range: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]")
                    print(f"  Y range: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
                    print(f"  Z range: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
                    print(f"  Center: ({mean_coords[0]:.3f}, {mean_coords[1]:.3f}, {mean_coords[2]:.3f})")

                    # Show sample vertices
                    print(f"\n  First 3 vertices:")
                    for i in range(min(3, len(vertices))):
                        v = vertices[i]
                        print(f"    v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

                return True

            except Exception as e:
                print(f"✗ Parsing failed: {e}")
                import traceback
                traceback.print_exc()
                return False

    print("✗ No OBJ file found to parse")
    print()
    return False


def test_bunny_loading():
    """Test complete bunny loading with resampling"""
    print("=" * 70)
    print("TEST 3: Stanford Bunny Loading (Complete)")
    print("=" * 70)

    test_counts = [100, 1000, 6000]

    for count in test_counts:
        print(f"\nLoading bunny with {count} particles...")
        try:
            bunny = load_stanford_bunny(count)

            print(f"✓ Success")
            print(f"  Shape: {bunny.shape}")
            print(f"  Point count: {len(bunny)} (expected: {count})")

            # Check coordinate ranges
            min_coords = np.min(bunny, axis=0)
            max_coords = np.max(bunny, axis=0)

            print(f"  X range: [{min_coords[0]:.1f}, {max_coords[0]:.1f}]")
            print(f"  Y range: [{min_coords[1]:.1f}, {max_coords[1]:.1f}]")
            print(f"  Z range: [{min_coords[2]:.1f}, {max_coords[2]:.1f}]")

            # Verify point count matches
            if len(bunny) == count:
                print(f"  ✓ Point count matches target")
            else:
                print(f"  ✗ Point count mismatch: got {len(bunny)}, expected {count}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()

    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("STANFORD BUNNY OBJ PARSING - TEST SUITE")
    print("=" * 70)
    print()

    # Run tests
    file_exists = test_obj_file_presence()
    if file_exists:
        parsing_works = test_obj_parsing()
        test_bunny_loading()
    else:
        print("\n⚠️  IMPORTANT:")
        print("Place stanford_bunny.obj or stanford-bunny.obj in the project root")
        print("The system will automatically use it when available")
        print("\nCurrent fallback: Ellipsoid approximation")

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
