#!/bin/bash
# Run Neural Particle Swarm
# Quick launcher script for HoloForge

echo "========================================"
echo "  Neural Particle Swarm - HoloForge"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import py5" 2>/dev/null; then
    echo "Error: py5 not installed!"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

# Run the particle swarm
echo "Starting particle swarm..."
echo ""
cd particle_swarm
python main.py

# Deactivate on exit
deactivate
