#!/bin/bash

echo "🧹 Scanning for Python cache files..."

# Find and delete __pycache__ directories
find . -type d -name "__pycache__" -prune -exec rm -rf {} +

# Find and delete stray compiled python files
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

echo "✨ Cleanup complete! All __pycache__ and compiled files removed."