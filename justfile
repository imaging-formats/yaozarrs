# https://github.com/casey/just

# Default recipe
default:
    @just --list

# Run full coverage tests across Python versions and dependency resolutions
test-cov-full:
    #!/usr/bin/env bash
    set -euo pipefail

    echo "🧪 Running comprehensive test coverage across Python versions and dependency resolutions..."

    # Clean up any existing coverage data
    rm -f .coverage .coverage.*
    rm -rf htmlcov/

    echo ""
    echo "📊 Step 1: Running tests with Python 3.10 + lowest dependencies..."
    uv run -p 3.10 --resolution lowest-direct --exact -U --no-dev --group test-full \
        coverage run --parallel-mode --source=src --omit="*/tests/*" -m pytest -v

    sleep 1  

    echo ""
    echo "📊 Step 2: Running tests with Python 3.13 + highest dependencies..."
    uv run -p 3.13 --resolution highest --exact -U --no-dev --group test-full \
        coverage run --parallel-mode --source=src --omit="*/tests/*" -m pytest -v

    uv run coverage combine

    echo ""
    echo "=================================="
    echo "📈 COMBINED COVERAGE REPORT"
    echo "=================================="
    uv run coverage report --show-missing --skip-covered

# Clean coverage artifacts
clean-cov:
    rm -f .coverage .coverage.*
    rm -rf htmlcov/

# Clean build artifacts
clean: clean-cov
    rm -rf dist/
    rm -rf build/
    rm -rf *.egg-info/
    find . -type d -name __pycache__ -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete
