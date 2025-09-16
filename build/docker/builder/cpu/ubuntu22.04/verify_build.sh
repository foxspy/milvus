#!/bin/bash

# Docker build verification script for GCC 15 upgrade

set -e

echo "=== Milvus GCC 15 Docker Build Verification ==="
echo

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Check Docker daemon
if ! docker version &> /dev/null; then
    echo "❌ Cannot connect to Docker daemon. Please check:"
    echo "  1. Docker daemon is running"
    echo "  2. Current user is in docker group: sudo usermod -aG docker $USER"
    echo "  3. Log out and back in after adding to docker group"
    exit 1
fi

echo "✅ Docker is available"

# Check required files
DOCKERFILE="build/docker/builder/cpu/ubuntu22.04/Dockerfile"
CONAN_PROFILE="build/docker/builder/cpu/ubuntu22.04/conanprofile.txt"

if [[ ! -f "$DOCKERFILE" ]]; then
    echo "❌ Dockerfile not found: $DOCKERFILE"
    exit 1
fi

if [[ ! -f "$CONAN_PROFILE" ]]; then
    echo "❌ Conan profile not found: $CONAN_PROFILE"
    exit 1
fi

echo "✅ Required files exist"

# Verify GCC download URL
echo "🔍 Verifying GCC 15.1.0 download URL..."
if curl -sI https://ftp.gnu.org/gnu/gcc/gcc-15.1.0/gcc-15.1.0.tar.xz | grep -q "200 OK"; then
    echo "✅ GCC 15.1.0 download URL is accessible"
else
    echo "❌ GCC 15.1.0 download URL is not accessible"
    exit 1
fi

# Verify CMake download URL
echo "🔍 Verifying CMake 3.32.0 download URL..."
ARCH=$(uname -m)
if curl -sI "https://cmake.org/files/v3.32/cmake-3.32.0-linux-${ARCH}.tar.gz" | grep -q "200 OK"; then
    echo "✅ CMake 3.32.0 download URL is accessible"
else
    echo "❌ CMake 3.32.0 download URL is not accessible"
    echo "ℹ️  Available architecture: $ARCH"
fi

# Start Docker build
echo
echo "🚀 Starting Docker build..."
echo "This may take 30-60 minutes due to GCC compilation..."
echo

docker build -t milvus-builder-gcc15 build/docker/builder/cpu/ubuntu22.04/

if [[ $? -eq 0 ]]; then
    echo
    echo "✅ Docker build completed successfully!"
    echo
    echo "🔍 Verifying GCC version in container..."
    docker run --rm milvus-builder-gcc15 gcc --version
    echo
    echo "🔍 Verifying Conan configuration..."
    docker run --rm milvus-builder-gcc15 conan profile show default
else
    echo
    echo "❌ Docker build failed"
    exit 1
fi

echo
echo "🎉 GCC 15 upgrade verification completed successfully!"