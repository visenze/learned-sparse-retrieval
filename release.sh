#!/bin/bash
# Release script for LSR package
# Usage: ./release.sh <version>
# Example: ./release.sh 1.0.0

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.0.0"
    exit 1
fi

# Validate version format (semantic versioning)
if ! [[ $VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must be in format X.Y.Z (e.g., 1.0.0)"
    exit 1
fi

echo "Creating release version $VERSION"
echo "=================================="

# Update version in setup.py
echo "Updating version in setup.py..."
sed -i "s/VERSION = \".*\"/VERSION = \"$VERSION\"/" setup.py

# Update version in pyproject.toml
echo "Updating version in pyproject.toml..."
sed -i "s/version = \".*\"/version = \"$VERSION\"/" pyproject.toml

# Show changes
echo ""
echo "Version updated in:"
grep "VERSION = " setup.py
grep "version = " pyproject.toml

# Ask for confirmation
echo ""
read -p "Commit and create tag? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. You can manually commit and tag later."
    exit 0
fi

# Git operations
echo ""
echo "Git operations..."
git add setup.py pyproject.toml
git commit -m "Bump version to $VERSION"
git tag -a "v$VERSION" -m "Release version $VERSION"

echo ""
echo "✓ Created commit and tag v$VERSION"
echo ""
echo "To push the release:"
echo "  git push origin main"
echo "  git push origin v$VERSION"
echo ""
echo "Or push both at once:"
echo "  git push origin main --tags"
