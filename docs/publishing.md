# Publishing to PyPI - Setup Guide

This guide explains how to set up your repository to automatically publish packages to PyPI using GitHub Actions.

## Overview

The repository includes two CI/CD workflows:

1. **CI Workflow** (`.github/workflows/ci.yml`): Runs tests, linting, and type checking on every push and pull request
2. **Publish Workflow** (`.github/workflows/publish.yml`): Builds and publishes the package to PyPI when a release is created

## How to Publish a Release

### 1. Update Version Number

First, update the version in `pyproject.toml`:

```toml
[project]
version = "0.1.1"  # Update this
```

### 2. Commit and Push Changes

```bash
git add pyproject.toml
git commit -m "Bump version to 0.1.1"
git push origin main
```

### 3. Create a Git Tag

```bash
git tag v0.1.1
git push origin v0.1.1
```

### 4. Create a GitHub Release

#### Via GitHub Web UI:

1. Go to your repository on GitHub
2. Click on "Releases" (right sidebar)
3. Click "Draft a new release"
4. Click "Choose a tag" and select `v0.1.1` (or the tag you created)
5. Set the release title (e.g., "v0.1.1")
6. Add release notes describing changes
7. Click "Publish release"

#### Via GitHub CLI:

```bash
gh release create v0.1.1 --title "v0.1.1" --notes "Release notes here"
```

### 5. Monitor the Workflow

1. Go to the "Actions" tab in your repository
2. Watch the "Publish to PyPI" workflow run
3. Once complete, your package will be available on PyPI at: https://pypi.org/project/make-mlops-easy/

## Verification

After publishing, verify your package:

```bash
# Install from PyPI
pip install make-mlops-easy

# Test it works
make-mlops-easy --version
```
