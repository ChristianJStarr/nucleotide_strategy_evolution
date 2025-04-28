#!/bin/bash
# Bash script to build the documentation

echo "Building Nucleotide Strategy Evolution documentation..."

# Navigate to the docs directory
pushd docs > /dev/null

# Clean previous build
echo "Cleaning previous build..."
make clean

# Generate API documentation
echo "Generating API documentation..."
python generate_api_docs.py

# Build HTML documentation
echo "Building HTML documentation..."
make html

if [ $? -eq 0 ]; then
    echo "Documentation built successfully!"
    echo "Open docs/_build/html/index.html in your browser to view the docs."
else
    echo "Error building documentation"
fi

# Return to the original directory
popd > /dev/null 