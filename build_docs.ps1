# PowerShell script to build the documentation
Write-Host "Building Nucleotide Strategy Evolution documentation..."

# Navigate to the docs directory
Push-Location -Path "docs"

try {
    # Clean previous build
    Write-Host "Cleaning previous build..."
    & .\make.bat clean

    # Generate API documentation
    Write-Host "Generating API documentation..."
    python generate_api_docs.py

    # Build HTML documentation
    Write-Host "Building HTML documentation..."
    & .\make.bat html

    Write-Host "Documentation built successfully!"
    Write-Host "Open docs/_build/html/index.html in your browser to view the docs."
} 
catch {
    Write-Host "Error building documentation: $_"
}
finally {
    # Return to the original directory
    Pop-Location
} 