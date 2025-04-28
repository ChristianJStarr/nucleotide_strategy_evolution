============
Contributing
============

Thank you for your interest in contributing to the Nucleotide Strategy Evolution framework! This document provides guidelines for contributions.

Getting Started
--------------

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Create a new branch** for your feature or bugfix
4. **Make your changes**
5. **Run tests** to ensure your changes don't break existing functionality
6. **Submit a pull request**

Setting Up Development Environment
--------------------------------

.. code-block:: bash

    # Clone your fork
    git clone https://github.com/YOUR_USERNAME/nucleotide_strategy_evolution.git
    cd nucleotide_strategy_evolution
    
    # Create and activate virtual environment
    python -m venv .venv
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    # source .venv/bin/activate
    
    # Install development dependencies
    pip install -e ".[dev]"

Code Style
---------

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Run these tools before submitting a pull request:

.. code-block:: bash

    # Format code
    black nucleotide_strategy_evolution tests
    
    # Run linting
    flake8 nucleotide_strategy_evolution tests
    
    # Run type checking
    mypy nucleotide_strategy_evolution

Documentation
------------

When adding new features, please include:

- **Docstrings** for all modules, classes, and functions (using NumPy/Google format)
- **Examples** showing how to use the feature
- **Updates to relevant documentation pages**

Run the documentation build locally:

.. code-block:: bash

    # Install Sphinx and dependencies
    pip install -e ".[dev]"
    
    # Build documentation
    cd docs
    make html
    
    # View documentation (open _build/html/index.html in your browser)

Testing
------

We use pytest for testing. All new features should include tests.

.. code-block:: bash

    # Run all tests
    pytest
    
    # Run tests with coverage
    pytest --cov=nucleotide_strategy_evolution

Submit Pull Requests
------------------

When submitting a pull request:

1. **Reference any related issues**
2. **Describe what your changes do**
3. **Explain how you've tested your changes**
4. **Update documentation** if needed
5. **Ensure all tests pass**

Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~

- [ ] Code follows style guidelines
- [ ] Tests added/updated and passing
- [ ] Documentation updated
- [ ] All checks pass (format, lint, type check)
- [ ] Rebased onto the latest main branch

Feature Requests and Bug Reports
------------------------------

- **Feature Requests**: Open an issue with the tag "enhancement"
- **Bug Reports**: Open an issue with the tag "bug", include steps to reproduce, expected vs. actual behavior

Thank you for contributing! 