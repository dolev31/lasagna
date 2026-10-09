"""Example experiment configurations.

This package exists so the YAML files beside it ship in the wheel. Without an
``__init__.py`` setuptools' package discovery skips the directory, and
``package-data`` patterns can only match inside a discovered package -- so the
example configs the docs tell people to run were absent from every install.
"""
