import sys
import os

def pytest_sessionstart(session):
    """
    Allows tests to find the 'src' module by adding the project root
    to the Python path before test collection begins.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)