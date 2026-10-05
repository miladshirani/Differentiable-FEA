"""pytest configuration: make the ``scr`` package importable and silence a harmless warning."""
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "test")))
warnings.filterwarnings("ignore", message=".*torch.jit.script.*")
