import pytest
import numpy as np
import networkx as nx

@pytest.fixture(autouse=True)
def set_random_seed():
    """
    Automatically resets random seeds before every test function 
    to ensure deterministic and reproducible tests.
    """
    np.random.seed(42)
    # If modules use their own RNG instances, this global seed might not catch everything,
    # but the package components generally allow passing a seed or use numpy global.
    yield