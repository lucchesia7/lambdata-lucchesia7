import pandas as pd
import numpy as np
import pytest
from helper_functions_test import Wrangle, Randomize

### Instantiate Classes
wrangle_methods = Wrangle()
rand_methods = Randomize()

### Create DF for testing
df = pd.DataFrame(
    np.random.randint(0, 100, size=(15, 4)),
    columns=list('ABCD')
)


def test_nc():
    """
    Test Null_Count Method from Wrangle Class
    """
    assert wrangle_methods.null_count(df) == 0
    
    ### Compare null count to ensure proper accuracy of function.
    assert wrangle_methods.null_count(df) == df.isnull().sum().sum()


def test_rand():
    """
    Test Randomize Method from Randomize Class
    """
    assert len(rand_methods.randomize(df, seed=42)) == len(df)
