import pandas as pd
import numpy as np


def get_population_diversity(evolution_id: str):
    data = {"iteration": np.arange(50), "diversity": np.random.rand(50) * 0.5 + 0.2}
    return pd.DataFrame(data)


def get_code_complexity(evolution_id: str):
    data = {"iteration": np.arange(50), "complexity": np.random.randint(10, 30, 50)}
    return pd.DataFrame(data)


def get_linter_scores(evolution_id: str):
    data = {"iteration": np.arange(50), "linter_score": np.random.rand(50) * 4 + 6}
    return pd.DataFrame(data)
