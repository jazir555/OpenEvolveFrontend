"""predict package."""

from .test_aggregation import TestAggregation
from .test_best_of_n import TestBestOfN
from .test_chain_of_thought import TestChainOfThought
from .test_code_act import TestCodeAct
from .test_knn import TestKnn
from .test_multi_chain_comparison import TestMultiChainComparison
from .test_parallel import TestParallel
from .test_predict import TestPredict
from .test_program_of_thought import TestProgramOfThought
from .test_react import TestReact

__all__ = ['test_aggregation', 'test_best_of_n', 'test_chain_of_thought', 'test_code_act', 'test_knn', 'test_multi_chain_comparison', 'test_parallel', 'test_predict', 'test_program_of_thought', 'test_react']
