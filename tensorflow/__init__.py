"""
TensorFlow compatibility stub for environments without TensorFlow installed.

This module provides stub implementations of TensorFlow classes and functions
for code that imports TensorFlow but doesn't strictly require it for basic operation.
This allows importing modules that use TensorFlow without having the full
TensorFlow package installed.

Note: This is NOT a functional replacement for TensorFlow. It only provides
dummy implementations to allow imports to succeed. Any actual TensorFlow
operations will need the real package installed.
"""

import logging
import sys
import warnings
from typing import Any, List, Dict, Optional, Callable, Union, Tuple
import numpy as np

logger = logging.getLogger(__name__)

warnings.warn(
    "Using TensorFlow stub module. This is not a functional replacement for TensorFlow. "
    "Install tensorflow package for full functionality.",
    RuntimeWarning,
    stacklevel=2
)

# Version info
__version__ = "2.15.0-stub"
VERSION = "2.15.0-stub"


class Tensor:
    """
    Stub implementation of tf.Tensor.
    
    Wraps a numpy array to provide a tensor-like interface.
    """
    
    def __init__(self, data: Any, dtype: Optional[Any] = None, name: Optional[str] = None):
        if isinstance(data, np.ndarray):
            self._data = data
        else:
            self._data = np.array(data, dtype=dtype)
        self.dtype = dtype
        self.name = name
        self.shape = self._data.shape
        self.device = "/cpu:0"
    
    def numpy(self) -> np.ndarray:
        """Convert tensor to numpy array."""
        return self._data
    
    def __repr__(self) -> str:
        return f"<tf.Tensor: shape={self.shape}, dtype={self.dtype}, stub=True>"
    
    def __add__(self, other: Any) -> 'Tensor':
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data + other_data)
    
    def __mul__(self, other: Any) -> 'Tensor':
        other_data = other._data if isinstance(other, Tensor) else other
        return Tensor(self._data * other_data)


class Variable(Tensor):
    """Stub implementation of tf.Variable."""
    
    def __init__(self, initial_value: Any, dtype: Optional[Any] = None, name: Optional[str] = None):
        super().__init__(initial_value, dtype, name)
        self.trainable = True
    
    def assign(self, value: Any) -> None:
        """Assign a new value to the variable."""
        if isinstance(value, Tensor):
            self._data = value._data
        else:
            self._data = np.array(value)


class Session:
    """Stub implementation of tf.Session (TensorFlow 1.x compatibility)."""
    
    def __init__(self, target: str = "", graph: Optional[Any] = None, config: Optional[Any] = None):
        self.target = target
        self.graph = graph
        self.config = config
    
    def run(self, fetches: Any, feed_dict: Optional[Dict] = None) -> Any:
        """Stub run method."""
        logger.debug("Session.run() called - using stub implementation")
        if isinstance(fetches, Tensor):
            return fetches.numpy()
        return fetches
    
    def __enter__(self) -> 'Session':
        return self
    
    def __exit__(self, *args: Any) -> None:
        pass


class Graph:
    """Stub implementation of tf.Graph."""
    
    def __init__(self):
        self._nodes = []
    
    def as_default(self) -> 'Graph':
        return self
    
    def __enter__(self) -> 'Graph':
        return self
    
    def __exit__(self, *args: Any) -> None:
        pass


# Stub constants and dtypes
class DType:
    """Stub implementation of tf.dtypes."""
    def __init__(self, name: str):
        self.name = name
    
    def __repr__(self) -> str:
        return f"tf.{self.name}"


float32 = DType("float32")
float64 = DType("float64")
int32 = DType("int32")
int64 = DType("int64")
bool_ = DType("bool")
string = DType("string")

# Data type aliases
int8 = DType("int8")
int16 = DType("int16")
uint8 = DType("uint8")
uint16 = DType("uint16")
uint32 = DType("uint32")
uint64 = DType("uint64")
float16 = DType("float16")
complex64 = DType("complex64")
complex128 = DType("complex128")


# Stub functions
def constant(value: Any, dtype: Optional[Any] = None, shape: Optional[Tuple] = None, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.constant."""
    if shape is not None:
        data = np.full(shape, value, dtype=dtype)
    else:
        data = value
    return Tensor(data, dtype, name)


def Variable(initial_value: Any, dtype: Optional[Any] = None, name: Optional[str] = None) -> Variable:  # noqa: N802
    """Stub implementation of tf.Variable constructor."""
    return Variable(initial_value, dtype, name)


def zeros(shape: Tuple, dtype: DType = float32, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.zeros."""
    return Tensor(np.zeros(shape), dtype, name)


def ones(shape: Tuple, dtype: DType = float32, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.ones."""
    return Tensor(np.ones(shape), dtype, name)


def random_normal(shape: Tuple, mean: float = 0.0, stddev: float = 1.0, dtype: DType = float32, seed: Optional[int] = None, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.random.normal."""
    if seed is not None:
        np.random.seed(seed)
    data = np.random.normal(mean, stddev, shape)
    return Tensor(data, dtype, name)


def random_uniform(shape: Tuple, minval: float = 0.0, maxval: float = 1.0, dtype: DType = float32, seed: Optional[int] = None, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.random.uniform."""
    if seed is not None:
        np.random.seed(seed)
    data = np.random.uniform(minval, maxval, shape)
    return Tensor(data, dtype, name)


def matmul(a: Tensor, b: Tensor, transpose_a: bool = False, transpose_b: bool = False, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.matmul."""
    a_data = a._data.T if transpose_a else a._data
    b_data = b._data.T if transpose_b else b._data
    return Tensor(np.matmul(a_data, b_data), name=name)


def add(x: Tensor, y: Tensor, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.add."""
    return Tensor(x._data + y._data, name=name)


def multiply(x: Tensor, y: Tensor, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.multiply."""
    return Tensor(x._data * y._data, name=name)


def reduce_mean(input_tensor: Tensor, axis: Optional[Union[int, List]] = None, keepdims: bool = False, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.reduce_mean."""
    return Tensor(np.mean(input_tensor._data, axis=axis, keepdims=keepdims), name=name)


def reduce_sum(input_tensor: Tensor, axis: Optional[Union[int, List]] = None, keepdims: bool = False, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.reduce_sum."""
    return Tensor(np.sum(input_tensor._data, axis=axis, keepdims=keepdims), name=name)


def cast(x: Tensor, dtype: DType, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.cast."""
    return Tensor(x._data.astype(dtype.name if hasattr(dtype, 'name') else dtype), name=name)


def reshape(tensor: Tensor, shape: List[int], name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.reshape."""
    return Tensor(tensor._data.reshape(shape), name=name)


def placeholder(dtype: DType, shape: Optional[Tuple] = None, name: Optional[str] = None) -> Tensor:
    """Stub implementation of tf.placeholder (TF 1.x compatibility)."""
    if shape is None:
        shape = (1,)
    return Tensor(np.zeros(shape), dtype, name)


def get_default_graph() -> Graph:
    """Stub implementation of tf.get_default_graph."""
    return Graph()


def Session(target: str = "", graph: Optional[Any] = None, config: Optional[Any] = None) -> Session:  # noqa: N802
    """Stub implementation of tf.Session (TF 1.x compatibility)."""
    return Session(target, graph, config)


# Keras submodule stub
class keras:
    """Stub implementation of tf.keras."""
    
    class Model:
        """Stub Keras Model."""
        def __init__(self, *args: Any, **kwargs: Any):
            self.layers = []
            self.inputs = []
            self.outputs = []
        
        def compile(self, *args: Any, **kwargs: Any) -> None:
            pass
        
        def fit(self, *args: Any, **kwargs: Any) -> Any:
            return type('History', (), {'history': {'loss': [0.1, 0.05], 'accuracy': [0.9, 0.95]}})()
        
        def predict(self, x: Any, *args: Any, **kwargs: Any) -> np.ndarray:
            if isinstance(x, np.ndarray):
                return np.zeros((x.shape[0], 1))
            return np.zeros((1, 1))
        
        def evaluate(self, *args: Any, **kwargs: Any) -> List[float]:
            return [0.05, 0.95]
    
    class Sequential(Model):
        """Stub Keras Sequential model."""
        def __init__(self, layers: Optional[List] = None):
            super().__init__()
            self._layers = layers or []
        
        def add(self, layer: Any) -> None:
            self._layers.append(layer)
    
    class layers:
        """Stub Keras layers."""
        
        class Layer:
            """Base layer stub."""
            def __init__(self, *args: Any, **kwargs: Any):
                pass
        
        class Dense(Layer):
            """Stub Dense layer."""
            def __init__(self, units: int, activation: Optional[str] = None, **kwargs: Any):
                self.units = units
                self.activation = activation
        
        class LSTM(Layer):
            """Stub LSTM layer."""
            def __init__(self, units: int, **kwargs: Any):
                self.units = units
        
        class Conv2D(Layer):
            """Stub Conv2D layer."""
            def __init__(self, filters: int, kernel_size: Tuple, **kwargs: Any):
                self.filters = filters
                self.kernel_size = kernel_size
        
        class Flatten(Layer):
            """Stub Flatten layer."""
            pass
        
        class Dropout(Layer):
            """Stub Dropout layer."""
            def __init__(self, rate: float, **kwargs: Any):
                self.rate = rate
        
        class BatchNormalization(Layer):
            """Stub BatchNormalization layer."""
            pass
        
        class Embedding(Layer):
            """Stub Embedding layer."""
            def __init__(self, input_dim: int, output_dim: int, **kwargs: Any):
                self.input_dim = input_dim
                self.output_dim = output_dim


# Export everything
__all__ = [
    'Tensor', 'Variable', 'Session', 'Graph',
    'constant', 'zeros', 'ones', 'random_normal', 'random_uniform',
    'matmul', 'add', 'multiply', 'reduce_mean', 'reduce_sum', 'cast', 'reshape',
    'placeholder', 'get_default_graph',
    'float32', 'float64', 'int32', 'int64', 'bool_', 'string',
    'keras', '__version__', 'VERSION'
]

# Ensure keras submodule is available
from . import keras as _keras_module
sys.modules['tensorflow.keras'] = _keras_module
