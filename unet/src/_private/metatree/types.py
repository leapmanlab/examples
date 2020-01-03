"""Collection of custom variable types used by metatree.

"""
import tensorflow as tf

from typing import Dict, TypeVar, Sequence, Tuple, Union

# Possible sources for edges. Used as keys in the computation graph build map
EdgeSource = TypeVar(Union['Gene', str])

# Type for the map used during Gene.build() and Edge.build() to associate
# Genes with the Tensors they construct
BuildMap = TypeVar(Dict[EdgeSource, Union[tf.Tensor, 'Gene']])

# An interval of int or float values. Used for defining hyperparameter
# feasible regions
_Interval = TypeVar(Tuple[Union[int, float], Union[int, float]])
# Type for dicts of feasible regions for hyperparameter values
Regions = TypeVar(Dict[str, Union[_Interval, Sequence[_Interval]]])
