"""Base class for mutatable nodes in a tree.

More description to come.

"""
import logging
import math
import re

from .HyperparameterConfig import HyperparameterConfig

from typing import Any, Callable, Dict, List, Union, Optional

logger = logging.getLogger('metatree._MutableTreeNode')


class MutableTreeNode:
    """
    TODO
    """
    def __init__(self,
                 name: str,
                 hyperparameter_config: HyperparameterConfig,
                 parent: Optional['MutableTreeNode']=None):
        """Constructor.

        Args:
            name (str): Name of this MutableTreeNode
            hyperparameter_config (HyperparameterConfig): This
                MutableTreeNode's hyperparameter configuration object.
            parent (Optional[MutableTreeNode]): This MutableTreeNode's
                parent, if it has one.
        """
        # Name of this MutableTreeNode
        self.name = name
        # Configuration governing this MutableTreeNode's hyperparameters
        self.hyperparameter_config = hyperparameter_config
        # Shape of the data input to this MutableTreeNode's build process
        self.data_shape_in: List[int] = None
        # Shape of the data output by this MutableTreeNode's build process
        self.data_shape_out: List[int] = None

        # Root of this MutableTreeNode's tree
        self.root: MutableTreeNode
        # This MutableTreeNode's level in its tree
        self.level: int
        # This MutableTreeNode's children in its tree
        self.children: List['MutableTreeNode'] = []
        # Hyperparameter value delta dictionary. By default, all values are 0
        self.deltas: Dict[str, Union[int, float]]

        if parent is not None:
            # This is not a root MutableTreeNode
            self.add_parent(parent)

        else:
            self.parent = None
            # This is a root MutableTreeNode
            self.root = self
            self.level = 0
            # Make sure we have a HyperparameterConfig
            if self.hyperparameter_config is None:
                raise ValueError('No HyperparameterConfig supplied to root '
                                 'MutableTreeNode')
            # Use default values from the HyperparameterConfig
            self.deltas = self.hyperparameter_config.internal_values()

        pass

    def __str__(self) -> str:
        """Returns the name of this MutableTreeNode.

        Returns:
            self.name: Name of this MutableTreeNode.

        """
        return self.name

    def abbreviation(self) -> str:
        """Create an abbreviated name for this MutableTreeNode.

        Returns:
            (str): The abbreviated name

        """
        # Split name along non-alphanumeric characters
        name_parts = re.split('[^a-zA-Z\d]', self.name)

        # Characters that will eventually be joined to form the abbreviated
        # name
        abbreviation_parts = []

        for part in name_parts:
            if not part.isdigit():
                # Add the first character, in uppercase, if the part isn't
                # one big number
                abbreviation_parts.append(part[0].upper())
            else:
                # part is one big number, use the whole thing
                abbreviation_parts.append(part)

        # Form the abbreviated name
        abbreviation = ''.join(abbreviation_parts)
        return abbreviation

    def add(self, **kwargs):
        """Add numbers to the values of some hyperparameters at this gene.

        Updates are passed to this function as key-value pairs.

        Args:
            **kwargs: Key-value pairs for deltas to update. For each
                key-value pair, self.deltas[key] = value.

        Returns: None

        """
        # Apply mutations
        for key in kwargs:
            # Just set self.deltas[key] equal to kwargs[key]. Has the effect
            # of adding kwargs[key] to self.hyperparam(key)
            self.deltas[key] = kwargs[key]

        # Update this MutableTreeNode and all its descendants
        self.propagate('setup_children')
        pass

    def add_parent(self, parent: 'MutableTreeNode'):
        """Add a parent to this _MutableTreeNode.

        Args:
            parent (MutableTreeNode): The parent _MutableTreeNode

        Returns:

        """
        self.parent = parent
        self.root = parent.root
        self.level = parent.level + 1
        if self.hyperparameter_config is None:
            self.hyperparameter_config = parent.hyperparameter_config

        if (not hasattr(self, 'deltas')) or self.deltas is None or \
                len(self.deltas) == 0:
                self.deltas = {key: 0
                               for key in self.hyperparameter_config.names()}

        pass

    def descendants(self, level: Optional[int]=None) -> List['MutableTreeNode']:
        """Get all descendants of this node at a specified level. If no level is
            provided, return the leaf descendants of this node.
        
        Args:
            level (Optional[int]): Level of descendants to return: 0 for 
                this node, 1 for this node's children, 2 for this node's 
                children's children, etc. If `level` is not supplied, return the 
                highest-level descendants.

        Returns:
            (List[MutableTreeNode]): A list of descendant nodes.

        """
        if level is None:
            level = math.inf
        current_level = 0
        parents: List[MutableTreeNode] = []
        children = [self]
        while current_level < level:
            if not children:
                return parents
            parents = children[:]
            children = [c for p in parents for c in p.children]
            current_level += 1
        return children

    def hyperparam(self, param_key: str) -> Any:
        """Returns the value of the hyperparameter with name param_key
        associated with this MutableTreeNode.

        Hyperparameter values are tracked by MutableTreeNodes as numerical
        quantities, but for hyperparameters which are not numeric, this value is
        an index into a list of objects. MutableTreeNode.hyperparam() returns
        the hyperparameter object, not its internal index value.

        Args:
            param_key (str): Name of the hyperparameter.

        Returns:
            hyperparam (Any): Hyperparameter value associated with this
                MutableTreeNode.

        """
        param_type = self.hyperparameter_config.types()[param_key]

        if param_type == 'object':
            # The hyperparameter is a general object. The MutableTreeNode
            # parameter is an index into a list of hyperparameter objects.

            # Get the index value
            idx = self._accumulate(param_key)

            # Get the object associated with that index value
            value = self.hyperparameter_config.object_lists()[param_key][idx]

        elif param_type == 'float':
            # The hyperparameter is a floating point number
            value = float(self._accumulate(param_key))

        elif param_type == 'int':
            # The hyperparameter is an integer
            value = self._accumulate(param_key)

        else:
            # Unrecognized param type
            raise KeyError(f'Unrecognized param_type {param_type} for '
                           f'hyperparam {param_key}')

        return value

    def last_descendant(self) -> 'MutableTreeNode':
        """Returns the last descendant node among this node's descendants,
        as ordered by the indexing of each node's `children` attribute.

        Returns:
            (MutableTreeNode): The last descendant node among this node's
                descendants, as ordered by the indexing of each node's
                `children` attribute.

        """
        last = self
        while len(last.children) > 0:
            last = last.children[-1]
        return last

    def path_name(self, n_levels: int=-1) -> str:
        """Return the path name of this MutableTreeNode in a tree of nodes.
        The path name includes this node1's abbreviated name, plus the
        abbreviated names of ancestors up to `n_levels` levels above this
        node1. Default value of -1 uses all ancestor name abbreviations.


        Returns:
            (str): The path name.

        """
        # All the parts that will become the path name
        name_parts = [self.abbreviation()]
        # The currently active gene
        active = self

        level_counter = 0
        while active.parent is not None and level_counter != n_levels:
            active = active.parent
            level_counter += 1
            name_parts.insert(0, active.abbreviation())

        full_name = '/'.join(name_parts)
        return full_name

    def propagate(self, f: Union[str, Callable], *args, **kwargs):
        """Call a function at this MutableTreeNode and all of its descendants.

        This can be used to either call a method by name at each
        MutableTreeNode (different functions for different nodes),
        or to directly call one function at each MutableTreeNode (same
        function for different nodes).

        Args:
            f (Union[str, Callable]): Name of the method to call if str,
                else a callable function. The first argument of f is always
                this MutableTreeNode, and should not be supplied as a part of
                *args or **kwargs.
            *args: Function arguments.
            **kwargs: Function keyword arguments.

        Returns:
            None

        """
        if isinstance(f, str):
            # f is a string. Get the method with name f in this MutableTreeNode
            f_actual = getattr(self, f)
            # Call the function
            f_actual(*args, **kwargs)
        else:
            # f is a callable function. Just use it as-is
            f(self, *args, **kwargs)

        # Call the same method in all children
        if self.children is not None:
            for child in self.children:
                child.propagate(f, *args, **kwargs)
        pass

    def set(self, call_setup: bool=True, **kwargs):
        """For each hyperparameter name key in kwargs, adjust self.deltas[
        key] so that self.hyperparam(key) == kwargs[key].

        Updates are passed to this function as key-value pairs;

        Args:
            call_setup (bool): Determines whether or not this function 
                propagates a setup_children call.
            **kwargs: Key-value pairs for hyperparameters to update.

        Returns: None

        """
        # Apply mutations
        for key in kwargs:
            param_type = self.hyperparameter_config.types()[key]
            if param_type == 'object':
                # For object hyperparameters, need to refer back to the index
                # associated with each value and change that
                object_list = self.hyperparameter_config.object_lists()[key]
                target_idx = object_list.index(kwargs[key])
                current_idx = self._accumulate(key)
                self.deltas[key] += target_idx - current_idx
            else:
                # Set self.deltas[key] to the value that makes
                # self.hyperparam(key) equal kwargs[key]. In other words,
                # subtract the current hyperparam value from kwargs[key]
                current_value = self.hyperparam(key)
                self.deltas[key] += kwargs[key] - current_value
        if call_setup:
            # Update this MutableTreeNode and all its descendants
            self.propagate('setup_children')
        pass

    def setup_children(self):
        """Set up child MutableTreeNodes.

        Returns:
            None

        """
        pass

    def _accumulate(self, param_key: str) -> Union[int, float]:
        """Calculate hyperparameter internal value at current MutableTreeNode.

        Hyperparameter internal values for each MutableTreeNode are specified
        as a sum of the internal value deltas for this MutableTreeNode and
        all of its ancestors.

        Args:
            param_key (str): Name of hyperparameter to accumulate a value for.

        Returns:
            value (Union[int, float]): Accumulated parameter internal value.

        """
        # Accumulated parameter value
        value = 0
        # Current MutableTreeNode in the ancestry
        current_node = self
        # Move up the ancestry, adding deltas to value
        while current_node.parent is not None:
            value += current_node.deltas[param_key]
            current_node = current_node.parent
        # Add the root MutableTreeNode  delta
        value += current_node.deltas[param_key]

        return value
