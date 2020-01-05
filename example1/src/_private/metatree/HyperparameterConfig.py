"""HyperparameterConfig class specifies a collection of neural network
hyperparameters and feasible regions for those parameters.


"""

from typing import Any, Dict, List, Sequence, Union


class HyperparameterConfig:
    """Specifies a collection of neural network hyperparameters.

    Hyperparameters may have arbitrary type T, but HyperparameterConfig handles
    int and float types differently from others. If T in [int, float, bool],
    hyperparameter specification requires only a name and default value. For
    other T, hyperparameter specification requires a name, default value,
    and list of all possible values.

    TODO
    Attributes:

    """
    def __init__(self, hyperparam_dict: Dict[str, Any]):
        """Constructor.

        Args:
            hyperparam_dict (Dict[str, Sequence[Any]]): Dictionary of
                hyperparameter information. Hyperparameters can be
                numeric - floats or ints, or objects - elements from a
                list of Python objects.

                Numeric hyperparameter entries take the form
                    hyperparam_dict[numeric_key] = default_value,
                e.g. hyperparam_dict['n_kernels'] = 64.
                Note that 'int' and 'float' hyperparameters are distinguished
                by the type of default_value.

                Object hyperparameter entries take the form
                    hyperparam_dict[object_key] = (default_value, object_list),
                e.g. hyperparam_dict['activation'] =
                    (tf.nn.relu, [tf.nn.relu, tf.nn.elu, tf.nn.tanh]).

        """
        self.hyperparams = {}
        for hyperparam, info in hyperparam_dict.items():
            # Transform hyperparam info into its internal representation,
            # and add to self.hyperparams
            self.set(hyperparam, info)

        pass

    def __str__(self) -> str:
        """Build a string representation of the HyperparameterConfig.

        Returns:
            (str): String representation of the HyperparameterConfig.

        """
        return '\n'.join([f'{hp}: {self.hyperparams[hp]}'
                          for hp in self.hyperparams])

    def internal_values(self) -> Dict[str, Union[int, float]]:
        """Get internal default values for all hyperparameters.

        Note that these values are for the hyperparameter *internal*
        representations - object hyperparameters will return integer
        indices.

        Returns:
            (Dict[str, Union[int, float]]): Dictionary mapping hyperparameter
                names to their internal default values.

        """
        return {name: self.hyperparams[name][1] for name in self.names()}

    def names(self) -> List[str]:
        """Return a list of all hyperparameter names.

        Returns:
            (List[str]): Alphabetized list of all hyperparameter names.

        """
        return sorted(self.hyperparams)

    def object_lists(self) -> Dict[str, List[Any]]:
        """Return object lists for all object hyperparameters.

        Returns:
            (Dict[str, List[Any]]): Dictionary mapping object hyperparameter
                names to object lists.

        """
        return {name: self.hyperparams[name][2] for name in self.names()
                if self.hyperparams[name][0] == 'object'}

    def set(self, hyperparam: str, info: Union[int, float, Sequence]):
        """Create or modify a hyperparameter dict entry.

        If the input param is numeric, map {name: default_value} ->
        {name: (type, default_value)}.

        If the input param is bool, map {name: default_value} ->
        {name: ('object', default_value, [False, True])}.

        If the input param is object, map {name: (default_value, object_list)}
        -> (type, default_index, object_list).
        Object hyperparams are represented internally as an index into
        object_list. Genes manipulate the index, which is then translated
        back into an object in object_list at computation graph build time.

        Args:
            hyperparam (str): Name of the hyperparameter to set.
            info (Union[int, float, Sequence]): Hyperparameter default value
            (for numeric hyperparameters), or a Sequence of hyperparameter
            info (for other hyperparameters)

        Returns: None

        """
        # Determine hyperparameter type by info type
        if isinstance(info, int):
            self.hyperparams[hyperparam] = ('int', info)
        elif isinstance(info, float):
            self.hyperparams[hyperparam] = ('float', info)
        elif isinstance(info, Sequence):
            default_value = info[0]
            object_list = info[1]
            # Index of the default value in the object list
            default_idx = object_list.index(default_value)
            self.hyperparams[hyperparam] = ('object', default_idx, object_list)
        elif isinstance(info, bool):
            # Treat like an object, but handle this common special case for
            # simplicity
            self.hyperparams[hyperparam] = ('object', info, [False, True])

        pass

    def types(self) -> Dict[str, str]:
        """Get types for all hyperparameters (int, float, object).

        Returns:
            (Dict[str, tuple]): Dictionary mapping hyperparameter names to
                hyperparameter types.

        """
        return {name: self.hyperparams[name][0] for name in self.names()}

    def values(self) -> Dict[str, Any]:
        """Get default values for all hyperparameters.
`
        These values are for the hyperparameter public representations -
        object hyperparameters will return objects.

        Returns:
            (Dict[str, Any]): Dictionary mapping hyperparameter names to
                their default values.

        """
        all_values = {}
        for name in self.names():
            idx = self.hyperparams[name][1]
            hp_type = self.hyperparams[name][0]
            if hp_type == 'object':
                all_values[name] = self.hyperparams[name][2][idx]
            else:
                all_values[name] = idx

        return all_values
