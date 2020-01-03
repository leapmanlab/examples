"""Generate probability distributions over hyperparameter feasible regions,
use them to create or modify Gene graphs.

"""
import logging
import time

import numpy as np

from typing import Optional, Sequence, Union

from .Gene import Gene
from .HyperparameterConfig import HyperparameterConfig
from ._MutableTreeNode import MutableTreeNode
from .types import Regions


logger = logging.getLogger('metatree.Mutator')


class Mutator:
    """Generate probability distributions over hyperparameter feasible
    regions, use them to create or modify Gene graphs.
    
    """
    def __init__(self,
                 hyperparam_config: HyperparameterConfig,
                 mutable_hyperparams: Sequence[str],
                 feasible_regions: Regions,
                 additive: bool=False,
                 rates: Union[float, Sequence[float]]=1.,
                 addition_scales: Union[float, Sequence[float]]=1.,
                 seed: Optional[int]=None):
        """Constructor.

        Args:
            hyperparam_config (HyperparameterConfig): The
                HyperparameterConfig containing the feasible regions to
                generate probability distributions over.
            mutable_hyperparams (Sequence[str]): Collection of names of
                hyperparameters to apply the mutation operation to.
            feasible_regions (Regions): A dictionary specifying feasible
                regions for mutable hyperparameter values at each level of a
                Gene tree. Each dictionary key is the name of a
                hyperparameter whose name is also a key in
                `mutable_hyperparams`. Each corresponding value is one or
                more intervals specified as a tuple of boundary values,
                e.g. `feasible_regions['n_kernels'] = [(10, 100), (-20,
                20)]`. A Gene at level i in a Gene tree will sample
                hyperparameter `hp`'s delta values from
                `feasible_regions[hp][j]`, where
                `j = min(len(feasible_regions[hp]), i)`. For hyperparameters
                of 'object' type, feasible regions govern the value of an
                index into the hyperparameter's object list.
            additive (bool): If True, mutation samples are added to
                previous hyperparameter values. If False, mutation samples
                become the new hyperparameter values.
            rates (Union[float, Sequence[float]): During a Mutator's mutation
                pass on a Gene graph, hyperparameter mutations are generated as
                follows: For a Gene at level `i` in the Gene graph, for each
                hyperparameter `hp` whose name is in `mutable_hyperparams`,
                mutate `hp` with  probability `rates[j]`, where
                `j = min(len(rates), i)`.
            addition_scales (Union[float, Sequence[float]): Multiplicative
                scaling factors for mutations in additive mode. A
                hyperparameter mutation for a Gene at level `i` in a Gene
                graph has its effect size scaled by `scales[j]`, where
                `j = min(len(addition_scales), i)`.
            seed (Optional[int]): RNG seed for the Mutator.

        """
        self.hyperparam_config = hyperparam_config
        self.mutable_hyperparams = mutable_hyperparams
        self.feasible_regions = feasible_regions
        # Wrap single elements in a list
        for key in self.feasible_regions:
            regions = self.feasible_regions[key]
            if not isinstance(regions, Sequence):
                self.feasible_regions[key] = [regions]
        self.additive = additive
        # Wrap numeric rates or scales inputs in a list
        if not isinstance(rates, Sequence):
            rates = [rates]
        self.rates = rates
        if not isinstance(addition_scales, Sequence):
            addition_scales = [addition_scales]
        self.addition_scales = addition_scales
        # Create a RNG seed if none is provided
        if seed is None:
            seed = int(time.strftime('%Y%m%d%H%M%S')) % (2**32 - 1)
        self.seed = seed
        # Create a numpy RandomState for the Mutator's random operations
        self.rng = np.random.RandomState(self.seed)

        # Dict of functional hyperparameter groupings, paired with
        # multiplicative mutation rate modifiers.
        # Example:
        #   self.groups['training'] = (['learn_rate', 'momentum'], 0.5)
        self.groups = {}

        pass

    def add_group(self,
                  key: str,
                  hyperparams: Sequence[str],
                  rate_modifier: float):
        """Add a hyperparameter mutation group to self.groups.

        Hyperparameters can be grouped together functionally to provide
        different mutation instance_settings, e.g. for optimization algorithm
        hyperparameters vs. regularization hyperparameters. Each mutation
        group is labeled with a key, such that
        `self.groups[key] = (hyperparams, rate_modifier)`.

        Args:
            key (str): Hyperparameter mutation group name.
            hyperparams (Sequence[str]): List of names of hyperparameters in
                the group.
            rate_modifier (float): Multiplicative mutation rate modifier for
                hyperparameters in the new group.

        Returns: None

        """
        self.groups[key] = (hyperparams, rate_modifier)
        pass

    def mutate(self, node: MutableTreeNode) -> None:
        """Apply `Mutator._mutate_node()` to a MutableTreeNode and its input
        and descendants.

        Depending on whether Mutator.additive is True, mutation operations
        can sample hyperparameter values directly from feasible ranges,
        or add scaled samples to previous values. In the former case,
        a Mutator can be used to randomly sample a hyperparameter range. In
        the latter case, a Mutator can be used for modifying an existing
        network architecture as part of, e.g., an evolutionary procedure.

        Args:
            node (MutableTreeNode): A MutableTreeNode to mutate.

        Returns: None

        """
        # Apply the mutation operation to each Gene in the new Gene graph,
        # appending them recursively starting at the graph root, updating the
        # Gene graph after each mutation to reflect the downstream structural
        # changes potentially created by hyperparameter value changes
        nodes = [node]
        while len(nodes) > 0:
            # Get the first node in the list
            g = nodes.pop(0)
            # Mutate it
            self._mutate_node(g)
            g.setup_children()
            if isinstance(g, Gene):
                # Add its children and inputs to the list
                nodes += g.children[:]
                nodes += g.inputs[:]

    def _final_rate(self, hyperparam: str, rate: float) -> float:
        """Compute the final mutation rate for a hyperparameter, by taking a
        starting input rate and multiplying it by multipliers associated to
        all mutation groups containing the input hyperparam.

        Args:
            hyperparam (str): Name of the hyperparameter to compute a final
                rate for.
            rate (float): Base mutation rate for the input hyperparam.

        Returns:
            (float): The final mutation rate for the input hyperparam.

        """
        for group in self.groups:
            if hyperparam in group[0]:
                rate *= group[1]

        return rate

    def _mutate_node(self, node: MutableTreeNode):
        """Mutate the hyperparameters in a node.

        Args:
            node (MutableTreeNode): The Gene to modify.

        Returns: None

        """
        # The level of the Gene in its Gene graph
        level = node.level
        # Level-dependent mutation properties
        if level < len(self.rates):
            rate = self.rates[level]
        else:
            rate = self.rates[-1]

        if level < len(self.addition_scales):
            scale = self.addition_scales[level]
        else:
            scale = self.addition_scales[-1]

        # Mutate each mutable hyperparameter
        for hp in self.mutable_hyperparams:
            # Compute final mutation rate. Take into account any mutation group
            # modifications
            p_mutation = self._final_rate(hp, rate)
            # Dat Bernoulli trial
            # noinspection PyArgumentList
            if self.rng.rand() < p_mutation:
                # Generate a new value by sampling from the feasible range
                # appropriate to the input gene's level
                mutation_value = self._sample(hp, level)
                if self.additive:
                    node.deltas[hp] += scale * mutation_value
                else:
                    node.deltas[hp] = mutation_value

                # Clip the delta value to ensure that
                # gene.hyperparam(hp) stays in the root feasible
                # region for hp

                # Root feasible region for hp
                root_region = self.feasible_regions[hp][0]
                # gene's delta value for hp
                d = node.deltas[hp]
                # Use hp value accumulated across the gene's ancestors to
                # determine clipping range, if the gene has ancestors. Else,
                # just use the root region
                if node.parent is None:
                    d_region = (root_region[0], root_region[1])
                else:
                    # hp value accumulated across `gene`'s ancestors
                    parent_value = node.parent.hyperparam(hp)
                    # Replace with numerical index for object hyperparams
                    if node.hyperparameter_config.types()[hp] == 'object':
                        object_lists = node.hyperparameter_config.object_lists()
                        parent_value = object_lists[hp].index(parent_value)
                    # Calculate the interval d needs to stay in, in order for
                    # gene.hyperparam(hp) to stay in root_region
                    d_region = (root_region[0] - parent_value,
                                root_region[1] - parent_value)
                # Clip d
                d_clipped = min(d_region[1], max(d_region[0], d))
                node.deltas[hp] = d_clipped

        pass

    def _sample(self, hyperparam: str, level: int) -> Union[float, int]:
        """Sample a value from the feasible region for a hyperparameter at a
        specified level within a Gene graph.

        Args:
            hyperparam (str): Name of the hyperparameter for which to
                generate a sample.
            level (int): Level within the Gene graph for which to generate a
                sample. Level 0 samples come from root feasible ranges,
                level >0 samples come from descendant feasible ranges.

        Returns:
            val (Union[float, int]): The sampled hyperparameter value.

        """
        # Get the feasible region to use for the input hyperparam at the
        # input level
        regions = self.feasible_regions[hyperparam]
        region = regions[min(level, len(regions) - 1)]

        # Hyperparameter minimum and maximum feasible values
        min_val = region[0]
        max_val = region[1]

        # Sampling procedure depends on hyperparameter type
        hyperparam_type = self.hyperparam_config.types()[hyperparam]
        if hyperparam_type == 'float':
            # noinspection PyArgumentList
            val = min_val + (max_val - min_val) * self.rng.rand()
        else:
            val = self.rng.randint(min_val, max_val + 1)

        return val
