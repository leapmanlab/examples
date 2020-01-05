"""A check for train-eval loops to see whether training should stop.

Training should stop if the number of epochs completed equals a specified
`n_epochs`, or if an eval metric-based stopping criterion is met.

"""
from typing import Any, Dict, Tuple


def stop_check(eval_results: Dict[str, Any],
               stop_criterion: Tuple[str, float, int], 
               n_epochs: int, 
               epoch: int) -> bool:
    """A check for train-eval loops to see whether training should stop.

    Training should stop if the number of epochs completed equals a specified
    `n_epochs`, or if an eval metric-based stopping criterion is met. The
    stopping criterion is a triplet (metric, threshold, epoch), Where 'metric'
    is a key in the `eval_results` dict, 'threshold' is a metric cutoff value 
    - the stopping criterion is met if `eval_results[metric] < threshold` - and
    'epoch' is the number of the epoch when stop criterion testing should 
    begin.

    Args:
        eval_results (Dict[str, Any]): Dictionary of net evaluation results.
        stop_criterion (Tuple[str, float, int]): A triplet
            (metric, threshold, epoch), where 'metric' is a key in an
            `eval_results` dict returned by `genenet.GeneNet.evaluate`,
            'threshold' is a metric cutoff value - the stopping criterion is
            met if `eval_results[metric] < threshold` - and 'epoch' is the
            number of the epoch when stop criterion testing should begin.
        n_epochs (int): Maximum number of training epochs. Training will stop
            after `n_epochs` epochs are completed or if the stop criterion
            is met, whichever happens first.
        epoch (int): The current training epoch.

    Returns:
        (bool): True if training should stop now, False otherwise.    

    """
    # Unpack stop_criterion
    smetric, sthreshold, sepoch = stop_criterion
    # Check if the stop metric in the eval results is below the stop
    # threshold
    if isinstance(eval_results, dict) and smetric in eval_results:
        stop_condition_met = eval_results[smetric] < sthreshold
    else:
        stop_condition_met = False

    # Stop if the stop condition is met on epoch `stop_epoch` or after, or
    # if `n_epochs` epochs are completed
    stop_training = (stop_condition_met and epoch >= sepoch) \
        or epoch >= n_epochs - 1

    return stop_training

