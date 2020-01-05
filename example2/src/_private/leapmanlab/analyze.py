"""Utility script to summarize the output directories of an experiment run

"""
import json
import os
import sys

import numpy as np

from .evaluate import evaluate
from .saved_nets import saved_nets

from typing import Dict, Union

history_file = 'file_history.json'


def analyze(base_dir, multi_trial, check_best=False):
    experiment_dirs = saved_nets(base_dir)

    if multi_trial:
        # Map experiment_dirs to parent directories

        # Dirs containing individual trial runs
        trial_dirs = experiment_dirs[:]
        # Assign a list of trial dirs to each parent experiment dir
        experiment_dirs = {}
        for d in trial_dirs:
            parent_dir = os.path.dirname(d)
            if parent_dir in experiment_dirs:
                experiment_dirs[parent_dir].append(d)
            else:
                experiment_dirs[parent_dir] = [d]

        # Build up stats
        stats = []
        for d in experiment_dirs:
            trial_stats = []
            trials = experiment_dirs[d]
            for t in trials:
                if check_best:
                    eval_result = best_eval(t)
                else:
                    with open(os.path.join(t, history_file), 'r') as f:
                        eval_history = json.load(f)
                    eval_result = eval_history[-1]
                trial_stats.append(eval_result)
                n_params = np.array([ts['n_trainable_params']
                                     for ts in trial_stats])
                mious = np.array([ts['mean_iou'] for ts in trial_stats])
                aris = np.array([ts['adj_rand_idx'] for ts in trial_stats])
                exp_stats = {'n_params': {'min': n_params.min(),
                                          'max': n_params.max(),
                                          'mean': n_params.mean()},
                             'mious': {'min': mious.min(),
                                       'max': mious.max(),
                                       'mean': mious.mean()},
                             'aris': {'min': aris.min(),
                                      'max': aris.max(),
                                      'mean': aris.mean()},
                             'n_trials': len(trial_stats),
                             'experiment_name': os.path.relpath(d),
                             'full_stats': trial_stats}
                stats.append(exp_stats)

        # Sort stats by mean mean_iou
        stats = sorted(stats, key=lambda k: k['mious']['mean'])

        # Print stats summary
        summary = '\n'.join(['{} ({} trials)\n'
                             '  # params: min {}, max {}, mean {}\n'
                             '  mean iou: min {}, max {}, mean {}\n'
                             '  ari: min {}, max {}, mean {}'.format(
            s['experiment_name'], s['n_trials'],
            s['n_params']['min'], s['n_params']['max'],
            s['n_params']['mean'], s['mious']['min'],
            s['mious']['max'], s['mious']['mean'],
            s['aris']['min'], s['aris']['max'],
            s['aris']['mean']) for s in stats])
        # print(summary)
        return stats, summary

    else:
        # Not multi-trial mode
        stats = []

        for i, d in enumerate(experiment_dirs):
            try:
                if check_best:
                    eval_result = best_eval(d)
                else:
                    # Just get the last part of the dir path
                    with open(os.path.join(d, history_file), 'r') as fl:
                        eval_history = json.load(fl)
                    eval_result = eval_history[-1]
                eval_result['experiment_name'] = os.path.relpath(d)
                stats.append(eval_result)
            except:
                print(f'Ignoring {d} after error: {sys.exc_info()[0]}')


        # Sort stats by mean_iou
        stats = sorted(stats, key=lambda k: k['mean_iou'])

        # Print stats summary
        summary = '\n'.join(['{} ({} params): mean_iou = {}, ari = {}'.format(
            s['experiment_name'], s['n_trainable_params'], s['mean_iou'],
            s['adj_rand_idx']) for s in stats])
        # print(summary)
        return stats, summary


def best_eval(d: str) -> Dict[str, Union[float, str]]:
    with open(os.path.join(d, history_file), 'r') as f:
        eval_history = json.load(f)
    best_ckpt_dir = os.path.join(d, 'model', 'checkpoints', 'best')
    if not os.path.exists(best_ckpt_dir):
        print(f'Found no \'best\' checkpoint in {d}, '
              f'returning most recent eval data.')
        return eval_history[-1]

    ckpt_index = [f for f in os.listdir(best_ckpt_dir)
                  if os.path.splitext(f)[1] == '.index'][0]
    ckpt_iteration = int(ckpt_index.split('.')[-2].split('-')[1])

    best_eval = [h for h in eval_history
                 if h['global_step'] == ckpt_iteration][0]

    return best_eval


# def best_eval(d: str) -> Dict[str, Union[float, str]]:
#     best_ckpt_dir = os.path.join(d, 'model', 'checkpoints', 'best')
#     if not os.path.exists(best_ckpt_dir):
#         print(f'Found no \'best\' checkpoint in {d}, '
#               f'no checkpoint load performed.')
#         best_ckpt_dir = None
#
#     # TODO: Much less hacky way of associating training data
#     #  links with checkpoints
#     trial_log = [os.path.join(d, f) for f in os.listdir(d)
#                  if os.path.splitext(f)[1] == '.log'][0]
#     data_dir = eval_data_dir_from_log(trial_log)
#     image_file = os.path.join(data_dir, 'eval-volume.tif')
#     label_file = os.path.join(data_dir, 'eval-label-volume.tif')
#     eval_result = evaluate(
#         net_dir=d,
#         image_file=image_file,
#         label_file=label_file,
#         checkpoint_dir=best_ckpt_dir)
#     return eval_result


def eval_data_dir_from_log(trial_log: str) -> str:
    """Parse the eval data dir used for a GeneNet training trial's log file,
    subject to a bunch of assumptions about the structure of the log that may
    not always hold true.

    Args:
        trial_log (str): Log to parse.

    Returns:
        (str): The directory containing the data files used during training.

    """
    eval_load_line = None
    with open(trial_log, 'r') as fd:
        # Scan log lines until we find one about loading eval data
        for line in fd:
            if 'INFO: Loaded eval data from' in line:
                eval_load_line = line
                break

    eval_data_dir = eval_load_line.split(' ')[-1].rstrip()

    trial_log_path_parts = os.path.normpath(trial_log).split(os.sep)
    output_folder_idx = trial_log_path_parts.index('output')
    experiment_dir = os.path.join(*trial_log_path_parts[:output_folder_idx])
    eval_data_path = os.path.abspath(os.path.join(experiment_dir,
                                                  eval_data_dir))
    return eval_data_path
