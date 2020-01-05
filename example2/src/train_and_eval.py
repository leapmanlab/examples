"""Train and evaluate a segmentation network.

"""
import logging

import tensorflow as tf

# From _private

import genenet as gn
import leapmanlab as lab

from typing import Any, Dict


def train_and_eval(net: gn.GeneNet,
                   train_info: Dict[str, Any],
                   eval_info: Dict[str, Any],
                   logger: logging.Logger) -> gn.GeneNet:
    gene_graph = net.gene_graph

    data_handler = train_info['data_handler']
    n_epochs = train_info['n_epochs']
    max_steps = train_info['max_steps']
    stop_criterion = train_info['stop_criterion']
    data_seed = train_info['data_seed']
    window_spacing = train_info['window_spacing']

    n_its_between_evals = eval_info['n_its_between_evals']
    instance_dir = eval_info['instance_dir']
    eval_media_dir = eval_info['eval_media_dir']
    sample_image = eval_info['sample_image']
    sample_label = eval_info['sample_label']
    image_cmaps = eval_info['image_cmaps']

    # Evaluation input function
    eval_input_fn = data_handler.input_fn(
        mode=tf.estimator.ModeKeys.EVAL,
        graph_source=gene_graph)

    logger.debug('Beginning train-eval loop')

    # Track iterations since last eval run
    its_since_last_eval = 0
    its_per_epoch = data_handler.n_samples_per_epoch(
        net,
        window_spacing)

    # Train for `n_epochs` or until an early stopping criterion is met
    stop_training = False
    # Track the current epoch
    epoch = -1
    best_miou_so_far = -1
    while not stop_training:
        epoch += 1
        # Use a sequence of random seeds
        train_seed = (data_seed + epoch) % 2 ** 32
        # Training input function
        train_input_fn = data_handler.input_fn(
            mode=tf.estimator.ModeKeys.TRAIN,
            graph_source=gene_graph,
            train_window_spacing=window_spacing,
            num_epochs=1,
            random_seed=train_seed)

        logger.debug('Created training input_fn')

        # Train
        net.train(train_input_fn,
                  max_steps=max_steps)

        logger.debug(f'Finished training epoch {epoch}')

        # Only run eval every so many epochs when epoch batches are small
        its_since_last_eval += its_per_epoch
        if its_since_last_eval >= n_its_between_evals:
            its_since_last_eval -= n_its_between_evals

            # Evaluate
            eval_results = net.evaluate(eval_input_fn)
            # Save most recent checkpoint each eval
            lab.save_checkpoint('last', instance_dir)

            if eval_results['mean_iou'] > best_miou_so_far:
                best_miou_so_far = eval_results['mean_iou']
                lab.save_checkpoint('best', instance_dir)

            logger.info(f'Evaluation epoch {epoch} results:\n {eval_results}')

            # Make snapshot images
            gn.segment(net,
                       data_handler=data_handler,
                       data_vol=sample_image,
                       label_vol=sample_label,
                       save_dir=eval_media_dir,
                       image_cmaps=image_cmaps,
                       save_name=f'{eval_results["global_step"]}')

            # Check whether to stop training
            stop_training = lab.stop_check(eval_results,
                                           stop_criterion,
                                           n_epochs,
                                           epoch)

    # One last checkpoint
    lab.save_checkpoint('last', instance_dir)

    return net
