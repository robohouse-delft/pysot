# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch


logger = logging.getLogger('global')


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # filter 'num_batches_tracked'
    missing_keys = [x for x in missing_keys
                    if not x.endswith('num_batches_tracked')]
    if len(missing_keys) > 0:
        logger.info('[Warning] missing keys: {}'.format(missing_keys))
        logger.info('missing keys:{}'.format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.info('[Warning] unused_pretrained_keys: {}'.format(
            unused_pretrained_keys))
        logger.info('unused checkpoint keys:{}'.format(
            len(unused_pretrained_keys)))
    logger.info('used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, \
        'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters
    share common prefix 'module.' '''
    logger.info('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info('load pretrained model from {}'.format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(pretrained_path,
        map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'],
                                        'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')

    # try:
    #     check_keys(model, pretrained_dict)
    # except:
    #     logger.info('[Warning]: using pretrain as features.\
    #             Adding "features." as prefix')
    #     new_dict = {}
    #     for k, v in pretrained_dict.items():
    #         k = 'features.' + k
    #         new_dict[k] = v
    #     pretrained_dict = new_dict
    #     check_keys(model, pretrained_dict)
    # model.load_state_dict(pretrained_dict, strict=False)
    # Load the pre-trained weights
	# pretrained_state_dict = torch.load(args.input_model, map_location="cpu")

    # Create a new state dict with modified keys
    new_state_dict = {}

    for k, v in pretrained_dict.items():
        new_k = k  # Default to original name

        # Convert "neck.downsample2.*" to "neck.downsample.downsample2.*"
        if new_k.startswith("neck.downsample"):
            parts = new_k.split(".")
            if len(parts) > 2:  # Ensure downsample is inserted
                new_k = ".".join([parts[0], "downsample"] + parts[1:])

        # Convert "rpn_head.rpn2.*" to "rpn_head.rpn.rpn2.*"
        if new_k.startswith("rpn_head.rpn"):
            parts = new_k.split(".")
            if len(parts) > 2:  # Ensure rpn is inserted
                new_k = ".".join([parts[0], "rpn"] + parts[1:])

        new_state_dict[new_k] = v  # Store renamed key

        # Debugging: Print renamed keys
        if k != new_k:
            print(f"Renamed: {k} → {new_k}")
    
    # Load your model
    model_state_dict = model.state_dict()

    # Only keep keys that match the model's state dict
    filtered_state_dict = {k: v for k, v in new_state_dict.items() if k in model_state_dict}

    check_keys(model, filtered_state_dict)

    # Load the corrected state dict into the model
    model.load_state_dict(filtered_state_dict)

    print("Pre-trained weights successfully loaded with key modifications.")

    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path,
        map_location=lambda storage, loc: storage.cuda(device))
    epoch = ckpt['epoch']

    ckpt_model_dict = remove_prefix(ckpt['state_dict'], 'module.')
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    check_keys(optimizer, ckpt['optimizer'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, epoch
