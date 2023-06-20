# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved
import rldev

import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
np.set_printoptions(precision=6, linewidth=65536, suppress=True, threshold=np.inf)
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from eval_utils import eval_utils
from mtr.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from mtr.datasets import build_dataloader
from mtr.models import model as model_utils
from mtr.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg




from interpretable_driving.waymo import split_roadline, calculate_intention
from interpretable_driving.waymo.vis import RoadLineVis, AgentVis



class RoadDim(object):
    valid = 0
    x = 1
    y = 2
    z = 3
    dx = 4
    dy = 5
    dz = 6
    type = 7
    # id = 8


class AgentDim(object):
    valid = 0
    time = 1
    x = 2
    y = 3
    z = 4
    bbox_yaw = 5
    length = 6
    width = 7
    height = 8
    speed = 9
    # vel_yaw = 10
    vx = 10
    vy = 11





def visualize(axes, data_point):
    roadline = np.concatenate([
        np.expand_dims(data_point.map_polylines_mask.astype(np.float32), axis=2),
        data_point.map_polylines,
    ], axis=2)
    roadline = roadline.reshape(-1, roadline.shape[-1])


    agents_history = np.concatenate([
        np.expand_dims(data_point.obj_trajs_mask.astype(np.float32), axis=2),
        data_point.obj_trajs,
    ], axis=2)
    agents_future = np.concatenate([
        np.expand_dims(data_point.obj_trajs_future_mask.astype(np.float32), axis=2),
        data_point.obj_trajs_future_state,
    ], axis=2)

    agents_history = np.concatenate([
        agents_history[..., [0]],  ### valid
        agents_history[..., [23]],  ### time
        agents_history[..., [1,2,3]],  ### x, y, z
        np.arctan2(agents_history[..., [24]], agents_history[..., [25]]),   ### bbox_yaw
        agents_history[..., [4,5,6]],  ### length, width, height
        np.hypot(agents_history[..., [26]], agents_history[..., [27]]),   ### speed
        agents_history[..., [26,27]],  ### vx, vy
    ], axis=2)



    ################################################
    ################################################
    ################################################
    ax = axes[0]
    ax.set_title(f'{data_point.index.item()} - {data_point.scenario_id}: roadline')
    RoadLineVis(roadline, RoadDim, verbose=1).visualize(ax, alpha=0.6)


    ################################################
    ################################################
    ################################################
    ax = axes[1]
    ax.set_title(f'{data_point.index.item()}: agents trajectories')
    RoadLineVis(roadline, RoadDim, verbose=0).visualize(ax, alpha=0.6)



    for (
        agent_current,
        agent_past,
        agent_future,
        # agent_to_predict,
        # agent_of_interest,
        agent_id,
        # agent_is_sdc,
        agent_type,
    ) in zip(
        agents_history[:,[-1]],
        agents_history[:,:-1],
        agents_future[...,[0,1,2]],
        # data_point.agents_to_predict,
        # data_point.agents_of_interest,
        data_point.obj_ids,
        # data_point.agents_is_sdc,
        data_point.obj_types,
    ):
        agent_vis = AgentVis(
            agent_id,
            agent_type,
            0, # agent_is_sdc,
            0, # agent_to_predict,
            0, # agent_of_interest,
            agent_current,
            agent_past,
            agent_future,
            AgentDim,
        )
        agent_vis.visualize(ax)


    # import pdb; pdb.set_trace()


    return



def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        if args.local_rank is None:
            args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    # else:
    #     assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
    #     args.batch_size = args.batch_size // total_gpus
          
    output_dir = cfg.ROOT_DIR / 'results' / 'vis'
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    dataset, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    # for i, batch_dict in enumerate(test_loader):
    #     data = rldev.Data(**batch_dict['input_dict'])
    #     import pdb; pdb.set_trace()
        

    for data_dict in dataset:
        data = rldev.Data(**data_dict)
        data.obj_types = np.expand_dims(data.obj_types, axis=0).repeat(data.index.shape[0], axis=0)
        data.obj_ids = np.expand_dims(data.obj_ids, axis=0).repeat(data.index.shape[0], axis=0)

        print()
        print(f'data shape: {data.shape}')
        print()


        ################################################
        ### vis ########################################
        ################################################
        num_rows = 1
        num_columns = 2
        fig, axes = plt.subplots(num_rows, num_columns, dpi=100)
        for ax in axes.flatten():
            ax.set_aspect('equal')

        visualize(axes, data[0:1].squeeze(0))
        
        
        fig.set_tight_layout(True)

        w = (num_columns + (num_columns-1)*fig.subplotpars.wspace) * max([a.get_position().width for a in axes.flatten()])
        h = (num_rows + (num_rows-1)*fig.subplotpars.hspace) * max([a.get_position().height for a in axes.flatten()])
        fig.set_size_inches(w  *50, h  *50)

        plt.show()
        


if __name__ == '__main__':
    main()
