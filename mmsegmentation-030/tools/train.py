# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import sys
sys.path.append('/home/br/workspace/GRIC/mmsegmentation-030')
# sys.path.append('/home/br/mmcv-1.7.1')

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)




def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    # # 提取出实验名称作为work_dir
    output_dir = '/home/br/workspace/GRIC/output'
    args.work_dir = f"{output_dir}/{args.config.split('/')[-1].split('_')[0]}"
    kw_list = args.config.split('/')[-1].split('_')
    fold = [i.strip("fold") for i in kw_list if 'fold' in i][0]
    print("work_dir:", args.work_dir)
    print("fold:", fold)
    

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from is not None:
        cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    cfg.device = get_device()
    seed = init_random_seed(args.seed, device=cfg.device)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        model = revert_sync_batchnorm(model)

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
    
    return args.work_dir, args.gpu_id, fold




def get_log_path(work_dir):
    import glob
    jsonfiles = glob.glob(f"{work_dir}/*.json")
    json_path = max(jsonfiles)

    logfiles = glob.glob(f"{work_dir}/*.log")
    log_path = max(logfiles)

    return json_path, log_path


def parse_json(json_path):
    import json
    import pandas as pd
    with open(json_path, 'r') as f:
        logs = f.readlines()

    # 将每一行的json转换为字典
    train_list = []
    valid_list = []
    for log in logs:
        log = json.loads(log)
        if log.get('mode') and log.get('mode') == 'train':
            train_list.append(log)
        elif log.get('mode') and log.get('mode') == 'val':
            valid_list.append(log)

    # 将字典转换为DataFrame
    train_df = pd.DataFrame(train_list)
    valid_df = pd.DataFrame(valid_list)
    return train_df, valid_df

def dice_score(y_p, y_t, smooth=1e-6):
    '''
    Args:
        y_p: [bs, 1, 256, 256]
        y_t: [bs, 1, 256, 256]
    Return:
        是所有图片(bs)像素汇总到后计算出的Dice score
    '''
    i = torch.sum(y_p * y_t, dim=(0, 2, 3))
    u = torch.sum(y_p, dim=(0, 2, 3)) + torch.sum(y_t, dim=(0, 2, 3))
    score = (2 * i+smooth)/(u + smooth)
    return torch.mean(score)

def infer(work_dir, gpu_id, best_epoch, fold=0):
    # 读取配置文件和模型
    cfg = max(glob.glob(f'{work_dir}/*.py'))
    ckpt = f'{work_dir}/epoch_{best_epoch}.pth'

    cfg = config.Config.fromfile(cfg)
    cfg.model.test_cfg.return_logits = True
    model = init_segmentor(cfg, ckpt, device=f'cuda:{gpu_id}')

    # 图片和gt路径
    image_dir = '/home/br/workspace/GRIC/input/mmseg_data/images'
    gt_dir = '/home/br/workspace/GRIC/input/mmseg_data/labels'
    with open(f'/home/br/workspace/GRIC/input/mmseg_data/splits/holdout_{fold}.txt', 'r') as f:
        holdout = f.read().splitlines()
    
    # 开始预测，获取preds
    preds = []
    for img_id in tqdm(holdout):
        img  = np.load(f'{image_dir}/{img_id}.npy')
        pred = inference_segmentor(model, img)[0]
        preds.append(pred)

    preds = torch.tensor(preds)
    preds = preds.unsqueeze(1)

    # 读取gts
    gts = []
    for img_id in tqdm(holdout):
        # read grayscale image
        img  = cv2.imread(f'{gt_dir}/{img_id}.png', cv2.IMREAD_GRAYSCALE)
        gts.append(img)
    gts = torch.tensor(gts)
    gts = gts.unsqueeze(1)

    # 计算dice score
    scores = []

    thr_list = np.arange(0.2, 1.00, 0.05)
    for thr in thr_list:
        preds_binary = preds >= thr 
        # preds_binary = preds_binary.squeeze(1)
        score = dice_score(preds_binary, gts)
        scores.append(score.item())

    print(f"best score: {max(scores)}, best threshold: {thr_list[np.argmax(scores)]:.2f}")
    plt.plot(thr_list, scores)
    # save figure
    plt.savefig(f'{work_dir}/thr_score.png')

    return thr_list, scores


if __name__ == '__main__':
    # main()
    work_dir, gpu_id, fold = main()

    from mmseg.apis import init_segmentor, inference_segmentor
    from mmcv.utils import config
    import cv2
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    import glob
    import gc
    import matplotlib.pyplot as plt
    # ======= 从实验日志提取出实验结果 =======
    # 获取日志路径
    json_path, log_path = get_log_path(work_dir)
    train_df, valid_df = parse_json(json_path)
    # {"mode": "val", "epoch": 25, "iter": 4106, "lr": 1e-05, "aAcc": 0.9896, "mDice": 0.7444, "mAcc": 0.9475, "Dice.BG": 0.9947, "Dice.contrails": 0.4941, "Acc.BG": 0.9901, "Acc.contrails": 0.9049}
    valid_df = valid_df[["mode", "epoch", "iter", "lr", "mDice", "Dice.BG", "Dice.contrails"]]
    # 写入验证集结果
    with open(f"{work_dir}/result.txt", 'w') as f:
        f.write(valid_df.to_string())

    best_row = valid_df.iloc[valid_df["Dice.contrails"].argmax()]
    best_epoch = best_row["epoch"]

    # ======= 从实验结果获取最佳模型得分和阈值 =======
    thr_list, scores = infer(work_dir, gpu_id, best_epoch, fold)
    with open(f"{work_dir}/result.txt", 'a+') as f:
        for thr, score in zip(thr_list, scores):
            f.write(f"threshold: {thr:.2f}, score: {score}\n")
        best_score = max(scores)
        best_thr = thr_list[np.argmax(scores)]
        f.write(f"best threshold: {best_thr:.2f}, best score: {best_score}\n")


