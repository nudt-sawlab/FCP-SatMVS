"""FCP-SatMVS inference entry point.

Dataset note
------------
This repository does not ship a dataset loader. Users must provide a
``dataset`` package whose signature matches the one used in ``train.py``.

Each prediction sample dict must contain ``imgs``, ``cam_para``,
``depth_values``, ``out_view`` and ``out_name``.
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import find_dataset_def
from dataset.data_io import save_pfm
from networks.casmvs import CascadeMVSNet
from tools.utils import (
    DictAverageMeter, make_nograd_func, print_args, tensor2numpy, tocuda,
)

cudnn.benchmark = True


parser = argparse.ArgumentParser(description='FCP-SatMVS Prediction')
parser.add_argument('--model', default="casmvs", choices=["casmvs"], help='select model')
parser.add_argument('--geo_model', default="rpc", choices=["rpc", "pinhole"], help='geometry model')
parser.add_argument('--use_qc', default=False, help='whether to use Quaternary Cubic Form for RPC warping.')
parser.add_argument('--dataset_root', default="", help='dataset root')
parser.add_argument('--loadckpt', default="", help='checkpoint path')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images.')
parser.add_argument('--ref_view', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=1, help='predict batch size')

# Cascade parameters
parser.add_argument('--ndepths', type=str, default="64,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=2.5, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
parser.add_argument('--gpu_id', type=str, default="0")

# output directory
parser.add_argument('--output_dir', type=str, default="mvs_results", help='output directory')
parser.add_argument('--save_pfm', action='store_true', help='whether to save pfm files')
parser.add_argument('--save_png', action='store_true', help='whether to save png visualizations')

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# Mandatory arguments
assert args.dataset_root, "Please specify the dataset root via --dataset_root"
assert args.loadckpt, "Please specify the checkpoint path via --loadckpt"

# Consistency checks
assert args.geo_model in args.dataset_root, "--dataset_root must contain the geo_model substring"
assert args.geo_model in args.loadckpt, "--loadckpt must contain the geo_model substring"
assert args.model in args.loadckpt, "--loadckpt must contain the model substring"


def predict():
    print("argv:", sys.argv[1:])
    print_args(args)

    # dataset, dataloader
    MVSDataset = find_dataset_def(args.geo_model)
    pre_dataset = MVSDataset(
        args.dataset_root, "pred", args.view_num,
        ref_view=args.ref_view, use_qc=args.use_qc,
    )
    Pre_ImgLoader = DataLoader(
        pre_dataset, args.batch_size, shuffle=False, num_workers=0, drop_last=False,
    )

    model = CascadeMVSNet(
        min_interval=args.min_interval,
        ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
        geo_model=args.geo_model,
        use_qc=args.use_qc,
    )
    print("===============> Model: Cascade MVS Net (FCP-SatMVS) ===========>")

    model = nn.DataParallel(model)
    model.cuda()

    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # output folder
    output_folder = args.output_dir
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    avg_test_scalars = DictAverageMeter()
    t0 = time.time()

    for batch_idx, sample in enumerate(Pre_ImgLoader):
        bview = sample['out_view'][0]
        bname = sample['out_name'][0]

        start_time = time.time()
        image_outputs = predict_sample(model, sample)
        print("Iter {}/{}, {}, time = {:3f}".format(
            batch_idx, len(Pre_ImgLoader), bname, time.time() - start_time))

        # save results
        depth_est = np.float32(np.squeeze(tensor2numpy(image_outputs["depth_est"])))
        prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))

        # paths
        output_folder2 = os.path.join(output_folder, str(bview))
        for sub in ['prob', 'init', 'prob/color', 'init/color']:
            sub_path = os.path.join(output_folder2, sub)
            if not os.path.exists(sub_path):
                os.makedirs(sub_path)

        init_depth_map_path = os.path.join(output_folder2, 'init', '{}.pfm'.format(bname))
        prob_map_path = os.path.join(output_folder2, 'prob', '{}.pfm'.format(bname))

        if args.save_pfm:
            save_pfm(init_depth_map_path, depth_est)
            save_pfm(prob_map_path, prob)

        if args.geo_model == "pinhole":
            depth_est = np.max(depth_est) - depth_est

        if args.save_png:
            depth_png = os.path.join(output_folder2, 'init', 'color', '{}.png'.format(bname))
            prob_png = os.path.join(output_folder2, 'prob', 'color', '{}.png'.format(bname))
            plt.imsave(depth_png, depth_est, format='png')
            plt.imsave(prob_png, prob, format='png')

        del image_outputs

    print("final, time = {:3f}, test results = {}".format(time.time() - t0, avg_test_scalars.mean()))


@make_nograd_func
def predict_sample(model, sample):
    model.eval()

    sample_cuda = tocuda(sample)
    outputs = model(sample_cuda["imgs"], sample_cuda["cam_para"], sample_cuda["depth_values"])
    depth_est = outputs["stage3"]["depth"]
    photometric_confidence = outputs["stage3"]["photometric_confidence"]

    image_outputs = {
        "depth_est": depth_est,
        "photometric_confidence": photometric_confidence,
        "ref_img": sample["imgs"][:, 0],
    }
    return image_outputs


if __name__ == '__main__':
    predict()
