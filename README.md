# FCP-SatMVS

A cascade multi-view stereo (MVS) network for satellite imagery, augmented with three novel components:

- **CDAM** (Cross-Dimensional Attention Module): attention is modeled along
  the H and W axes independently, then combined via an outer product and
  normalized by softmax to yield a per-pixel attention map.
- **PUIP** (Pixel-wise Unequal Interval Partition): in later cascade stages,
  a per-pixel non-uniform depth sampling interval is constructed from the
  probability volume and depth hypotheses produced by the previous stage.
- **FFE** (Fusion Feature Extractor): cross-stage multi-scale feature fusion.
  The three FeatureNet outputs are aligned in channel and spatial size for
  the current stage, then refined by CDAM.

## Repository Layout

```
FCP-SatMVS/
├── environment.yml           # conda environment
├── train.py                  # training entry
├── predict.py                # inference entry
├── README.md
├── networks/
│   ├── casmvs.py             # CascadeMVSNet (uses FFE / PUIP / CDAM)
│   └── loss.py               # cas_mvsnet_loss
├── modules/
│   ├── module.py             # FeatureNet / CostRegNet / basic blocks
│   ├── warping.py            # homography / RPC warping
│   ├── CDAM.py               # Cross-Dimensional Attention Module
│   ├── PUIP.py               # Pixel-wise Unequal Interval Partition
│   └── FFE.py                # Fusion Feature Extractor
└── tools/
    └── utils.py              # shared utilities for train / predict
```

## Dataset Interface

This repository **does not** ship a dataset loader. Users are expected to
provide a `dataset` package exposing the following interface:

```python
from dataset import find_dataset_def           # returns a dataset class
from dataset.data_io import save_pfm           # used to save .pfm files

MVSDataset = find_dataset_def(geo_model)        # geo_model in {"rpc", "pinhole"}
ds = MVSDataset(root, mode, view_num, ref_view, use_qc)
# mode is one of {"train", "test", "pred"}
```

Each sample dict must contain at least the following keys:

| Key            | Description                                              |
| -------------- | -------------------------------------------------------- |
| `imgs`         | `[N, C, H, W]` multi-view images                         |
| `cam_para`     | camera or RPC parameters (a dict keyed by stage)         |
| `depth_values` | initial depth hypotheses for stage 1                     |
| `depth`        | per-stage ground-truth depth dict (train / test only)    |
| `mask`         | per-stage validity mask dict (train / test only)         |
| `out_view`     | sub-directory name used for output                       |
| `out_name`     | file stem used for output                                |

The expected dataset directory layout matches the default script paths:

```
<dataset_root>/
└── open_dataset_<geo_model>/
    ├── train/
    └── test/
```

## Environment

```bash
conda env create -f environment.yml
conda activate satmvs
```

Core requirements:

- Python ≥ 3.7
- PyTorch ≥ 1.8 (PUIP relies on the `torch.nan` literal)
- tensorboardX, matplotlib, numpy, torchvision

## Input Image Size

`FFE` instantiates three `CDAM_Block`s whose internal
`AdaptiveAvgPool2d((h, 1))` and `nn.Linear(input_size, 1)` layers fix `h`
and `w` at construction time. The default is `384 × 768`, corresponding to
stage-1 `96 × 192`, stage-2 `192 × 384`, stage-3 `384 × 768`.

**Training and inference must use the same input size**, otherwise the
attention layers will raise a shape mismatch. To use a different size,
pass it explicitly when building the model:

```python
CascadeMVSNet(..., img_h=H, img_w=W)
```

## Checkpoint Compatibility

In this release the previously inline `stageN_fusion = nn.Sequential(...)`
blocks have been refactored into the `FFE` module. As a result the
`state_dict` keys change from

```
module.stageN_fusion.X.weight
```

to

```
module.ffe.stageN_fusion.X.weight
```

Old checkpoints are therefore **not** directly compatible with the new
code. To port pre-existing weights, rename the keys before loading:

```python
state_dict = torch.load(old_ckpt)
new_sd = {}
for k, v in state_dict['model'].items():
    new_sd[k.replace('module.stage', 'module.ffe.stage')] = v
state_dict['model'] = new_sd
```

## Training

```bash
python train.py \
  --mode train \
  --model casmvs \
  --geo_model pinhole \
  --dataset_root /path/to/WHU-TLC \
  --view_num 3 \
  --ref_view 2 \
  --batch_size 1 \
  --ndepths 64,32,8 \
  --depth_inter_r 4,2,1 \
  --min_interval 2.5 \
  --dlossw 0.5,1.0,2.0 \
  --cr_base_chs 8,8,8 \
  --epochs 30 \
  --lr 0.001 \
  --lrepochs 10,12,14:2 \
  --logdir ./checkpoints \
  --gpu_id 0
```

Key arguments:

| Argument           | Meaning                                                         |
| ------------------ | --------------------------------------------------------------- |
| `--mode`           | `train` / `test` / `profile`                                    |
| `--geo_model`      | geometry model: `rpc` (satellite RPC) or `pinhole`              |
| `--use_qc`         | use Quaternary Cubic Form for RPC warping                       |
| `--ndepths`        | number of depth hypotheses per stage (three stages)             |
| `--depth_inter_r`  | per-stage depth-interval ratio (relative to the finest stage)   |
| `--min_interval`   | minimum depth interval of the finest stage                      |
| `--dlossw`         | per-stage loss weights                                          |
| `--resume`         | resume training from the latest ckpt under `--logdir`           |
| `--loadckpt`       | load a specific checkpoint path                                 |

Logs are written to `--logdir/<model>/<geo_model>` and can be visualized
with TensorBoard:

```bash
tensorboard --logdir ./checkpoints/casmvs/pinhole
```

## Inference

```bash
python predict.py \
  --model casmvs \
  --geo_model rpc \
  --dataset_root /path/to/open_dataset_rpc/test \
  --loadckpt /path/to/checkpoints/casmvs/rpc/model_000006.ckpt \
  --view_num 3 \
  --ref_view 2 \
  --batch_size 1 \
  --ndepths 64,32,8 \
  --depth_inter_r 4,2,1 \
  --min_interval 2.5 \
  --cr_base_chs 8,8,8 \
  --output_dir ./mvs_results \
  --save_pfm \
  --save_png \
  --gpu_id 0
```

Output directory layout:

```
<output_dir>/
└── <out_view>/
    ├── init/<out_name>.pfm           # estimated depth (--save_pfm)
    ├── init/color/<out_name>.png     # depth visualization (--save_png)
    ├── prob/<out_name>.pfm           # confidence
    └── prob/color/<out_name>.png     # confidence visualization
```

Notes:

- The `geo_model` substring must appear in both `--dataset_root` and
  `--loadckpt` (enforced by `assert`).
- For `pinhole`, the depth map is reversed via `max - depth` before
  visualization, matching the original implementation.

## Module Reference

### CDAM (`modules/CDAM.py`)
- `CAM(channel, input_size)`: linear attention along a single axis.
- `CDAM_Block(channel, h, w)`: combines two CAMs along H and W and
  produces a `[B, C, h, w]` attention map.

### PUIP (`modules/PUIP.py`)
- `PUIP.sample(cur_depth, ndepth, depth_interval, shape, prob_volume_prev, depth_values_prev)`:
  computes `Hmin / Hmax` from the previous-stage probability volume,
  derives per-pixel sampling counts from the local depth gradient, and
  returns a `[B, ndepth, H, W]` depth-hypothesis volume.
- `get_depth_range_samples(...)`: unified entry; uniform sampling for
  stage 1, PUIP for later stages.

### FFE (`modules/FFE.py`)
- `FFE(img_h, img_w)`: builds three fusion branches; each branch
  performs identity mapping (primary) + two scale-aligned auxiliary
  paths + BN + dilated identity mapping + CDAM.
- `forward(feat1, feat2, feat3, cur_stage)`: selects the branch
  corresponding to `cur_stage` and returns the fused feature.

## Acknowledgements

This work builds upon SatMVS, CasMVSNet and UCS-Net. We thank the
authors of those projects for releasing their code.
