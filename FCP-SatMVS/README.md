# FCP-SatMVS

基于级联 MVS 框架的卫星多视立体（Satellite MVS）深度估计网络。
本仓库包含完整的网络结构、损失函数、训练 / 预测脚本以及核心算法模块，
在标准级联 MVSNet 基础上引入了两个核心改进：

- **CDAM**（Cross-Dimensional Attention Module）：跨维度注意力模块。
  沿 H、W 两个维度分别建模注意力，再通过广播相乘进行融合，得到
  逐像素的注意力图，用于增强特征。
- **PUIP**（Pixel-wise Unequal Interval Partition）：按像素的不等
  间隔深度采样模块。在级联结构的后续 stage 中，根据上一阶段的概率体
  与深度采样值，针对每个像素自适应地构造非均匀深度采样区间。
- **FFE**（Fusion Feature Extractor）：跨 stage 多尺度特征融合提取器。
  把 FeatureNet 输出的三个尺度特征按当前 stage 进行通道与尺寸对齐，
  并在融合后引入 CDAM 增强。

## 项目结构

```
FCP-SatMVS/
├── environment.yml           # conda 环境
├── train.py                  # 训练脚本
├── predict.py                # 预测脚本
├── README.md
├── networks/
│   ├── casmvs.py             # CascadeMVSNet 主网络（含 FFE / PUIP / CDAM 调用）
│   └── loss.py               # cas_mvsnet_loss
├── modules/
│   ├── module.py             # FeatureNet / CostRegNet / 基础卷积块等
│   ├── warping.py            # 单应 / RPC warping
│   ├── CDAM.py               # Cross-Dimensional Attention Module
│   ├── PUIP.py               # Pixel-wise Unequal Interval Partition
│   └── FFE.py                # Fusion Feature Extractor
└── tools/
    └── utils.py              # 训练 / 预测的常用工具函数
```

## 数据集说明

仓库中 **不包含数据集加载脚本**，需要自行准备一个 `dataset` Python 包，
向 `find_dataset_def` 注册数据集。期望接口如下：

```python
from dataset import find_dataset_def           # 返回数据集类
from dataset.data_io import save_pfm           # 用于保存 .pfm

MVSDataset = find_dataset_def(geo_model)        # geo_model in {"rpc", "pinhole"}
ds = MVSDataset(root, mode, view_num, ref_view, use_qc)
# mode 取值: "train" / "test" / "pred"
```

每个样本 dict 至少包含以下键：

| 键              | 说明                                                       |
| --------------- | ---------------------------------------------------------- |
| `imgs`          | `[N, C, H, W]` 多视角影像                                  |
| `cam_para`      | 相机/RPC 参数（按 stage 划分的 dict）                      |
| `depth_values`  | 初始深度值采样（stage1 用）                                |
| `depth`         | 多 stage 深度真值 dict（仅训练 / 测试需要）                |
| `mask`          | 多 stage 有效掩膜 dict（仅训练 / 测试需要）                |
| `out_view`      | 输出子目录名                                               |
| `out_name`      | 输出文件名                                                 |

数据集目录约定（与脚本默认路径一致）：

```
<dataset_root>/
└── open_dataset_<geo_model>/
    ├── train/
    └── test/
```

## 环境配置

```bash
conda env create -f environment.yml
conda activate satmvs
```

主要依赖：

- Python 3.7+
- PyTorch ≥ 1.8（PUIP 使用了 `torch.nan` 字面量）
- tensorboardX、matplotlib、numpy、torchvision

## 关于输入图像尺寸

`FFE` 内部的 `CDAM_Block` 使用 `AdaptiveAvgPool2d((h, 1))` 与 `nn.Linear(input_size, 1)`，
其中 `h`、`w` 在构造模型时就已固定（默认 384×768，对应 stage1=96×192、
stage2=192×384、stage3=384×768）。**训练和推理时的输入图像尺寸必须一致**，
否则注意力层会因尺寸不匹配而报错。如需更换尺寸，请在构造 `CascadeMVSNet`
时显式传入：

```python
CascadeMVSNet(..., img_h=H, img_w=W)
```

## 关于 checkpoint 兼容性

本次整理把原有的内联 `stageN_fusion = nn.Sequential(...)` 统一封装为 `FFE`
模块。因此 state_dict 的 key 由

```
module.stageN_fusion.X.weight
```

变为

```
module.ffe.stageN_fusion.X.weight
```

旧版 ckpt 与新代码 **不能直接互相加载**。若需迁移旧权重，请在加载时手工
重命名 key：

```python
state_dict = torch.load(old_ckpt)
new_sd = {}
for k, v in state_dict['model'].items():
    new_sd[k.replace('module.stage', 'module.ffe.stage')] = v
state_dict['model'] = new_sd
```



## 训练

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

常用参数：

| 参数               | 含义                                                         |
| ------------------ | ------------------------------------------------------------ |
| `--mode`           | `train` / `test` / `profile`                                 |
| `--geo_model`      | 几何模型：`rpc`（RPC 卫星）或 `pinhole`（针孔）              |
| `--use_qc`         | RPC warping 是否使用 Quaternary Cubic Form                  |
| `--ndepths`        | 各 stage 深度采样数（三个 stage）                            |
| `--depth_inter_r`  | 各 stage 相对最底层 stage 的深度间隔比例                     |
| `--min_interval`   | 最底层 stage 的最小深度间隔（单位与数据相同）                |
| `--dlossw`         | 各 stage 损失权重                                            |
| `--resume`         | 从 `--logdir` 下最新 ckpt 恢复训练                           |
| `--loadckpt`       | 加载指定 ckpt 路径                                           |

训练日志会写入 `--logdir/<model>/<geo_model>`，并通过 TensorBoard 可视化：

```bash
tensorboard --logdir ./checkpoints/casmvs/pinhole
```

## 预测

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

预测输出目录结构：

```
<output_dir>/
└── <out_view>/
    ├── init/<out_name>.pfm           # 估计深度（启用 --save_pfm）
    ├── init/color/<out_name>.png     # 深度可视化（启用 --save_png）
    ├── prob/<out_name>.pfm           # 置信度
    └── prob/color/<out_name>.png     # 置信度可视化
```

注意：

- `--dataset_root` 与 `--loadckpt` 中的 `geo_model` 字符串需一致（脚本会做断言）。
- `pinhole` 几何模型下，可视化前会做 `max - depth` 反转处理（保持与原始实现一致）。

## 关键模块说明

### CDAM（`modules/CDAM.py`）
- `LAM(channel, input_size)`：在单个维度上做线性注意力。
- `CDAM_Block(channel, h, w)`：在 H、W 两个维度做 LAM 并外积融合，
  最终得到 `[B, C, h, w]` 的注意力图。

### PUIP（`modules/PUIP.py`）
- `PUIP.sample(cur_depth, ndepth, depth_interval, shape, prob_volume_prev, depth_values_prev)`：
  按算法计算 `Hmin / Hmax`，并基于像素邻域梯度做非均匀采样，输出
  `[B, ndepth, H, W]` 的深度采样体。
- `get_depth_range_samples(...)`：统一入口；stage1 走均匀采样，
  stage2/3 调用 PUIP。

### FFE（`modules/FFE.py`）
- `FFE(img_h, img_w)`：构建三个 stage 的融合分支，每个分支由
  「身份映射 + 两路尺度对齐 + BN + ReLU + 膨胀卷积身份映射 + CDAM」组成。
- `forward(feat1, feat2, feat3, cur_stage)`：根据 `cur_stage` 选择
  对应分支输出融合特征。

## 引用与致谢

本项目在 SatMVS / CasMVSNet / UCS-Net 等工作的基础上完成，
感谢相关开源项目的贡献。
