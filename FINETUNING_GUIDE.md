# SAM 3D Objects 微调指南

本指南说明如何基于 SAM 3D Objects 进行微调（Fine-tuning）。

## 许可证说明

根据 SAM License，您可以：
- ✅ 使用、修改和创建衍生作品
- ✅ 进行微调（License 第6行明确包含 "fine-tuning enabling code"）
- ✅ 分发微调后的模型（需遵守相同许可证）
- ⚠️ 必须遵守贸易管制和出口法规
- ⚠️ 需在发布研究成果时注明使用了 SAM Materials

## 架构概述

SAM 3D Objects 使用两阶段生成流程：

### Stage 1: 稀疏结构生成 (Sparse Structure)
- **模型**: `SparseStructureFlowModel` → `ss_generator`
- **输入**: RGBA 图像（RGB + alpha mask）
- **输出**: 3D 体素占用网格（稀疏坐标）
- **训练方法**: Flow Matching

### Stage 2: 结构化潜在生成 (Structured Latent)
- **模型**: `StructuredLatentFlowModel` → `slat_generator`
- **输入**: 图像 + Stage 1 的稀疏坐标
- **输出**: 每个体素的特征向量
- **训练方法**: Flow Matching

### Stage 3: 解码器
- **Gaussian Decoder** (`slat_decoder_gs`): 生成 3D 高斯溅射
- **Mesh Decoder** (`slat_decoder_mesh`): 通过 FlexiCubes 生成三角网格

## 微调策略

### 策略 1：仅微调解码器（推荐用于特定领域）

**适用场景**：
- 改进特定类别物体的网格质量
- 调整高斯溅射的渲染效果
- 添加新的输出格式

**实现步骤**：

```python
import torch
import torch.nn as nn
from sam3d_objects.pipeline.inference_pipeline import InferencePipeline

# 1. 加载预训练模型
config_path = "checkpoints/hf/pipeline.yaml"
pipeline = InferencePipeline(
    # ... 配置参数
)

# 2. 冻结生成器，只训练解码器
for param in pipeline.models["ss_generator"].parameters():
    param.requires_grad = False
for param in pipeline.models["slat_generator"].parameters():
    param.requires_grad = False

# 3. 解冻目标解码器
decoder = pipeline.models["slat_decoder_mesh"]  # 或 slat_decoder_gs
for param in decoder.parameters():
    param.requires_grad = True

# 4. 设置优化器
optimizer = torch.optim.AdamW(
    decoder.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# 5. 训练循环示例
def train_decoder(pipeline, dataloader, optimizer, num_epochs):
    decoder.train()

    for epoch in range(num_epochs):
        for batch in dataloader:
            # 获取 structured latent (冻结梯度)
            with torch.no_grad():
                ss_output = pipeline.sample_sparse_structure(batch)
                coords = ss_output["coords"]
                slat = pipeline.sample_slat(batch, coords)

            # 解码并计算损失
            output = decoder(slat)
            loss = compute_loss(output, batch["ground_truth"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}: Loss = {loss.item()}")
```

### 策略 2：微调 Stage 2 生成器

**适用场景**：
- 改进特定类别的 3D 特征表示
- 提高小物体或遮挡场景的重建质量

**实现步骤**：

```python
# 1. 冻结 Stage 1
for param in pipeline.models["ss_generator"].parameters():
    param.requires_grad = False

# 2. 解冻 Stage 2 生成器
slat_generator = pipeline.models["slat_generator"]
for param in slat_generator.parameters():
    param.requires_grad = True

# 3. 使用 Flow Matching 损失
from sam3d_objects.model.backbone.generator.flow_matching.model import FlowMatching

def train_stage2(pipeline, dataloader, optimizer):
    slat_generator.train()

    for batch in dataloader:
        # 获取 Stage 1 输出（冻结）
        with torch.no_grad():
            ss_output = pipeline.sample_sparse_structure(batch)
            coords = ss_output["coords"]

        # 准备条件输入
        condition_args, condition_kwargs = pipeline.get_condition_input(
            pipeline.condition_embedders["slat_condition_embedder"],
            batch,
            pipeline.slat_condition_input_mapping,
        )
        condition_args += (coords.cpu().numpy(),)

        # 计算 Flow Matching 损失
        # x1 是目标 structured latent (需要从 ground truth 3D 获得)
        x1 = get_target_slat(batch)  # 需要实现

        loss, detail_losses = slat_generator.loss(
            x1,
            *condition_args,
            **condition_kwargs
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 策略 3：端到端微调（需要大量数据）

**适用场景**：
- 完全新的领域（如医学图像、工业零件）
- 有大规模配对数据（图像 + 3D）

**注意**：需要大量计算资源和数据。

## 数据准备

### 数据格式要求

```python
# 每个训练样本应包含：
{
    "image": torch.Tensor,      # [C, H, W], RGB 图像
    "mask": torch.Tensor,       # [1, H, W], 二值掩码
    "ground_truth_3d": dict,    # 可选：ground truth 3D 数据
    # 对于监督训练，您需要：
    "gaussian": GaussianModel,  # 或
    "mesh": trimesh.Trimesh,    # 或
    "coords": torch.Tensor,     # 稀疏坐标 [N, 4]
}
```

### 数据预处理

```python
from sam3d_objects.pipeline import preprocess_utils

# 使用默认预处理器
preprocessor = preprocess_utils.get_default_preprocessor()

# 或自定义预处理
from sam3d_objects.data.dataset.tdfy.preprocessor import PreProcessor
from torchvision import transforms

custom_preprocessor = PreProcessor(
    img_transform=transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    mask_transform=transforms.Compose([
        # 自定义 mask 变换
    ]),
    # ... 其他配置
)
```

### 数据集类示例

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class SAM3DDataset(Dataset):
    def __init__(self, image_paths, mask_paths, preprocessor):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像和掩码
        image = np.array(Image.open(self.image_paths[idx]))
        mask = np.array(Image.open(self.mask_paths[idx]))

        # 合并为 RGBA
        if mask.ndim == 2:
            mask = mask[..., None]
        rgba = np.concatenate([image[..., :3], mask], axis=-1)

        # 预处理
        rgba_tensor = torch.from_numpy(rgba.astype(np.float32) / 255.0)
        rgba_tensor = rgba_tensor.permute(2, 0, 1)

        # 应用预处理器
        # (需要根据 preprocessor 实现适配)

        return {
            "image": rgba_tensor[:3],
            "mask": rgba_tensor[3:4],
        }
```

## 损失函数

### 重建损失（用于解码器）

```python
import torch.nn.functional as F

def gaussian_reconstruction_loss(pred_gs, gt_gs):
    """高斯溅射重建损失"""
    # 位置损失
    pos_loss = F.mse_loss(pred_gs.get_xyz, gt_gs.get_xyz)

    # 缩放损失
    scale_loss = F.mse_loss(pred_gs.get_scaling, gt_gs.get_scaling)

    # 旋转损失
    rot_loss = F.mse_loss(pred_gs.get_rotation, gt_gs.get_rotation)

    # 不透明度损失
    opacity_loss = F.mse_loss(pred_gs.get_opacity, gt_gs.get_opacity)

    # 颜色损失
    color_loss = F.mse_loss(pred_gs.get_features, gt_gs.get_features)

    total_loss = (
        pos_loss +
        0.1 * scale_loss +
        0.1 * rot_loss +
        0.1 * opacity_loss +
        color_loss
    )

    return total_loss

def mesh_reconstruction_loss(pred_mesh, gt_mesh):
    """网格重建损失"""
    # Chamfer 距离
    from pytorch3d.loss import chamfer_distance

    loss_chamfer, _ = chamfer_distance(
        pred_mesh.verts_packed().unsqueeze(0),
        gt_mesh.verts_packed().unsqueeze(0)
    )

    # 法线一致性
    from pytorch3d.loss import mesh_normal_consistency
    loss_normal = mesh_normal_consistency(pred_mesh)

    # 边缘长度正则化
    from pytorch3d.loss import mesh_edge_loss
    loss_edge = mesh_edge_loss(pred_mesh)

    total_loss = loss_chamfer + 0.1 * loss_normal + 0.01 * loss_edge

    return total_loss
```

### Flow Matching 损失（已内置）

```python
# Flow Matching 的损失已在模型中实现
# 见：sam3d_objects/model/backbone/generator/flow_matching/model.py:158-188

# 使用方式：
loss, detail_losses = model.loss(
    x1,  # target
    *condition_args,
    **condition_kwargs
)
```

## 训练配置建议

### 硬件要求

- **最小配置**: 1x A100 (40GB) 用于解码器微调
- **推荐配置**: 4x A100 (80GB) 用于生成器微调
- **完整训练**: 8x H100 (80GB) 用于端到端训练

### 超参数建议

```python
# 解码器微调
decoder_config = {
    "learning_rate": 1e-4,
    "batch_size": 4,  # 每 GPU
    "num_epochs": 50,
    "warmup_steps": 1000,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
}

# Stage 2 生成器微调
stage2_config = {
    "learning_rate": 5e-5,
    "batch_size": 2,
    "num_epochs": 100,
    "warmup_steps": 2000,
    "weight_decay": 0.05,
    "gradient_clip": 0.5,
    "inference_steps": 25,  # 训练时的推理步数
}

# 学习率调度器
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-6
)
```

## 完整训练示例

```python
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_sam3d_finetuning(
    pipeline,
    train_dataset,
    val_dataset,
    config,
    save_dir="checkpoints/finetuned"
):
    # 设置数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    # 选择要微调的组件
    trainable_params = []
    if config["finetune_decoder"]:
        trainable_params += list(pipeline.models["slat_decoder_gs"].parameters())
    if config["finetune_stage2"]:
        trainable_params += list(pipeline.models["slat_generator"].parameters())

    # 优化器
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["num_epochs"]
    )

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(config["num_epochs"]):
        # 训练
        train_loss = train_one_epoch(
            pipeline, train_loader, optimizer, config
        )

        # 验证
        val_loss = validate(
            pipeline, val_loader, config
        )

        # 调度器
        scheduler.step()

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(pipeline, optimizer, epoch, save_dir)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
              f"Val Loss = {val_loss:.4f}")

def train_one_epoch(pipeline, dataloader, optimizer, config):
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        # 前向传播
        if config["finetune_decoder"]:
            # 获取 structured latent（冻结梯度）
            with torch.no_grad():
                ss_output = pipeline.sample_sparse_structure(batch)
                coords = ss_output["coords"]
                slat = pipeline.sample_slat(batch, coords)

            # 解码
            output = pipeline.models["slat_decoder_gs"](slat)
            loss = compute_reconstruction_loss(output, batch)

        elif config["finetune_stage2"]:
            # Stage 2 Flow Matching 训练
            with torch.no_grad():
                ss_output = pipeline.sample_sparse_structure(batch)
                coords = ss_output["coords"]

            # 准备条件
            condition_args, condition_kwargs = pipeline.get_condition_input(
                pipeline.condition_embedders["slat_condition_embedder"],
                batch,
                pipeline.slat_condition_input_mapping,
            )

            # 计算损失
            x1 = get_target_slat(batch)  # 需要实现
            loss, _ = pipeline.models["slat_generator"].loss(
                x1, *condition_args, **condition_kwargs
            )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if config.get("gradient_clip"):
            torch.nn.utils.clip_grad_norm_(
                optimizer.param_groups[0]['params'],
                config["gradient_clip"]
            )

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(pipeline, dataloader, config):
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # 完整推理
            output = pipeline.run(
                batch["image"],
                batch["mask"],
                seed=42
            )

            # 计算验证损失
            loss = compute_reconstruction_loss(output, batch)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def save_checkpoint(pipeline, optimizer, epoch, save_dir):
    import os
    os.makedirs(save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'models': {
            name: model.state_dict()
            for name, model in pipeline.models.items()
        }
    }

    torch.save(
        checkpoint,
        os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
    )
```

## 评估指标

```python
def evaluate_3d_reconstruction(pred, gt):
    """评估 3D 重建质量"""
    metrics = {}

    # Chamfer Distance
    from pytorch3d.loss import chamfer_distance
    cd_loss, _ = chamfer_distance(pred.points, gt.points)
    metrics['chamfer_distance'] = cd_loss.item()

    # PSNR（如果有渲染图）
    if hasattr(pred, 'rendered_image') and hasattr(gt, 'rendered_image'):
        mse = torch.mean((pred.rendered_image - gt.rendered_image) ** 2)
        psnr = 10 * torch.log10(1.0 / mse)
        metrics['psnr'] = psnr.item()

    # LPIPS（感知损失）
    # 需要安装: pip install lpips
    # import lpips
    # lpips_fn = lpips.LPIPS(net='alex')
    # metrics['lpips'] = lpips_fn(pred_img, gt_img).item()

    return metrics
```

## 常见问题

### Q1: 我需要多少训练数据？

- **解码器微调**: 1,000-5,000 个样本
- **Stage 2 微调**: 10,000-50,000 个样本
- **端到端训练**: 100,000+ 个样本

### Q2: 如何处理显存不足？

```python
# 1. 启用梯度检查点
pipeline.models["slat_generator"].use_checkpoint = True

# 2. 减小批次大小
config["batch_size"] = 1

# 3. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = compute_loss(...)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 4. 使用梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q3: 如何获取 ground truth structured latent？

```python
# 方法 1: 使用 VAE 编码器（如果可用）
if pipeline.models["ss_encoder"] is not None:
    with torch.no_grad():
        gt_slat = pipeline.models["ss_encoder"](gt_3d_data)

# 方法 2: 从预训练模型提取（作为伪标签）
with torch.no_grad():
    ss_output = pipeline.sample_sparse_structure(batch)
    coords = ss_output["coords"]
    gt_slat = pipeline.sample_slat(batch, coords)
```

### Q4: 如何加载微调后的权重？

```python
# 保存
torch.save(
    pipeline.models["slat_decoder_gs"].state_dict(),
    "finetuned_decoder.pth"
)

# 加载
pipeline.models["slat_decoder_gs"].load_state_dict(
    torch.load("finetuned_decoder.pth")
)
```

## 高级技巧

### 1. LoRA 微调（低秩适应）

```python
# 安装: pip install loralib
import loralib as lora

# 将线性层替换为 LoRA 层
def apply_lora(model, r=4):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 获取父模块和属性名
            parent = model
            attrs = name.split('.')
            for attr in attrs[:-1]:
                parent = getattr(parent, attr)

            # 替换为 LoRA 层
            lora_layer = lora.Linear(
                module.in_features,
                module.out_features,
                r=r,
                bias=module.bias is not None
            )
            setattr(parent, attrs[-1], lora_layer)

    # 只训练 LoRA 参数
    lora.mark_only_lora_as_trainable(model)
    return model

# 应用到解码器
decoder = apply_lora(pipeline.models["slat_decoder_gs"], r=4)
```

### 2. 知识蒸馏

```python
def distillation_loss(student_output, teacher_output, temperature=2.0):
    """知识蒸馏损失"""
    student_logits = student_output / temperature
    teacher_logits = teacher_output / temperature

    kd_loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)

    return kd_loss
```

### 3. 渐进式训练

```python
# 先训练解码器，再微调生成器
def progressive_training(pipeline, config):
    # 阶段 1: 训练解码器
    print("Stage 1: Training decoder...")
    train_decoder(pipeline, config["decoder"])

    # 阶段 2: 微调 Stage 2 生成器
    print("Stage 2: Fine-tuning generator...")
    train_stage2_generator(pipeline, config["stage2"])

    # 阶段 3: 端到端微调（可选）
    if config.get("end_to_end"):
        print("Stage 3: End-to-end fine-tuning...")
        train_end_to_end(pipeline, config["e2e"])
```

## 资源和参考

- **Flow Matching 论文**: [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- **Rectified Flow**: [Flow Straight and Fast](https://arxiv.org/abs/2403.03206)
- **3D Gaussian Splatting**: [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- **FlexiCubes**: [Flexible Isosurface Extraction for Gradient-Based Mesh Optimization](https://research.nvidia.com/labs/toronto-ai/flexicubes/)

## 支持和贡献

如果您成功微调了 SAM 3D Objects 或有改进建议，欢迎：
1. 在原始仓库提交 Issue
2. 分享您的训练配置和结果
3. 贡献代码改进（需遵守 CLA）

---

**最后更新**: 2025-11-21
**作者**: Claude AI Assistant
**许可证**: SAM License
