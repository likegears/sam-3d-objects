# SAM 3D Objects 训练数据准备指南

本指南详细说明 SAM 3D Objects 所需的训练数据类型、格式要求以及数据配对方法。

## 目录

1. [数据类型概述](#数据类型概述)
2. [最小数据要求](#最小数据要求)
3. [完整训练数据](#完整训练数据)
4. [数据格式规范](#数据格式规范)
5. [数据来源与获取](#数据来源与获取)
6. [数据配对方法](#数据配对方法)
7. [数据预处理](#数据预处理)
8. [数据集组织](#数据集组织)

---

## 数据类型概述

SAM 3D Objects 是一个**图像到 3D 的生成模型**，根据不同的训练策略，需要不同类型的数据配对：

| 训练策略 | 输入数据 | 监督信号 | 数据量需求 |
|---------|---------|---------|-----------|
| **仅解码器微调** | 图像 + 掩码 | 3D 模型（Gaussian/Mesh） | 1K-5K |
| **Stage 2 微调** | 图像 + 掩码 | 稀疏结构 + 特征 | 10K-50K |
| **端到端训练** | 图像 + 掩码 | 完整 3D 数据 | 100K+ |

---

## 最小数据要求

### 1. 推理（Inference）

无需训练，只需要：

```
输入:
├── image.png         # RGB 图像（任意尺寸）
└── mask.png          # 二值掩码（与图像同尺寸）
```

**示例数据结构**（见 `notebook/images/`）：
```
shutterstock_stylish_kidsroom_1640806567/
├── image.png         # 原始场景图像 (1920x1080)
├── 0.png            # 第一个物体的掩码
├── 1.png            # 第二个物体的掩码
├── 2.png            # ...
└── ...
```

### 2. 解码器微调（最简单）

```
训练样本:
├── image.png         # RGB 图像
├── mask.png          # 二值掩码
└── ground_truth/     # 监督信号
    ├── splat.ply     # 3D 高斯溅射（推荐）
    └── mesh.obj      # 或三角网格
```

**关键点**：
- ✅ 只需要图像和对应的 3D 模型
- ✅ 可以使用其他 3D 重建方法生成监督数据
- ✅ 数据量相对较小（1K-5K 样本）

---

## 完整训练数据

### 数据配对要求

完整的训练样本应包含以下配对数据：

#### 1. 图像数据（必需）

```python
{
    # 原始图像（RGB）
    "image": torch.Tensor,          # Shape: [3, H, W], Range: [0, 1]

    # 掩码（分割遮罩）
    "mask": torch.Tensor,           # Shape: [1, H, W], Range: [0, 1]

    # 完整场景图像（可选）
    "rgb_image": torch.Tensor,      # Shape: [3, H, W]

    # 完整场景掩码（可选）
    "rgb_image_mask": torch.Tensor, # Shape: [1, H, W]
}
```

**要求**：
- 图像尺寸：推荐 512x512，支持任意尺寸（会自动调整）
- 格式：PNG、JPEG（推荐 PNG）
- 颜色空间：RGB
- 掩码：二值图像（0=背景，255=前景）或灰度图

#### 2. 3D 监督数据（取决于训练策略）

##### A. 高斯溅射（Gaussian Splatting）

```python
{
    "gaussian": {
        "xyz": torch.Tensor,         # [N, 3] - 高斯中心位置
        "features_dc": torch.Tensor, # [N, 3] - 颜色（DC 分量）
        "features_rest": torch.Tensor, # [N, 45] - 球谐函数系数（可选）
        "scaling": torch.Tensor,     # [N, 3] - 缩放参数
        "rotation": torch.Tensor,    # [N, 4] - 四元数旋转
        "opacity": torch.Tensor,     # [N, 1] - 不透明度
    }
}
```

**来源**：
- 从 `.ply` 文件加载（标准高斯溅射格式）
- 使用其他方法生成（如原始 3D Gaussian Splatting）

##### B. 网格（Mesh）

```python
{
    "mesh": {
        "vertices": torch.Tensor,    # [V, 3] - 顶点坐标
        "faces": torch.Tensor,       # [F, 3] - 三角形面索引
        "vertex_colors": torch.Tensor, # [V, 3] - 顶点颜色（可选）
        "vertex_normals": torch.Tensor, # [V, 3] - 顶点法线（可选）
        "texture_uvs": torch.Tensor,  # [V, 2] - UV 坐标（可选）
        "texture_image": Image,       # 纹理图像（可选）
    }
}
```

**来源**：
- OBJ、GLB、FBX 等格式
- 使用 `trimesh` 或 `pytorch3d` 加载

##### C. 稀疏结构（Sparse Structure）

```python
{
    "coords": torch.Tensor,          # [N, 4] - 稀疏坐标 [batch, x, y, z]
    "occupancy": torch.Tensor,       # [N] - 占用标签（0或1）
}
```

**来源**：
- 从 3D 模型体素化生成
- 使用预训练模型提取（伪标签）

##### D. 位姿信息（Pose）

```python
{
    # 物体在场景中的位姿
    "instance_scale": torch.Tensor,      # [1] - 缩放比例
    "instance_rotation": torch.Tensor,   # [4] - 四元数旋转
    "instance_translation": torch.Tensor, # [3] - 平移向量

    # 场景归一化信息
    "scene_scale": torch.Tensor,         # [1] - 场景缩放
    "scene_center": torch.Tensor,        # [3] - 场景中心
}
```

**来源**：
- 多视图重建时的相机姿态
- 人工标注或自动估计

---

## 数据格式规范

### 文件组织方式 1：每个物体单独存储

```
dataset/
├── train/
│   ├── object_0001/
│   │   ├── image.png                # RGB 图像
│   │   ├── mask.png                 # 二值掩码
│   │   ├── splat.ply                # 3D 高斯溅射（可选）
│   │   ├── mesh.obj                 # 网格（可选）
│   │   └── metadata.json            # 元数据
│   ├── object_0002/
│   │   └── ...
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

### 文件组织方式 2：场景级别存储

```
dataset/
├── scene_0001/
│   ├── image.png                    # 场景图像
│   ├── objects/
│   │   ├── 0_mask.png              # 物体 0 的掩码
│   │   ├── 0_splat.ply             # 物体 0 的 3D 数据
│   │   ├── 1_mask.png              # 物体 1 的掩码
│   │   ├── 1_splat.ply             # 物体 1 的 3D 数据
│   │   └── ...
│   └── metadata.json
└── ...
```

### metadata.json 格式

```json
{
    "image_id": "object_0001",
    "image_path": "image.png",
    "mask_path": "mask.png",
    "gaussian_path": "splat.ply",
    "mesh_path": "mesh.obj",
    "category": "chair",
    "source": "objaverse",
    "pose": {
        "scale": 1.0,
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "translation": [0.0, 0.0, 0.0]
    },
    "camera": {
        "fov": 49.1,
        "resolution": [512, 512]
    }
}
```

---

## 数据来源与获取

### 方案 1：使用公开数据集

#### A. Objaverse（推荐）

**优点**：
- 包含 800K+ 3D 模型
- 高质量、多样化
- 免费使用

**获取方式**：
```bash
# 安装 Objaverse
pip install objaverse

# 下载数据
python download_objaverse.py
```

**渲染图像 + 掩码**：
```python
import objaverse
import bpy  # Blender Python API

def render_object(obj_path, output_dir):
    """使用 Blender 渲染物体"""
    # 1. 加载 3D 模型
    bpy.ops.import_scene.gltf(filepath=obj_path)

    # 2. 设置相机
    camera = bpy.data.objects['Camera']
    camera.location = (2, -2, 2)
    camera.rotation_euler = (math.radians(60), 0, math.radians(45))

    # 3. 渲染 RGB 图像
    bpy.context.scene.render.filepath = f"{output_dir}/image.png"
    bpy.ops.render.render(write_still=True)

    # 4. 渲染掩码（使用 ID 遮罩）
    bpy.context.scene.render.filepath = f"{output_dir}/mask.png"
    bpy.context.scene.use_nodes = True
    # ... 配置节点渲染掩码
    bpy.ops.render.render(write_still=True)

# 批量处理
uids = objaverse.load_uids()
for uid in uids[:1000]:
    obj_path = objaverse.load_objects([uid])[uid]
    render_object(obj_path, f"dataset/train/{uid}")
```

#### B. ShapeNet

**特点**：
- 51,300 个 3D 模型
- 55 个类别
- 需要申请访问

**获取**：https://shapenet.org/

#### C. CO3D（Common Objects in 3D）

**特点**：
- 真实世界图像 + 3D 重建
- 19,000+ 视频序列
- Meta 发布

**获取**：https://github.com/facebookresearch/co3d

#### D. Google Scanned Objects

**特点**：
- 高质量扫描模型
- 1,000+ 日常物品
- 免费下载

**获取**：https://app.ignitionrobotics.org/GoogleResearch/fuel/collections/Google%20Scanned%20Objects

### 方案 2：从真实图像生成 3D 数据

如果您有图像但没有 3D 数据，可以使用现有方法生成伪标签：

#### A. 使用原始 3D Gaussian Splatting

```bash
# 1. 安装 3DGS
git clone https://github.com/graphdeco-inria/gaussian-splatting
cd gaussian-splatting
pip install -r requirements.txt

# 2. 准备多视图图像
# 需要 COLMAP 格式的相机参数

# 3. 训练高斯溅射
python train.py -s <path_to_data> -m <output_path>

# 4. 导出 PLY 文件
# 使用生成的 point_cloud.ply 作为监督数据
```

#### B. 使用其他单图 3D 重建方法

- **TripoSR**：快速单图到网格
- **DreamGaussian**：单图到高斯溅射
- **LRM**：Large Reconstruction Model
- **InstantMesh**：快速网格重建

```python
# 示例：使用 TripoSR 生成网格
from triposr import TripoSR

model = TripoSR.from_pretrained("stabilityai/TripoSR")

for image_path in image_list:
    # 生成网格
    mesh = model.generate_mesh(image_path)

    # 保存
    mesh.export(f"{output_dir}/mesh.obj")
```

### 方案 3：合成数据

使用 Blender 或 Unity 创建合成数据：

```python
# Blender 脚本示例
import bpy
import random

def create_synthetic_scene():
    # 1. 随机放置物体
    for i in range(random.randint(3, 10)):
        bpy.ops.mesh.primitive_cube_add(
            location=(random.uniform(-5, 5),
                     random.uniform(-5, 5),
                     random.uniform(0, 3))
        )

    # 2. 随机相机视角
    camera = bpy.data.objects['Camera']
    camera.location = random_sphere_point(radius=5)

    # 3. 渲染
    bpy.ops.render.render()
```

---

## 数据配对方法

### 方法 1：直接配对（有 3D 模型）

```python
import os
import json
from pathlib import Path

def create_paired_dataset(model_dir, output_dir):
    """
    从 3D 模型生成配对的图像和掩码

    参数：
    - model_dir: 3D 模型目录（.obj, .glb, .ply）
    - output_dir: 输出目录
    """
    pairs = []

    for model_file in Path(model_dir).glob("*.obj"):
        model_id = model_file.stem

        # 1. 渲染图像
        image_path = render_model(
            model_file,
            output_path=f"{output_dir}/{model_id}/image.png",
            resolution=(512, 512)
        )

        # 2. 生成掩码
        mask_path = render_mask(
            model_file,
            output_path=f"{output_dir}/{model_id}/mask.png"
        )

        # 3. 转换为高斯溅射
        gaussian_path = convert_to_gaussian(
            model_file,
            output_path=f"{output_dir}/{model_id}/splat.ply"
        )

        # 4. 保存配对信息
        pairs.append({
            "id": model_id,
            "image": str(image_path),
            "mask": str(mask_path),
            "gaussian": str(gaussian_path),
            "model": str(model_file)
        })

    # 保存数据集索引
    with open(f"{output_dir}/dataset.json", "w") as f:
        json.dump(pairs, f, indent=2)

    return pairs
```

### 方法 2：多视图重建配对

```python
def create_multiview_dataset(image_dir, output_dir):
    """
    从多视图图像重建 3D 并创建数据集

    流程：
    1. 使用 COLMAP 进行 SfM
    2. 使用 3DGS 或 NeRF 重建 3D
    3. 提取单视图作为训练样本
    """
    scenes = []

    for scene_dir in Path(image_dir).iterdir():
        if not scene_dir.is_dir():
            continue

        # 1. COLMAP 重建
        run_colmap(scene_dir)

        # 2. 训练 3D Gaussian Splatting
        gaussian_model = train_gaussian_splatting(
            scene_dir,
            output=f"{output_dir}/{scene_dir.name}/splat.ply"
        )

        # 3. 为每个视图创建训练样本
        for view_idx, image_path in enumerate(scene_dir.glob("images/*.png")):
            # 获取该视图的渲染结果
            rendered = render_view(gaussian_model, view_idx)

            # 分割前景物体
            mask = segment_foreground(image_path)

            # 保存配对数据
            save_pair(
                image=image_path,
                mask=mask,
                gaussian=gaussian_model,
                output_dir=f"{output_dir}/samples/view_{view_idx}"
            )
```

### 方法 3：使用 SAM 自动分割

利用 Segment Anything Model (SAM) 自动生成掩码：

```python
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2

def auto_segment_and_pair(image_dir, model_dir, output_dir):
    """
    自动分割图像并与 3D 模型配对
    """
    # 加载 SAM
    sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)

    for image_path in Path(image_dir).glob("*.png"):
        # 1. 读取图像
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. 自动生成掩码
        masks = mask_generator.generate(image_rgb)

        # 3. 为每个掩码创建样本
        for idx, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]

            # 4. 匹配对应的 3D 模型（需要实现匹配逻辑）
            model_path = find_matching_model(
                image_path,
                mask,
                model_dir
            )

            if model_path:
                # 保存配对
                save_sample(
                    image=image_rgb,
                    mask=mask,
                    model=model_path,
                    output_dir=f"{output_dir}/{image_path.stem}_{idx}"
                )
```

---

## 数据预处理

### 图像预处理

```python
import torch
import torchvision.transforms as transforms
from PIL import Image

class ImagePreprocessor:
    def __init__(self, size=512):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            # ImageNet 归一化
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)

class MaskPreprocessor:
    def __init__(self, size=512):
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ])

    def __call__(self, mask_path):
        mask = Image.open(mask_path).convert('L')
        mask = self.transform(mask)
        # 二值化
        mask = (mask > 0.5).float()
        return mask
```

### 3D 数据预处理

```python
import trimesh
import numpy as np

def preprocess_mesh(mesh_path, normalize=True):
    """预处理网格数据"""
    mesh = trimesh.load(mesh_path)

    if normalize:
        # 1. 中心化
        mesh.vertices -= mesh.centroid

        # 2. 归一化到单位球
        max_dist = np.max(np.linalg.norm(mesh.vertices, axis=1))
        mesh.vertices /= max_dist

    return {
        'vertices': torch.from_numpy(mesh.vertices).float(),
        'faces': torch.from_numpy(mesh.faces).long(),
    }

def preprocess_gaussian(ply_path):
    """预处理高斯溅射数据"""
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)

    xyz = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=-1)

    # 提取其他属性
    # ... (旋转、缩放、不透明度等)

    return {
        'xyz': torch.from_numpy(xyz).float(),
        # ... 其他字段
    }
```

---

## 数据集组织

### PyTorch Dataset 实现

```python
import torch
from torch.utils.data import Dataset
import json
from pathlib import Path

class SAM3DDataset(Dataset):
    """
    SAM 3D Objects 训练数据集

    数据格式：
    - 方案 A：每个样本一个文件夹
    - 方案 B：通过 JSON 索引文件
    """

    def __init__(
        self,
        data_root,
        split="train",
        load_gaussian=True,
        load_mesh=False,
        transform=None
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.load_gaussian = load_gaussian
        self.load_mesh = load_mesh
        self.transform = transform

        # 加载数据索引
        index_file = self.data_root / f"{split}.json"
        with open(index_file) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        # 1. 加载图像
        image = self.load_image(sample_info['image'])

        # 2. 加载掩码
        mask = self.load_mask(sample_info['mask'])

        # 3. 加载 3D 数据
        data = {
            'image': image,
            'mask': mask,
            'id': sample_info['id']
        }

        if self.load_gaussian and 'gaussian' in sample_info:
            data['gaussian'] = self.load_gaussian_data(
                sample_info['gaussian']
            )

        if self.load_mesh and 'mesh' in sample_info:
            data['mesh'] = self.load_mesh_data(
                sample_info['mesh']
            )

        # 4. 应用变换
        if self.transform:
            data = self.transform(data)

        return data

    def load_image(self, path):
        """加载并预处理图像"""
        from PIL import Image
        image = Image.open(self.data_root / path).convert('RGB')
        image = torch.from_numpy(np.array(image)).float() / 255.0
        return image.permute(2, 0, 1)  # HWC -> CHW

    def load_mask(self, path):
        """加载并预处理掩码"""
        from PIL import Image
        mask = Image.open(self.data_root / path).convert('L')
        mask = torch.from_numpy(np.array(mask)).float() / 255.0
        return mask.unsqueeze(0)  # HW -> 1HW

    def load_gaussian_data(self, path):
        """加载高斯溅射数据"""
        # 实现 PLY 加载逻辑
        return preprocess_gaussian(self.data_root / path)

    def load_mesh_data(self, path):
        """加载网格数据"""
        return preprocess_mesh(self.data_root / path)

# 使用示例
dataset = SAM3DDataset(
    data_root="path/to/dataset",
    split="train",
    load_gaussian=True
)

# DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_collate_fn  # 需要自定义
)
```

### 自定义 Collate 函数

```python
def custom_collate_fn(batch):
    """
    自定义批处理函数
    处理不同大小的网格和高斯数据
    """
    # 图像和掩码可以直接堆叠
    images = torch.stack([item['image'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])

    # 高斯数据需要特殊处理（点数不同）
    gaussians = [item.get('gaussian') for item in batch if 'gaussian' in item]

    # 网格数据保持为列表
    meshes = [item.get('mesh') for item in batch if 'mesh' in item]

    return {
        'image': images,
        'mask': masks,
        'gaussian': gaussians,  # List[Dict]
        'mesh': meshes,          # List[Dict]
        'id': [item['id'] for item in batch]
    }
```

---

## 数据质量检查

### 验证脚本

```python
def validate_dataset(dataset_dir):
    """验证数据集完整性和质量"""
    issues = []

    for sample_dir in Path(dataset_dir).iterdir():
        if not sample_dir.is_dir():
            continue

        sample_id = sample_dir.name

        # 1. 检查文件存在性
        image_path = sample_dir / "image.png"
        mask_path = sample_dir / "mask.png"
        gaussian_path = sample_dir / "splat.ply"

        if not image_path.exists():
            issues.append(f"{sample_id}: 缺少 image.png")

        if not mask_path.exists():
            issues.append(f"{sample_id}: 缺少 mask.png")

        # 2. 检查图像尺寸匹配
        if image_path.exists() and mask_path.exists():
            img = Image.open(image_path)
            mask = Image.open(mask_path)

            if img.size != mask.size:
                issues.append(
                    f"{sample_id}: 图像和掩码尺寸不匹配 "
                    f"({img.size} vs {mask.size})"
                )

        # 3. 检查掩码有效性
        if mask_path.exists():
            mask_arr = np.array(Image.open(mask_path))
            unique_values = np.unique(mask_arr)

            if not np.all(np.isin(unique_values, [0, 255])):
                issues.append(
                    f"{sample_id}: 掩码包含非二值数据"
                )

            # 检查掩码不为空
            if np.sum(mask_arr > 0) < 10:
                issues.append(
                    f"{sample_id}: 掩码太小或为空"
                )

        # 4. 检查 3D 数据
        if gaussian_path.exists():
            try:
                gs_data = preprocess_gaussian(gaussian_path)
                num_points = gs_data['xyz'].shape[0]

                if num_points < 100:
                    issues.append(
                        f"{sample_id}: 高斯点数太少 ({num_points})"
                    )
            except Exception as e:
                issues.append(
                    f"{sample_id}: 高斯文件损坏 - {e}"
                )

    # 输出报告
    print(f"数据集验证完成:")
    print(f"  总样本数: {len(list(Path(dataset_dir).iterdir()))}")
    print(f"  发现问题: {len(issues)}")

    if issues:
        print("\n问题列表:")
        for issue in issues[:10]:  # 只显示前10个
            print(f"  - {issue}")

    return issues
```

---

## 数据增强

```python
import albumentations as A

def get_augmentation_pipeline():
    """数据增强管道"""
    return A.Compose([
        # 几何变换
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            p=0.5
        ),

        # 颜色增强
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),

        # 噪声和模糊
        A.GaussianBlur(p=0.3),
        A.GaussNoise(p=0.2),

    ], additional_targets={'mask': 'mask'})

# 使用
transform = get_augmentation_pipeline()
augmented = transform(image=image, mask=mask)
```

---

## 快速开始：最小示例

```python
# 1. 准备数据（最小示例）
"""
my_dataset/
├── train.json
└── samples/
    ├── 0001/
    │   ├── image.png
    │   ├── mask.png
    │   └── splat.ply
    └── 0002/
        └── ...
"""

# train.json:
# [
#     {
#         "id": "0001",
#         "image": "samples/0001/image.png",
#         "mask": "samples/0001/mask.png",
#         "gaussian": "samples/0001/splat.ply"
#     },
#     ...
# ]

# 2. 创建数据集
from torch.utils.data import Dataset

dataset = SAM3DDataset(
    data_root="my_dataset",
    split="train"
)

# 3. 检查数据
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Mask shape: {sample['mask'].shape}")
print(f"Gaussian points: {sample['gaussian']['xyz'].shape[0]}")

# 4. 开始训练！
```

---

## 常见问题

### Q1: 如何从单张图像开始？

如果只有图像，使用以下流程生成配对数据：

```bash
# 步骤 1: 使用 SAM 生成掩码
python segment_with_sam.py --image your_image.png --output masks/

# 步骤 2: 使用单图重建生成 3D
python reconstruct_3d.py --image your_image.png --mask masks/0.png --output model.ply

# 步骤 3: 组织数据集
python organize_dataset.py --images images/ --masks masks/ --models models/ --output dataset/
```

### Q2: 数据量多少合适？

- **概念验证**: 100-500 样本
- **特定类别微调**: 1K-5K 样本
- **通用模型**: 10K-100K 样本
- **大规模训练**: 100K+ 样本

### Q3: 如何处理多物体场景？

```python
# 为每个物体单独创建样本
for obj_idx in range(num_objects):
    sample = {
        'image': scene_image,           # 完整场景
        'mask': object_masks[obj_idx],  # 单个物体掩码
        'gaussian': object_gaussians[obj_idx]  # 单个物体 3D
    }
```

### Q4: 3D 数据格式转换

```python
# Mesh → Gaussian
def mesh_to_gaussian(mesh_path, output_path, num_points=100000):
    mesh = trimesh.load(mesh_path)
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    # 创建基础高斯属性
    gaussians = {
        'xyz': points,
        'opacity': np.ones((num_points, 1)),
        'scaling': np.ones((num_points, 3)) * 0.01,
        'rotation': np.tile([1, 0, 0, 0], (num_points, 1)),
        'features_dc': mesh.visual.vertex_colors[face_indices, :3] / 255.0
    }

    save_gaussian_ply(output_path, gaussians)
```

---

## 总结

SAM 3D Objects 的训练数据准备流程：

1. **最简单**：图像 + 掩码 → 仅用于推理
2. **解码器微调**：图像 + 掩码 + 3D 模型（1K-5K）
3. **完整训练**：图像 + 掩码 + 完整 3D 数据（100K+）

**推荐起步方案**：
1. 从 Objaverse 下载 3D 模型
2. 使用 Blender 渲染图像和掩码
3. 组织成标准数据集格式
4. 开始解码器微调实验

**关键要点**：
- ✅ 图像和掩码必须尺寸匹配
- ✅ 掩码必须是二值或灰度
- ✅ 3D 数据需归一化到单位空间
- ✅ 使用 JSON 索引管理大规模数据集

---

**更新时间**: 2025-11-21
**维护者**: Claude AI Assistant
