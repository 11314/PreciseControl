import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

# =============================
# 1. 路径设置
# =============================

DATASET_ROOT = "/hxp/zy/PreciseControl/test_dict/dataset"  # 改成你的数据集路径
ENCODER_CKPT = "./weights/encoder/e4e_ffhq_encode.pt"
OUTPUT_JSON = "/hxp/zy/PreciseControl/test_dict/beard_delta_w.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# =============================
# 2. 加载 e4e 编码器
# =============================

import sys
import os

# 添加项目根目录到 Python 路径
project_root = "/hxp/zy/PreciseControl"
sys.path.insert(0, project_root)

# 现在可以正常导入了
from ldm.modules.e4e.psp import pSp   # 与原项目一致
from types import SimpleNamespace  # 添加这行

def load_e4e(checkpoint_path):
    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cuda')
    
    # 获取 opts
    opts = ckpt['opts']
    
    # 将字典转换为 Namespace 对象
    opts = SimpleNamespace(**opts)
    
    # 关键修改：添加 checkpoint_path 到 opts
    opts.checkpoint_path = checkpoint_path

    # 设置默认值（如果需要的话）
    if not hasattr(opts, 'encoder_type'):
        opts.encoder_type = 'Encoder4Editing'  # 或其他合适的默认值,GradualStyleEncoder/Encoder4Editing/SingleStyleCodeEncoder
    
    # 创建网络实例
    net = pSp(opts)
    net.eval()
    # # 关键：将模型移到 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    return net


encoder = load_e4e(ENCODER_CKPT)

# =============================
# 3. 图像预处理
# =============================

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).to(device)
    return img


# =============================
# 4. 计算 delta w
# =============================

all_deltas = []

folders = sorted(os.listdir(DATASET_ROOT))

for folder in tqdm(folders):
    folder_path = os.path.join(DATASET_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    original_path = os.path.join(folder_path, "original.jpg")

    # 自动找 beard 文件
    edited_path = None
    for file in os.listdir(folder_path):
        if "beard" in file:
            edited_path = os.path.join(folder_path, file)

    if edited_path is None:
        continue

    # ---- 编码 ----
    with torch.no_grad():
        original_img = load_image(original_path)
        edited_img = load_image(edited_path)

        w_original = encoder(original_img, return_latents=True)[1]
        w_edited   = encoder(edited_img, return_latents=True)[1]

        # shape: (1, 18, 512)
        delta = w_edited - w_original

        all_deltas.append(delta.squeeze(0).cpu().numpy())

print("Collected pairs:", len(all_deltas))

# =============================
# 5. 求平均方向
# =============================

all_deltas = np.stack(all_deltas, axis=0)   # (N, 18, 512)
delta_mean = np.mean(all_deltas, axis=0)    # (18, 512)

print("delta_mean shape:", delta_mean.shape)

# =============================
# 6. 保存为和原json一致的格式
# =============================

delta_dict = {
    "beard": [delta_mean.tolist()]
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(delta_dict, f)

print("Saved to", OUTPUT_JSON)