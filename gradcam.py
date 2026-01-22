"""
Grad-CAM 热力图生成示例代码
用于生成模型关注区域的可视化热力图，验证医学分割模型的医学相关性
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import cm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T

# 导入项目模块
from datasets.datasets import NPY_datasets
from configs.config_setting_isic17 import setting_config
from configs.model_configs import model_configs
from models.other_models import UNet, TransUNet, SwinUNet, LocalMamba, LocalVisionMamba
# 延迟导入LCVMUNet，避免不必要的依赖
# from models.PCViM import LCVMUNet
from utils import set_seed


class GradCAM:
    """
    Grad-CAM 实现类
    用于生成模型关注区域的热力图
    """
    
    def __init__(self, model, target_layer=None):
        """
        Args:
            model: 训练好的PyTorch模型
            target_layer: 目标卷积层，如果为None则自动选择倒数第二个Conv2d层
        """
        self.model = model
        # 确保模型参数需要梯度
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.eval()
        
        # 自动查找合适的卷积层（优先选择decoder的层，更接近输出）
        if target_layer is None:
            conv_layers = []
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Conv2d):
                    conv_layers.append((name, m))
            
            if len(conv_layers) >= 2:
                # 优先选择decoder部分的层（更接近输出，梯度更容易传播）
                target_name, target_layer = None, None
                for name, layer in conv_layers:
                    if 'dec' in name and ('1' in name or '2' in name or '3' in name or '4' in name):
                        # 选择decoder中最后一个卷积层
                        if 'dec1' in name:
                            target_name, target_layer = name, layer
                            break
                        elif 'dec2' in name and target_layer is None:
                            target_name, target_layer = name, layer
                        elif 'dec3' in name and target_layer is None:
                            target_name, target_layer = name, layer
                        elif 'dec4' in name and target_layer is None:
                            target_name, target_layer = name, layer
                
                # 如果没找到decoder层，使用倒数第二个
                if target_layer is None:
                    target_name, target_layer = conv_layers[-2]
                print(f"自动选择目标层: {target_name}")
            elif len(conv_layers) == 1:
                target_name, target_layer = conv_layers[-1]
                print(f"自动选择目标层: {target_name}")
            else:
                raise ValueError("模型中没有找到Conv2d层，无法进行Grad-CAM")
        else:
            print(f"使用指定的目标层: {target_layer}")
        
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # 注册前向和反向钩子 - 使用full_backward_hook以获取完整梯度
        def forward_hook(module, input, output):
            self.activations = output  # 不detach，保持梯度连接
        
        def backward_hook(module, grad_input, grad_output):
            # grad_output是tuple，取第一个
            if grad_output[0] is not None:
                self.gradients = grad_output[0]  # 不detach，保持梯度
            else:
                self.gradients = None
        
        self.target_layer.register_forward_hook(forward_hook)
        # 使用register_full_backward_hook以获取完整梯度信息
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor, target_mask=None):
        """
        生成Grad-CAM热力图
        
        Args:
            input_tensor: 输入图像张量 (B, C, H, W)
            target_mask: 目标掩膜 (B, 1, H, W)，可选。如果提供，则聚焦于病灶区域
        
        Returns:
            cam: 热力图数组 (B, H, W)，值域[0, 1]
        """
        # 确保输入需要梯度
        input_tensor = input_tensor.requires_grad_(True)
        
        self.model.zero_grad()
        
        # 前向传播
        output = self.model(input_tensor)  # (B, 1, H, W) 或 (B, num_classes, H, W)
        
        # 检查输出
        print(f"模型输出形状: {output.shape}, 值域: [{output.min().item():.4f}, {output.max().item():.4f}]")
        
        # 计算目标值 - 使用正类（前景）的响应
        # 对于分割任务，我们关注正类（病灶）的响应
        if target_mask is not None and target_mask.shape == output.shape:
            # 聚焦于病灶区域（医学相关性更强）
            # 计算mask区域内的平均响应
            mask_sum = target_mask.sum()
            if mask_sum > 0:
                target = (output * target_mask).sum() / mask_sum
            else:
                target = output.mean()
        else:
            # 默认：对输出求平均
            target = output.mean()
        
        print(f"目标值: {target.item():.6f}")
        
        # 反向传播
        target.backward(retain_graph=False)
        
        # 检查梯度和激活值
        if self.gradients is None:
            raise ValueError("梯度为None，可能目标层选择不当或梯度未正确传播")
        
        # 获取激活值和梯度（此时已经detach）
        acts = self.activations.detach()  # (B, C, h, w)
        grads = self.gradients.detach()  # (B, C, h, w)
        
        print(f"激活值形状: {acts.shape}, 值域: [{acts.min().item():.4f}, {acts.max().item():.4f}]")
        print(f"梯度形状: {grads.shape}, 值域: [{grads.min().item():.4f}, {grads.max().item():.4f}]")
        print(f"梯度绝对值均值: {grads.abs().mean().item():.6f}")
        print(f"梯度绝对值最大值: {grads.abs().max().item():.6f}")
        
        # 如果梯度仍然为0，使用Score-CAM方法（基于激活值的重要性）
        if grads.abs().mean().item() < 1e-6:
            print("警告: 梯度几乎为0，使用Score-CAM方法（基于激活值的重要性）")
            # Score-CAM: 使用激活值对输出的影响作为权重
            B, C, h, w = acts.shape
            
            # 上采样激活值到输出尺寸
            acts_upsampled = F.interpolate(acts, size=output.shape[2:], mode='bilinear', align_corners=False)
            
            # 计算每个通道激活值对输出的影响
            channel_weights = []
            with torch.no_grad():
                for c in range(min(C, 64)):  # 限制通道数，避免计算太慢
                    # 获取该通道的激活值并归一化到[0,1]
                    act_channel = acts_upsampled[:, c:c+1, :, :]  # (B, 1, H, W)
                    act_min = act_channel.min()
                    act_max = act_channel.max()
                    if (act_max - act_min).item() > 1e-6:
                        act_normalized = (act_channel - act_min) / (act_max - act_min + 1e-8)
                    else:
                        act_normalized = act_channel * 0  # 如果值相同，权重为0
                    
                    # 计算激活值与输出的相关性
                    if target_mask is not None:
                        # 计算激活值与mask区域输出的相关性
                        act_in_mask = (act_normalized * target_mask).sum()
                        mask_area = target_mask.sum()
                        if mask_area > 0:
                            weight = (act_in_mask / mask_area).item()
                        else:
                            weight = act_normalized.mean().item()
                    else:
                        # 计算激活值与输出的相关性（使用点积）
                        weight = (act_normalized * output).sum().item() / (act_normalized.sum().item() + 1e-8)
                    
                    channel_weights.append(max(0, weight))  # 只保留非负权重
            
            # 如果通道数被限制了，补充0权重
            if C > 64:
                channel_weights.extend([0.0] * (C - 64))
            
            # 转换为tensor并归一化
            channel_weights = torch.tensor(channel_weights, device=acts.device, dtype=acts.dtype).view(1, C, 1, 1)
            channel_weights = F.softmax(channel_weights * 10, dim=1)  # 使用温度参数增强差异
            
            # 使用权重对激活值进行加权求和
            cam = (channel_weights * acts).sum(dim=1, keepdim=True)  # (B, 1, h, w)
            cam = F.relu(cam)  # 只保留正向激活
            
            print(f"Score-CAM计算完成，通道权重范围: [{channel_weights.min().item():.6f}, {channel_weights.max().item():.6f}]")
        else:
            # 标准Grad-CAM: 计算权重：对梯度进行全局平均池化
            weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            print(f"权重形状: {weights.shape}, 值域: [{weights.min().item():.4f}, {weights.max().item():.4f}]")
            
            # 计算CAM
            cam = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, h, w)
            cam = F.relu(cam)  # 只保留正向激活
        
        print(f"CAM计算后形状: {cam.shape}, 值域: [{cam.min().item():.4f}, {cam.max().item():.4f}]")
        
        # 归一化到[0, 1]
        B, _, h, w = cam.shape
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1, keepdim=True)[0]
        cam_max = cam_flat.max(dim=1, keepdim=True)[0]
        
        print(f"CAM min: {cam_min.item():.6f}, max: {cam_max.item():.6f}")
        
        # 如果max和min相同或接近，尝试使用激活值本身
        if (cam_max - cam_min).item() < 1e-6:
            print("警告: CAM值几乎相同，尝试使用激活值的空间信息")
            # 使用激活值的空间平均作为CAM
            cam = acts.mean(dim=1, keepdim=True)  # (B, 1, h, w)
            cam = F.relu(cam)
            cam_flat = cam.view(B, -1)
            cam_min = cam_flat.min(dim=1, keepdim=True)[0]
            cam_max = cam_flat.max(dim=1, keepdim=True)[0]
            print(f"使用激活值后 CAM min: {cam_min.item():.6f}, max: {cam_max.item():.6f}")
        
        cam = (cam_flat - cam_min) / (cam_max - cam_min + 1e-8)
        cam = cam.view(B, 1, h, w)
        
        print(f"归一化后CAM值域: [{cam.min().item():.4f}, {cam.max().item():.4f}]")
        
        # 上采样到输入图像大小
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        cam_np = cam.squeeze(1).cpu().detach().numpy()  # (B, H, W)
        print(f"最终CAM形状: {cam_np.shape}, 值域: [{cam_np.min():.4f}, {cam_np.max():.4f}]")
        
        return cam_np


def overlay_gradcam_on_image(image_tensor, cam_map, alpha=0.5):
    """
    将Grad-CAM热力图叠加到原图上
    
    Args:
        image_tensor: 图像张量 (C, H, W)，值域可以是[0, 1]或[0, 255]
        cam_map: 热力图数组 (H, W)，值域[0, 1]
        alpha: 叠加透明度，0-1之间
    
    Returns:
        overlay: 叠加后的图像 (H, W, 3)，值域[0, 1]
    """
    # 确保图像是3通道
    img = image_tensor.detach().cpu().float()
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    
    # 归一化到[0, 1]
    max_val = img.max()
    if max_val > 1.0:
        img = img / 255.0
    
    img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3)
    
    # 使用jet colormap将热力图转换为彩色
    heatmap = cm.jet(cam_map)[..., :3]  # (H, W, 3)，值域[0, 1]
    
    # 混合原图和热力图
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def save_gradcam_visualization(image_tensor, cam_map, save_path, alpha=0.5):
    """
    保存Grad-CAM可视化结果
    
    Args:
        image_tensor: 图像张量 (C, H, W)
        cam_map: 热力图数组 (H, W)
        save_path: 保存路径
        alpha: 叠加透明度
    """
    # 生成叠加图像
    overlay = overlay_gradcam_on_image(image_tensor, cam_map, alpha=alpha)
    
    # 转换为PIL Image并保存
    overlay_img = (overlay * 255).astype(np.uint8)
    pil_img = Image.fromarray(overlay_img)
    pil_img.save(save_path)
    
    print(f"Grad-CAM可视化已保存到: {save_path}")


def generate_gradcam_for_batch(model, dataloader, save_dir, num_samples=10):
    """
    为数据加载器中的样本生成Grad-CAM热力图
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        save_dir: 保存目录
        num_samples: 要处理的样本数量
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'heatmaps'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'overlays'), exist_ok=True)
    
    # 初始化Grad-CAM
    grad_cam = GradCAM(model)
    
    count = 0
    model.eval()
    
    for idx, (img, mask) in enumerate(dataloader):
        if count >= num_samples:
            break
        
        img = img.cuda().float()
        mask = mask.cuda().float()
        
        # 生成CAM（使用GT mask聚焦病灶区域）
        with torch.enable_grad():
            cam_map = grad_cam.generate(img, target_mask=mask)
        
        cam_map = cam_map[0]  # 取第一个样本 (H, W)
        img_vis = img[0]  # (C, H, W)
        
        # 保存纯热力图
        cam_img = (cam_map * 255).astype(np.uint8)
        cam_pil = Image.fromarray(cam_img)
        cam_pil.save(os.path.join(save_dir, 'heatmaps', f'cam_{idx}.png'))
        
        # 保存叠加图像
        overlay = overlay_gradcam_on_image(img_vis, cam_map, alpha=0.5)
        overlay_img = (overlay * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_img)
        overlay_pil.save(os.path.join(save_dir, 'overlays', f'overlay_{idx}.png'))
        
        count += 1
        print(f"已处理 {count}/{num_samples} 个样本")
    
    print(f"Grad-CAM生成完成！结果保存在: {save_dir}")


# ========== 主程序 ==========
if __name__ == '__main__':
    """
    直接运行此脚本生成Grad-CAM热力图
    使用指定的模型权重和验证集数据
    """
    
    # 配置参数
    config = setting_config
    weight_path = "/Users/xuyang_wei/PycharmProjects/PC-ViM-Code/configs/pre_trained_weights/best-epoch96-loss0.0925.pth"
    num_samples = 10  # 要处理的样本数量
    save_dir = "./gradcam_results"  # 结果保存目录
    
    # 检测设备（优先使用CPU，因为CUDA可能不可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(config.seed)
    
    print("=" * 50)
    print("开始生成Grad-CAM热力图")
    print("=" * 50)
    
    # 1. 构建模型
    print("\n[1/4] 构建模型...")
    model_cfg = model_configs[config.network]
    
    if config.network == 'unet':
        model = UNet(**model_cfg)
    elif config.network == 'transunet':
        model = TransUNet(**model_cfg)
    elif config.network == 'swinunet':
        model = SwinUNet(**model_cfg)
    elif config.network == 'localmamba':
        model = LocalMamba(**model_cfg)
    elif config.network == 'localvisionmamba':
        model = LocalVisionMamba(**model_cfg)
    elif config.network == PC-Vamba':
        # 延迟导入，避免不必要的依赖
        from models.PCViM import LCVMUNet
        model_cfg_full = config.model_config
        model = LCVMUNet(
            num_classes=model_cfg_full['num_classes'],
            input_channels=model_cfg_full['input_channels'],
            depths=model_cfg_full['depths'],
            depths_decoder=model_cfg_full['depths_decoder'],
            drop_path_rate=model_cfg_full['drop_path_rate'],
            load_ckpt_path=model_cfg_full['load_ckpt_path'],
        )
    else:
        raise Exception(f'不支持的模型: {config.network}')
    
    # 2. 加载权重
    print(f"\n[2/4] 加载模型权重: {weight_path}")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    
    state_dict = torch.load(weight_path, map_location='cpu')
    # 处理不同的权重文件格式
    if isinstance(state_dict, dict):
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)  # 使用检测到的设备
    model.eval()
    print("模型权重加载完成！")
    
    # 3. 加载单张图像
    print(f"\n[3/4] 加载图像...")
    image_path = "/Users/Documents/ISIC.jpg"
    mask_path = "/Users/Documents/ISIC_segmentation.png"
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    image_name = os.path.basename(image_path)
    image_name_no_ext = os.path.splitext(image_name)[0]
    
    # 加载图像和mask
    img_pil = Image.open(image_path).convert('RGB')
    img_np = np.array(img_pil)
    
    if os.path.exists(mask_path):
        mask_pil = Image.open(mask_path).convert('L')
        mask_np = np.expand_dims(np.array(mask_pil), axis=2) / 255.0
        print(f"加载mask文件: {mask_path}")
        has_mask = True
    else:
        # 如果没有mask，创建一个全零的mask（不使用target_mask）
        mask_np = np.zeros((img_np.shape[0], img_np.shape[1], 1), dtype=np.float32)
        print(f"警告: mask文件不存在 {mask_path}，将不使用target_mask进行Grad-CAM")
        has_mask = False
    
    # 使用与数据集相同的预处理
    img_processed, mask_processed = config.test_transformer((img_np, mask_np))
    
    # 转换为batch格式 (1, C, H, W)
    img_batch = img_processed.unsqueeze(0).to(device).float()
    mask_batch = mask_processed.unsqueeze(0).to(device).float()
    
    print(f"图像尺寸: {img_batch.shape}")
    print(f"Mask尺寸: {mask_batch.shape}")
    
    # 4. 生成Grad-CAM
    print(f"\n[4/4] 生成Grad-CAM热力图...")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'heatmaps'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'overlays'), exist_ok=True)
    
    grad_cam = GradCAM(model)
    
    # 生成CAM（如果有mask则使用，否则不使用target_mask）
    with torch.enable_grad():
        if has_mask:
            cam_map_batch = grad_cam.generate(img_batch, target_mask=mask_batch)
        else:
            cam_map_batch = grad_cam.generate(img_batch, target_mask=None)
    
    cam_map = cam_map_batch[0]  # 取第一个样本 (H, W)
    img_vis = img_batch[0]  # (C, H, W)
    
    # 确保图像是3通道
    if img_vis.shape[0] == 1:
        rgb_img = img_vis.repeat(3, 1, 1)
    else:
        rgb_img = img_vis
    
    # 保存纯热力图
    cam_img = (cam_map * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_img)
    save_cam_path = os.path.join(save_dir, 'heatmaps', f'cam_{image_name_no_ext}.png')
    cam_pil.save(save_cam_path)
    print(f"  热力图已保存: {save_cam_path}")
    
    # 保存叠加图像
    overlay = overlay_gradcam_on_image(rgb_img, cam_map, alpha=0.5)
    overlay_img = (overlay * 255).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay_img)
    save_overlay_path = os.path.join(save_dir, 'overlays', f'overlay_{image_name_no_ext}.png')
    overlay_pil.save(save_overlay_path)
    print(f"  叠加图已保存: {save_overlay_path}")
    
    print("\n" + "=" * 50)
    print(f"Grad-CAM生成完成！")
    print(f"结果保存在: {save_dir}")
    print(f"  - 热力图: {save_dir}/heatmaps/")
    print(f"  - 叠加图: {save_dir}/overlays/")
    print("=" * 50)

