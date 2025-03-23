import torch
from torch.nn import functional as F

from models.team15_SMT.utils.registry import MODEL_REGISTRY
from models.team15_SMT.models.sr_mutliscale_model import SRModel_mutliscale


@MODEL_REGISTRY.register()
class MambaIRv2Model_combined(SRModel_mutliscale):
    """MambaIRv2 model for image restoration with combined window sizes and TTA."""
    
    def test(self):
        # 设置固定的窗口大小和权重
        crop_sizes = [100, 200, 300, 400, 500]
        weights = [1.0, 1.0, 1.2, 0.8, 0.8]  # 对应每个窗口大小的权重
        # crop_sizes = [200, 300, 400]
        # weights = [1.0, 1.2, 0.8]  # 对应每个窗口大小的权重
        
        # 归一化权重
        weights = [w/sum(weights) for w in weights]
        
        def inference_once(img_lqs, split_size):
            """
            批量处理相同尺寸的图像
            img_lqs: 图像列表[tensor1, tensor2, ...]，每个tensor的形状相同
            split_size: 分割大小
            """
            # 所有输入图像应当有相同的尺寸
            _, C, h, w = img_lqs[0].size()
            split_token_h = h // split_size + 1  
            split_token_w = w // split_size + 1  
            
            # padding
            mod_pad_h, mod_pad_w = 0, 0
            if h % split_token_h != 0:
                mod_pad_h = split_token_h - h % split_token_h
            if w % split_token_w != 0:
                mod_pad_w = split_token_w - w % split_token_w
            
            padded_images = []
            for img_lq in img_lqs:
                img = F.pad(img_lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
                padded_images.append(img)
            
            _, _, H, W = padded_images[0].size()
            split_h = H // split_token_h
            split_w = W // split_token_w
            
            shave_h = split_h // 10
            shave_w = split_w // 10
            scale = self.opt.get('scale', 1)
            ral = H // split_h
            row = W // split_w
            
            # 预先计算所有slices
            slices = []
            for i in range(ral):
                for j in range(row):
                    if i == 0 and i == ral - 1:
                        top = slice(i * split_h, (i + 1) * split_h)
                    elif i == 0:
                        top = slice(i*split_h, (i+1)*split_h+shave_h)
                    elif i == ral - 1:
                        top = slice(i*split_h-shave_h, (i+1)*split_h)
                    else:
                        top = slice(i*split_h-shave_h, (i+1)*split_h+shave_h)
                    if j == 0 and j == row - 1:
                        left = slice(j*split_w, (j+1)*split_w)
                    elif j == 0:
                        left = slice(j*split_w, (j+1)*split_w+shave_w)
                    elif j == row - 1:
                        left = slice(j*split_w-shave_w, (j+1)*split_w)
                    else:
                        left = slice(j*split_w-shave_w, (j+1)*split_w+shave_w)
                    slices.append((top, left, i, j))
            
            # 处理每个图像的结果
            all_images_results = []
            
            # 使用模型
            if hasattr(self, 'net_g_ema'):
                model = self.net_g_ema
            else:
                model = self.net_g
                
            model.eval()
            
            with torch.no_grad():
                # 对每个图像分别提取图像块
                for img_idx, img in enumerate(padded_images):
                    # 提取所有图像块
                    img_chops = []
                    for top, left, _, _ in slices:
                        img_chops.append(img[..., top, left])
                    
                    # 批处理相似尺寸的图像块
                    batch_size = 4  # 可以根据GPU内存调整
                    outputs = []
                    
                    # 按照块尺寸对图像块分组
                    chop_sizes = {}
                    for idx, chop in enumerate(img_chops):
                        size_key = (chop.size(2), chop.size(3))
                        if size_key not in chop_sizes:
                            chop_sizes[size_key] = []
                        chop_sizes[size_key].append((idx, chop))
                    
                    # 对每个尺寸组批处理
                    for size_key, chop_group in chop_sizes.items():
                        indices = [item[0] for item in chop_group]
                        chops = [item[1] for item in chop_group]
                        
                        # 批处理
                        for i in range(0, len(chops), batch_size):
                            batch_chops = chops[i:i+batch_size]
                            if len(batch_chops) == 1:
                                # 单个样本
                                out, _, _ = model(batch_chops[0])
                                # out = model(batch_chops[0])
                                outputs.append((indices[i], out))
                            else:
                                # 批处理
                                batch = torch.cat(batch_chops, dim=0)
                                out, _, _ = model(batch)
                                # out = model(batch)
                                for j in range(len(batch_chops)):
                                    if i+j < len(indices):
                                        outputs.append((indices[i+j], out[j:j+1]))
                    
                    # 按原始索引排序输出
                    outputs.sort(key=lambda x: x[0])
                    outputs = [item[1] for item in outputs]
                    
                    # 重建完整图像
                    _img = torch.zeros(1, C, H * scale, W * scale, device=self.lq.device)
                    
                    for idx, (top, left, i, j) in enumerate(slices):
                        top_out = slice(i * split_h * scale, (i + 1) * split_h * scale)
                        left_out = slice(j * split_w * scale, (j + 1) * split_w * scale)
                        if i == 0:
                            _top = slice(0, split_h * scale)
                        else:
                            _top = slice(shave_h*scale, (shave_h+split_h)*scale)
                        if j == 0:
                            _left = slice(0, split_w*scale)
                        else:
                            _left = slice(shave_w*scale, (shave_w+split_w)*scale)
                        _img[..., top_out, left_out] = outputs[idx][..., _top, _left]
                    
                    # 裁剪掉padding部分
                    _img = _img[:, :, 0:h * scale, 0:w * scale]
                    all_images_results.append(_img)
            
            if not hasattr(self, 'net_g_ema'):
                model.train()
                
            return all_images_results

        # 划分相似变换为组，以便批处理
        # 组1: 不涉及转置的变换 (尺寸不变)
        non_transpose_transforms = [
            lambda x: x,                                    # 原始
            lambda x: x.flip(-1),                          # 水平翻转
            lambda x: x.flip(-2),                          # 垂直翻转
            lambda x: x.flip(-1).flip(-2),                 # 水平+垂直翻转
        ]
        
        non_transpose_inv_transforms = [
            lambda x: x,                                    # 原始
            lambda x: x.flip(-1),                          # 水平翻转
            lambda x: x.flip(-2),                          # 垂直翻转
            lambda x: x.flip(-1).flip(-2),                 # 水平+垂直翻转
        ]
        
        # 组2: 涉及转置的变换 (宽高交换)
        transpose_transforms = [
            lambda x: x.permute(0, 1, 3, 2),              # 转置
            lambda x: x.permute(0, 1, 3, 2).flip(-1),     # 转置+水平翻转
            lambda x: x.permute(0, 1, 3, 2).flip(-2),     # 转置+垂直翻转
            lambda x: x.permute(0, 1, 3, 2).flip(-1).flip(-2)  # 转置+水平+垂直翻转
        ]
        
        transpose_inv_transforms = [
            lambda x: x.permute(0, 1, 3, 2),              # 转置
            lambda x: x.flip(-1).permute(0, 1, 3, 2),     # 水平翻转+转置
            lambda x: x.flip(-2).permute(0, 1, 3, 2),     # 垂直翻转+转置
            lambda x: x.flip(-1).flip(-2).permute(0, 1, 3, 2)  # 水平+垂直翻转+转置
        ]
        
        # 存储所有输出
        final_output = None
        total_weight = 0
        
        # 输出处理进度信息
        total_iterations = len(crop_sizes) * 2  # 两组变换
        current_iteration = 0
        
        # print(f"Starting inference with 8 transforms and {len(crop_sizes)} window sizes...")
        
        # 处理不涉及转置的变换 (批处理)
        for c_idx, (crop_size, weight) in enumerate(zip(crop_sizes, weights)):
            current_iteration += 1
            # print(f"Processing non-transpose transforms (4), window {c_idx+1}/{len(crop_sizes)} - {current_iteration}/{total_iterations}")
            
            # 对每个crop_size，同时应用4种不涉及转置的变换
            transformed_images = []
            for transform in non_transpose_transforms:
                transformed_images.append(transform(self.lq))
            
            # 批量推理
            outputs = inference_once(transformed_images, crop_size)
            
            # 应用逆变换并累加结果
            for output, inv_transform in zip(outputs, non_transpose_inv_transforms):
                output = inv_transform(output) * weight
                if final_output is None:
                    final_output = output
                else:
                    final_output += output
                total_weight += weight
            
            # 释放内存
            torch.cuda.empty_cache()
        
        # 处理涉及转置的变换 (批处理) - 改进的部分
        for c_idx, (crop_size, weight) in enumerate(zip(crop_sizes, weights)):
            current_iteration += 1
            # print(f"Processing transpose transforms (4), window {c_idx+1}/{len(crop_sizes)} - {current_iteration}/{total_iterations}")
            
            # 对每个crop_size，同时应用4种涉及转置的变换
            transformed_images = []
            for transform in transpose_transforms:
                transformed_images.append(transform(self.lq))
            
            # 批量推理
            outputs = inference_once(transformed_images, crop_size)
            
            # 应用逆变换并累加结果
            for output, inv_transform in zip(outputs, transpose_inv_transforms):
                output = inv_transform(output) * weight
                final_output += output
                total_weight += weight
            
            # 释放内存
            torch.cuda.empty_cache()
        
        # 计算最终输出
        self.output = final_output / total_weight
        # print("Inference completed successfully!")

