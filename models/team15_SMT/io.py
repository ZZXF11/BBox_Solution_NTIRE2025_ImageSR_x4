import logging
import torch
import os
from os import path as osp
import sys
import glob
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import numpy as np
import cv2
import math
from torchvision.utils import make_grid

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 导入必要的工具函数
from models.team15_SMT.models import build_model
from models.team15_SMT.utils.options import dict2str

# 设置随机种子，但保留cudnn.benchmark开启
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 不修改cudnn设置，避免影响性能或引起CUDA错误
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# 自定义工具函数
def get_root_logger(logger_name='basicsr', log_level=logging.INFO):
    """获取root logger"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    
    # 避免重复的处理器
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """将tensor转换为图像numpy数组"""
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # 灰度图像
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """写入图像到文件"""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create directory {dir_name}: {e}")
            if not os.path.exists(dir_name):
                raise
    
    try:
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        
        ok = cv2.imwrite(file_path, img, params)
        if not ok:
            raise IOError('Failed in writing images.')
        return ok
    except Exception as e:
        print(f"Error writing image to {file_path}: {e}")
        return False

# 设置默认配置
def get_default_config():
    """返回写死的默认配置"""
    return {
        # general settings
        "name": "mambairv2",
        "model_type": "MambaIRv2Model_combined",
        "scale": 4,
        "num_gpu": 1,  # 默认使用1个GPU
        "manual_seed": 42,  # 固定随机种子
        "dist": False,  # 默认不使用分布式处理
        "is_train": False,  # 表明是测试模式

        # network structures
        "network_g": {
            "type": "MambaIRv2_Multiscale",
            "upscale": 4,
            "in_chans": 3,
            "img_size": 128,
            "img_range": 1.0,
            "embed_dim": 174,
            "d_state": 16,
            "depths": [6, 6, 6, 6, 6, 6, 6, 6, 6],
            "num_heads": [6, 6, 6, 6, 6, 6, 6, 6, 6],
            "window_size": 16,
            "inner_rank": 64,
            "num_tokens": 128,
            "convffn_kernel_size": 5,
            "mlp_ratio": 2.0,
            "upsampler": "pixelshuffle",
            "resi_connection": "1conv"
        },

        # validation settings
        "val": {
            "save_img": True,
            "suffix": 200,
            "pbar": True,
            "metrics": {
                "psnr": {
                    "type": "calculate_psnr",
                    "crop_border": 4,
                    "test_y_channel": True
                },
                "ssim": {
                    "type": "calculate_ssim",
                    "crop_border": 4,
                    "test_y_channel": True
                }
            }
        },

        # dist training settings
        "dist_params": {
            "backend": "nccl",
            "port": 29590
        }
    }

def init_dist_env():
    """初始化分布式环境，检测是否在分布式模式下运行"""
    # 检查是否在分布式环境中
    is_distributed = False
    rank = 0
    world_size = 1
    local_rank = 0
    
    # 检查相关环境变量是否存在
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        is_distributed = True
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # 初始化进程组
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return is_distributed, rank, world_size, local_rank

def log_with_timestamp(message, rank=0):
    """添加时间戳的日志函数"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [GPU-{rank}] {message}")

def process_image(model, img_path, output_dir, device, opt, rank):
    """处理单张图像，保持处理方式一致稳定"""
    try:
        # 读取图像
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            log_with_timestamp(f"无法读取图像: {img_path}", rank)
            return None
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 标准化到[0,1]
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_chw = np.transpose(img_float, (2, 0, 1))
        
        # 转为tensor并移至设备
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
        
        # 构建模型输入
        data = {'lq': img_tensor, 'lq_path': img_path}
        
        # 使用模型处理图像
        model.feed_data(data)
        with torch.no_grad():
            model.test()
        
        # 获取结果
        visuals = model.get_current_visuals()
        output_tensor = visuals['result']
        
        # 转换为图像
        sr_img = tensor2img(output_tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
        
        # 获取图像名并保存
        img_name = osp.splitext(osp.basename(img_path))[0]
        save_img_path = osp.join(output_dir, f'{img_name}.png')
        
        # 使用固定参数保存图像
        params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        imwrite(sr_img, save_img_path, params)
        
        return save_img_path
        
    except Exception as e:
        log_with_timestamp(f"处理图像时发生错误: {e}", rank)
        import traceback
        log_with_timestamp(traceback.format_exc(), rank)
        return None

def main(model_dir, input_path, output_path, model_id=None, device=None):
    """
    主函数，处理图像超分辨率
    
    参数:
        model_dir: 模型权重文件路径
        input_path: 输入图像目录路径
        output_path: 输出图像保存目录路径（已由上层脚本创建）
        model_id: 模型ID，仅用于日志
        device: 指定设备，None则自动选择
    """
    # 设置随机种子
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 检测并初始化分布式环境
    is_distributed, rank, world_size, local_rank = init_dist_env()
    
    # 设置设备
    if device is None:
        if is_distributed:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 设置CUDA环境 - 不改变cudnn设置
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True  # 保持benchmark开启
    
    log_with_timestamp(f"使用设备: {device}, 分布式模式: {is_distributed}, 进程数: {world_size}", rank)
    log_with_timestamp(f"已设置随机种子为: 42", rank)
    
    # 获取默认配置
    opt = get_default_config()
    opt['dist'] = is_distributed
    opt['rank'] = rank
    opt['world_size'] = world_size
    opt['local_rank'] = local_rank
    opt['path'] = {
        'pretrain_network_g': model_dir,
        'strict_load_g': True,
        'visualization': output_path  # 直接使用上层脚本提供的路径
    }
    
    # 确保所有进程同步
    if is_distributed:
        dist.barrier()
    
    # 只在主进程上创建日志
    if rank == 0:
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)
        logger.info(f"使用模型: {model_dir}")
        logger.info(f"输出目录: {output_path}")
        logger.info(f"随机种子: 42")
    else:
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO)
    
    # 创建模型
    model = build_model(opt)
    
    # 查找输入图像
    input_img_list = sorted(glob.glob(osp.join(input_path, '*.[jpJP][pnPN]*[gG]')))
    
    if rank == 0:
        log_with_timestamp(f"共发现 {len(input_img_list)} 张图像需要处理", rank)
    
    # 在分布式环境中，按GPU分配图像
    if is_distributed and world_size > 1:
        # 计算当前进程应处理的图像索引
        indices = list(range(len(input_img_list)))
        per_gpu = len(indices) // world_size
        remainder = len(indices) % world_size
        
        # 计算当前GPU的分配
        start_idx = rank * per_gpu + min(rank, remainder)
        end_idx = start_idx + per_gpu + (1 if rank < remainder else 0)
        
        # 获取当前GPU要处理的图像
        assigned_images = [input_img_list[i] for i in indices[start_idx:end_idx]]
        log_with_timestamp(f"GPU-{rank} 分配到 {len(assigned_images)} 张图像", rank)
    else:
        assigned_images = input_img_list
    
    # 使用tqdm创建进度条
    progress_bar = tqdm(total=len(assigned_images), desc=f"GPU-{rank} 处理进度", 
                        disable=rank != 0, unit="图片")
    
    # 处理分配的图像
    for i, img_path in enumerate(assigned_images):
        img_name = osp.basename(img_path)
        log_with_timestamp(f"正在处理 [{i+1}/{len(assigned_images)}]: {img_name}", rank)
        
        save_img_path = process_image(model, img_path, output_path, device, opt, rank)
        if save_img_path:
            log_with_timestamp(f"已保存到: {save_img_path}", rank)
        
        # 更新进度条
        progress_bar.update(1)
    
    # 关闭进度条
    progress_bar.close()
    
    log_with_timestamp(f"GPU-{rank} 已完成所有任务", rank)
    
    # 在分布式环境中，等待所有进程完成
    if is_distributed and world_size > 1:
        dist.barrier()
        # 清理分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()
    
    if rank == 0:
        log_with_timestamp("所有图像处理完成", rank)


if __name__ == '__main__':
    import argparse
    import random
    
    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    parser = argparse.ArgumentParser(description='超分辨率推理程序')
    parser.add_argument('--model', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--input', type=str, required=True, help='输入图像目录路径')
    parser.add_argument('--output', type=str, required=True, help='输出图像保存目录路径')
    parser.add_argument('--model_id', type=str, default=None, help='模型ID，用于日志')
    parser.add_argument('--device', type=str, default=None, help='指定设备，默认自动选择')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0, 
                      help='分布式训练的本地rank，通常由启动器自动设置')
    
    args = parser.parse_args()
    
    # 如果指定了设备，则设置设备
    device = None
    if args.device is not None:
        device = torch.device(args.device)
    
    main(
        model_dir=args.model,
        input_path=args.input,
        output_path=args.output,
        model_id=args.model_id,
        device=device
    )