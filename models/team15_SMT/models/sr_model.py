import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from models.team15_SMT.archs import build_network
from models.team15_SMT.losses import build_loss
from models.team15_SMT.metrics import calculate_metric
from models.team15_SMT.utils import get_root_logger, imwrite, tensor2img
from models.team15_SMT.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import json
import os


def create_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3):
    """创建高斯模糊核"""
    # 创建一维高斯核
    x = np.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
    kernel_1d = np.exp(-0.5 * x**2 / sigma**2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # 转换为二维核
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    
    # 转为PyTorch张量
    kernel = torch.FloatTensor(kernel_2d).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

class HFLoss(nn.Module):
    """高频损失"""
    def __init__(self, kernel_size=5, sigma=1.0):
        super(HFLoss, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = None
    
    def forward(self, x, y):
        """
        Args:
            x: 重建的SR图像
            y: 目标HR图像
        """
        b, c, h, w = x.size()
        
        # 延迟初始化kernel，确保它在正确的设备上
        if self.kernel is None or self.kernel.device != x.device:
            self.kernel = create_gaussian_kernel(self.kernel_size, self.sigma, c).to(x.device)
        
        # 高斯模糊
        padding = self.kernel_size // 2
        x_blur = F.conv2d(x, self.kernel, padding=padding, groups=c)
        y_blur = F.conv2d(y, self.kernel, padding=padding, groups=c)
        
        # 高频信息提取
        x_high = x - x_blur
        y_high = y - y_blur
        
        # L1损失
        loss = torch.mean(torch.abs(x_high - y_high))
        return loss

class DCTLoss(nn.Module):
    """DCT频域损失"""
    def __init__(self):
        super(DCTLoss, self).__init__()
    
    def dct_2d(self, x, norm='ortho'):
        """
        2D离散余弦变换
        """
        x_shape = x.shape
        N1, N2 = x_shape[-2:]
        
        # 创建DCT-II变换矩阵
        n1 = torch.arange(N1, device=x.device).type(torch.float)
        k1 = torch.arange(N1, device=x.device).type(torch.float)
        dct1 = torch.cos(torch.outer(n1 + 0.5, k1) * torch.pi / N1)
        
        n2 = torch.arange(N2, device=x.device).type(torch.float)
        k2 = torch.arange(N2, device=x.device).type(torch.float)
        dct2 = torch.cos(torch.outer(n2 + 0.5, k2) * torch.pi / N2)
        
        # 标准化
        if norm == 'ortho':
            dct1[:, 0] *= 1.0 / np.sqrt(2)
            dct2[:, 0] *= 1.0 / np.sqrt(2)
            dct1 *= torch.sqrt(torch.tensor(2.0 / N1, device=x.device))
            dct2 *= torch.sqrt(torch.tensor(2.0 / N2, device=x.device))
        
        # 重塑输入以适应矩阵乘法
        x_reshaped = x.reshape(-1, N1, N2)
        
        # 应用DCT-II(先行后列)
        x_dct = torch.matmul(torch.matmul(dct1.t(), x_reshaped), dct2)
        
        # 重塑回原始形状
        return x_dct.reshape(x_shape)
    
    def forward(self, x, y):
        """
        计算DCT频域损失
        Args:
            x: 重建的SR图像
            y: 目标HR图像
        """
        # 分离通道
        x_list = torch.unbind(x, dim=1)
        y_list = torch.unbind(y, dim=1)
        
        loss = 0
        for x_c, y_c in zip(x_list, y_list):
            # 分别对每个通道应用DCT
            x_dct = self.dct_2d(x_c.unsqueeze(1))
            y_dct = self.dct_2d(y_c.unsqueeze(1))
            
            # 计算L1损失
            channel_loss = torch.mean(torch.abs(x_dct - y_dct))
            loss += channel_loss
        
        return loss / len(x_list)  # 平均所有通道的损失

@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        # DCT loss
        # 初始化高频损失
        self.cri_hf = HFLoss(kernel_size=5, sigma=1.0)
        self.l_hf_weight = 1.0  # α=1.0
        
        # 初始化DCT损失
        self.cri_dct = DCTLoss()
        self.l_dct_weight = 0.001  # β=0.001

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def calculate_edge_density(self, image, threshold=0.1):
        """计算图像的边缘密度，用于动态调整权重"""
        # 将图像转换为灰度
        if image.size(1) == 3:
            # 输入是 [B, 3, H, W]，转为 [B, 1, H, W]
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:1+1] + 0.114 * image[:, 2:2+1]
        else:
            # 如果已经是单通道，确保维度正确 [B, 1, H, W]
            gray = image if image.dim() == 4 else image.unsqueeze(1)
        
        # 现在 gray 的维度应该是 [B, 1, H, W]
        b, c, h, w = gray.size()  # 现在能正确解包为4个值
        
        # 计算Sobel边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
        
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        edge = torch.sqrt(edge_x**2 + edge_y**2)
        
        # 归一化
        edge = edge / edge.max() if edge.max() > 0 else edge
        
        # 计算边缘密度 (高于阈值的像素比例)
        density = (edge > threshold).float().mean()
        
        return density

    def calculate_edge_loss(self, sr, hr):
        """计算边缘一致性损失"""
        # 提取边缘
        sr_edge = self.extract_edges(sr)
        hr_edge = self.extract_edges(hr)
        
        # 计算边缘的L1损失
        return F.l1_loss(sr_edge, hr_edge)

    def extract_edges(self, img):
        """使用Sobel算子提取边缘"""
        b, c, h, w = img.size()
        
        # 转为灰度
        if c == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img
        
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                            dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                            dtype=torch.float32, device=img.device).unsqueeze(0).unsqueeze(0)
        
        edge_x = F.conv2d(gray, sobel_x, padding=1)
        edge_y = F.conv2d(gray, sobel_y, padding=1)
        
        # 计算梯度幅值
        edge = torch.sqrt(edge_x**2 + edge_y**2)
        
        return edge

    def get_adaptive_weight(self, current_iter, max_iter=100000, min_weight=0.05, max_weight=0.2):
        """根据训练进度动态调整感知损失权重"""
        # 权重随着训练进行逐渐增加，有助于先学习基本结构，再增强视觉质量
        ratio = min(current_iter / max_iter, 1.0)
        return min_weight + ratio * (max_weight - min_weight)
    
    def optimize_parameters_dct(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        # 高频损失
        if hasattr(self, 'cri_hf') and self.cri_hf:
            l_hf = self.cri_hf(self.output, self.gt)
            l_total += self.l_hf_weight * l_hf  # 权重α=1
            loss_dict['l_hf'] = l_hf
        
        # DCT频域损失
        if hasattr(self, 'cri_dct') and self.cri_dct:
            l_dct = self.cri_dct(self.output, self.gt)
            l_total += self.l_dct_weight * l_dct  # 权重β=0.001
            loss_dict['l_dct'] = l_dct

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def optimize_parameters_enhanced(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        
        # 动态权重调整
        # 计算边缘密度来调整高频损失权重
        edge_map = self.calculate_edge_density(self.gt)
        
        # 像素损失
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        # 高频损失 - 帮助恢复毛发、建筑细节
        if hasattr(self, 'cri_hf') and self.cri_hf:
            l_hf = self.cri_hf(self.output, self.gt)
            # 权重根据边缘密度动态调整，范围0.5-1.5
            hf_weight = torch.clamp(self.l_hf_weight * (1.0 + edge_map * 0.5), 0.5, 1.5)
            l_total += hf_weight * l_hf
            loss_dict['l_hf'] = l_hf
            # 如果hf_weight是标量或0维张量，直接使用
            if isinstance(hf_weight, float) or (isinstance(hf_weight, torch.Tensor) and hf_weight.dim() == 0):
                loss_dict['hf_weight'] = hf_weight
            else:
                # 如果是高维张量，取平均值
                loss_dict['hf_weight'] = hf_weight.mean() if hasattr(hf_weight, 'mean') else hf_weight
        
        # DCT频域损失
        if hasattr(self, 'cri_dct') and self.cri_dct:
            l_dct = self.cri_dct(self.output, self.gt)
            l_total += self.l_dct_weight * l_dct
            loss_dict['l_dct'] = l_dct
        
        # 感知损失
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                # 使用固定权重或自适应权重
                percep_weight = 0.1  # 固定权重
                l_total += percep_weight * l_percep
                loss_dict['l_percep'] = l_percep
                loss_dict['percep_weight'] = percep_weight
            if l_style is not None:
                style_weight = 0.5  # 对于毛发和建筑物等细节，样式损失很重要
                l_total += style_weight * l_style
                loss_dict['l_style'] = l_style
                loss_dict['style_weight'] = style_weight
        
        # 边缘感知损失 - 如果有定义
        if hasattr(self, 'cri_edge') and self.cri_edge:
            try:
                l_edge = self.cri_edge(self.output, self.gt)
                edge_weight = 0.1  # 边缘损失权重
                l_total += edge_weight * l_edge
                loss_dict['l_edge'] = l_edge
            except Exception as e:
                print(f"边缘损失计算错误: {e}")
                # 继续执行，不让边缘损失错误影响整体训练
        
        # 反向传播与优化
        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def optimize_parameters_fabric(self, current_iter, fabric):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        
        # 高频损失
        if hasattr(self, 'cri_hf') and self.cri_hf:
            l_hf = self.cri_hf(self.output, self.gt)
            l_total += self.l_hf_weight * l_hf  # 权重α=1
            loss_dict['l_hf'] = l_hf
        
        # DCT频域损失
        if hasattr(self, 'cri_dct') and self.cri_dct:
            l_dct = self.cri_dct(self.output, self.gt)
            l_total += self.l_dct_weight * l_dct  # 权重β=0.001
            loss_dict['l_dct'] = l_dct

        fabric.backward(l_total)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')


        # pbar = tqdm(total=len(dataloader), desc="Validation", unit="batch")
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            # import pdb; pdb.set_trace()
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
    
    
@MODEL_REGISTRY.register()
class SRModel_debug(SRModel):
    


    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        psnr_dict = {}
        # pbar = tqdm(total=len(dataloader), desc="Validation", unit="batch")
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()
            

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)
            
            if with_metrics:
                # calculate metrics
                metric_results_tmp = {}
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                    metric_results_tmp[name] = calculate_metric(metric_data, opt_)
                print(metric_results_tmp)
            
            psnr_dict[img_name] = metric_results_tmp['psnr']
            
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        # json_path = f'psnr_results_300.json'
        # with open(json_path, 'w') as f:
        #     json.dump(psnr_dict, f, indent=4)
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
