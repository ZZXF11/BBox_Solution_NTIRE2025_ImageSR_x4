import os.path
import logging
import torch
import argparse
import json
import glob

from pprint import pprint
from utils.model_summary import get_model_flops
from utils import utils_logger
from utils import utils_image as util
import cv2
import numpy as np
from tqdm import tqdm

def calculate_adaptive_weights(outputs):
    stacked_outputs = np.stack(outputs, axis=0)
    mean_output = np.mean(stacked_outputs, axis=0)

    similarity_scores = []
    for output in outputs:
        diff = np.mean(np.abs(output - mean_output))
        similarity = 1.0 / (diff + 1e-8)
        similarity_scores.append(similarity)

    total_score = sum(similarity_scores)
    weights = [score / total_score for score in similarity_scores]

    print("MSE自适应集成权重:", [f"{w:.4f}" for w in weights])

    final_output = sum(output * weight for output, weight in zip(outputs, weights))
    
    return final_output


def select_model(args, device):
    # Model ID is assigned according to the order of the submissions.
    # Different networks are trained with input range of either [0,1] or [0,255]. The range is determined manually.
    model_id = args.model_id
    if model_id == 15:
        from models.team15_SMT import main as MambaIRv2
        # name_mamba = f"{model_id:02}_MambaIRv2_baseline"
        model_path_mamba = os.path.join('model_zoo', 'team15_SMT_Mamba.pth')
        name_mamba = './tmp/mamba/'
        model_func_mamba = MambaIRv2

        from models.team15_SMT_HAT import main as HAT
        # name_hat = f"{model_id:02}_HAT_baseline"
        model_path_hat = os.path.join('model_zoo', 'team15_SMT_HAT.pth')
        name_hat = './tmp/hat/'
        model_func_hat = HAT

        return model_func_mamba, model_func_hat, model_path_mamba, model_path_hat, name_mamba, name_hat

    # if model_id == 15:
    #     from models.team15_SMT import main as MambaIRv2
    #     name = f"{model_id:02}_MambaIRv2_baseline"
    #     model_path = os.path.join('model_zoo', 'team15_SMT_Mamba.pth')
    #     model_func = MambaIRv2

    else:
        raise NotImplementedError(f"Model {model_id} is not implemented.")

    return model_func, model_path, name


def run(model_func, model_name, model_path, device, args, mode="test"):
    # --------------------------------
    # dataset path
    # --------------------------------
    if mode == "valid":
        data_path = args.valid_dir
    elif mode == "test":
        data_path = args.test_dir
    assert data_path is not None, "Please specify the dataset path for validation or test."
    
    # save_path = os.path.join(args.save_dir, model_name, mode)
    save_path = os.path.join(model_name, mode)
    util.mkdir(save_path)

    # 获取当前进程的rank
    rank = 0
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])

    # 只在rank 0进程上进行计时
    if rank == 0:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    # start.record()
    model_func(model_dir=model_path, input_path=data_path, output_path=save_path, device=device)

    # end.record()
    # torch.cuda.synchronize()
    # print(f"Model {model_name} runtime (Including I/O): {start.elapsed_time(end)} ms")
    # 只在rank 0进程上完成计时并打印
    if rank == 0:
        end.record()
        torch.cuda.synchronize()
        try:
            print(f"Model {model_name} runtime (Including I/O): {start.elapsed_time(end)} ms")
        except Exception as e:
            print(f"计时错误: {e}")
            print(f"已完成模型 {model_name} 处理，但无法测量运行时间")


def main(args):
    utils_logger.logger_info("NTIRE2025-ImageSRx4", log_path="NTIRE2025-ImageSRx4.log")
    logger = logging.getLogger("NTIRE2025-ImageSRx4")

    # --------------------------------
    # basic settings
    # --------------------------------
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_dir = os.path.join(os.getcwd(), "results.json")
    if not os.path.exists(json_dir):
        results = dict()
    else:
        with open(json_dir, "r") as f:
            results = json.load(f)

    # --------------------------------
    # load model
    # --------------------------------
    # model_func, model_path, model_name = select_model(args, device)
    model_func_mamba, model_func_hat, model_path_mamba, model_path_hat, name_mamba, name_hat = select_model(args, device)
    # logger.info(model_name)
    logger.info(name_mamba)
    logger.info(name_hat)

    # if model not in results:
    if args.valid_dir is not None:
        # run(model_func, model_name, model_path, device, args, mode="valid")
        run(model_func_hat, name_hat, model_path_hat, device, args, mode="valid")
        run(model_func_mamba, name_mamba, model_path_mamba, device, args, mode="valid")
        
    if args.test_dir is not None:
        # run(model_func, model_name, model_path, device, args, mode="test")
        rank = 0
        if 'RANK' in os.environ:
            rank = int(os.environ['RANK'])
        if rank == 0:
            run(model_func_hat, name_hat, model_path_hat, device, args, mode="test")
        run(model_func_mamba, name_mamba, model_path_mamba, device, args, mode="test")

    rank = 0
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    if rank == 0:
        if args.test_dir is not None:
            mode = 'test'
        if args.valid_dir is not None:
            mode = 'valid'
        mamba_result_path = os.path.join(name_mamba, mode)
        hat_result_path = os.path.join(name_hat, mode)

        mamba_output = os.listdir(mamba_result_path)
        hat_output = os.listdir(hat_result_path)

        assert len(mamba_output) == len(hat_output), "两文件夹中的图像数量必须相同"

        util.mkdir(os.path.join(args.save_dir, mode))

        for i in tqdm(range(len(mamba_output))):
            img1 = cv2.imread(os.path.join(mamba_result_path, mamba_output[i]))
            img2 = cv2.imread(os.path.join(hat_result_path, hat_output[i]))
            
            if img1 is None or img2 is None:
                print(f"无法读取图像: {mamba_output[i]} 或 {hat_output[i]}")
                continue

            if img1.shape != img2.shape:
                print(f"图像尺寸不一致: {mamba_output[i]} 和 {hat_output[i]}")
                continue

            model_outputs = [img1, img2]
            fused_image = calculate_adaptive_weights(model_outputs)
            
            fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
            
            save_path = os.path.join(args.save_dir, mode)
            
            fused_image_name = os.path.join(save_path, hat_output[i])
            
            cv2.imwrite(fused_image_name, fused_image)

        print("融合后的图像已保存至:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("NTIRE2025-ImageSRx4")
    parser.add_argument("--valid_dir", default=None, type=str, help="Path to the validation set")
    parser.add_argument("--test_dir", default=None, type=str, help="Path to the test set")
    parser.add_argument("--save_dir", default="NTIRE2025-ImageSRx4/results", type=str)
    parser.add_argument("--model_id", default=0, type=int)

    args = parser.parse_args()
    pprint(args)

    main(args)
