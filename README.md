# [NTIRE 2025 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/) -[BBox Team 15]

The souce files of the factsheet are available at `factsheet`.

## How to test the team15 model?
### Step1: load docker image
docker load --input bbox.tar
 [BaiduYun:r6ji](https://pan.baidu.com/s/1lE0eDndu55Z5rmUqF_d6Kg?pwd=r6ji)

### Step2: download the pretrained model
- The pretrained models are available at `model_zoo/team15_SMT/team15_SMT_HAT.txt` and `model_zoo/team15_SMT/team15_SMT_Mamba.txt`.
- Download the two pretrained models and put them into `model_zoo`.

### Step3: inference
1. For single GPU:
```bash
CUDA_VISIBLE_DEVICES=<gpu_id> torchrun --nproc_per_node=1 test.py --model_id 15 test_dir [path to test data dir] --save_dir [path to your save dir]
```

2. For multi-GPU:

```bash
CUDA_VISIBLE_DEVICES=<gpu_ids> torchrun --nproc_per_node=<num_gpus> test.py --model_id 15 test_dir [path to test data dir] --save_dir [path to your save dir]
```

If you are using a different number or type of GPUs, please adjust the `nproc_per_node` parameter accordingly to match your hardware configuration.

**Note:🚨** Inference speed on a single GPU will be relatively slow. Specifically, we use **10× A100 GPUs** for inference, which takes approximately **40 minutes**. The extended inference time is due to the **multi-window self-ensemble strategy** applied to two models, along with the final **model ensemble integration**.  
