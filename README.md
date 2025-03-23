# [NTIRE 2025 Challenge on Image Super-Resolution (x4)](https://cvlai.net/ntire/2025/) @ [CVPR 2025](https://cvpr.thecvf.com/) -[BBox Team 15]


## How to test the team15 model?
### step1: load docker image
docker load --input bbox.tar
 (baidu cloud: https://pan.baidu.com/s/1lE0eDndu55Z5rmUqF_d6Kg?pwd=r6ji passwd: r6ji )

### step2: download the pretrained model
The pretrained models are available at Google Drive or Baidu Netdisk (access code: qyrl).


### step3: inference
torchrun --nproc_per_node=10 test.py --model_id 15 --test_dir /NTIRE2025_SR_Challenge/DIV2K_test_LR --save_dir

```bash
    torchrun --nproc_per_node=10 test.py --model_id 15 test_dir [path to test data dir] --save_dir [path to your save dir]
```

If you are using a different number or type of GPUs, please adjust the `nproc_per_node` parameter accordingly to match your hardware configuration.

**Note:🚨** We use **10× A100 GPUs** for inference, which takes approximately **40 minutes**. The extended inference time is due to the **multi-window self-ensemble strategy** applied to two models, along with the final **model ensemble integration**.  
