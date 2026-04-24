<h2 align="center"> <a href="https://easi3r.github.io/">Easi3R: Estimating Disentangled Motion from DUSt3R Without Training</a>
</h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2503.24391-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2503.24391) 
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://easi3r.github.io/) 
[![Demo](https://img.shields.io/badge/%20Interactive-Demo-ffdf0f)](https://easi3r.github.io/interactive.html)
[![X](https://img.shields.io/badge/@Xingyu%20Chen-black?logo=X)](https://x.com/RoverXingyu)  [![Bluesky](https://img.shields.io/badge/@Xingyu%20Chen-white?logo=Bluesky)](https://bsky.app/profile/xingyu-chen.bsky.social)


[Xingyu Chen](https://rover-xingyu.github.io/),
[Yue Chen](https://fanegg.github.io/),
[Yuliang Xiu](https://xiuyuliang.cn/),
[Andreas Geiger](https://www.cvlibs.net/),
[Anpei Chen](https://apchenstu.github.io/)
</h5>

<div align="center">
Easi3R is a simple training-free approach adapting DUSt3R for dynamic scenes.
</div>
<br>


https://github.com/user-attachments/assets/80091ab5-2576-4b48-b880-a230376a6edf



## Getting Started

### Installation

1. Clone Easi3R.
```bash
git clone https://github.com/Inception3D/Easi3R.git
cd Easi3R
```

2. Create the environment, here we show an example using conda.
```bash
conda create -n easi3r python=3.10 cmake=3.31
conda activate easi3r
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia  # use the correct version of cuda for your system
pip install -r requirements.txt
# install 4d visualization tool
pip install -e viser
# install SAM2
pip install -e third_party/sam2 --verbose
# compile the cuda kernels for RoPE (as in CroCo v2).
# DUST3R relies on RoPE positional embeddings for which you can compile some cuda kernels for faster runtime.
cd croco/models/curope/
python setup.py build_ext --inplace
cd ../../../
```

### Download Checkpoints

To download the weights of DUSt3R, MonST3R, RAFT and SAM2, run the following commands:
```bash
# download the weights
cd data
bash download_ckpt.sh
cd ..
```

### Inference

To run the interactive inference demo, you can use the following command:
```bash
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5 python demo.py \
    --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth 
# To change backbone, --weights checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth
```

The results will be saved in the `demo_tmp/{Sequence Name}` (by default is `demo_tmp/NULL`) folder for future visualization.

You can also run the inference in a non-interactive mode:
```bash
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5 python demo.py --input demo_data/dog-gooses \
    --output_dir demo_tmp --seq_name dog-gooses \
    --weights checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth 
# To change backbone, --weights checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth
# To use SAM2, add: --sam2_mask_refine
# use video as input: --input demo_data/dog-gooses.mp4 
# reduce the memory cost: set maximum number of frames used from video --num_frames 65 
# faster video option: down sample the video fps to --fps 5
```

### Visualization

To visualize the interactive 4D results, you can use the following command:
```bash
python viser/visualizer.py --data demo_tmp/dog-gooses --port 9081
```

## Evaluation

We provide here an example on the **DAVIS** dataset. 

First, download the dataset:
```bash
cd data; python download_prepare_davis.py; cd ..
```

Then, run the evaluation script:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29604 launch.py \
    --mode=eval_pose \
    --pretrained="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=davis --output_dir="results/davis/easi3r_dust3r" \
    --use_atten_mask
# To change backbone, --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"
# To use SAM2, add: --sam2_mask_refine
```
If you just need dynamic mask, execute:
```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=29604 launch.py \
    --mode=eval_pose --n_iter 0 \
    --pretrained="checkpoints/MonST3R_PO-TA-S-W_ViTLarge_BaseDecoder_512_dpt.pth"   \
    --eval_dataset=davis --output_dir="results/davis/easi3r_monst3r_sam" \
    --use_atten_mask --sam2_mask_refine
```
The results will be saved in the `results/davis/easi3r_monst3r_sam` folder. You could then run `python mask_metric.py --results_path results/davis/easi3r_monst3r_sam` to evaluate the mask results, and run `python vis_attention.py --method_name easi3r_monst3r_sam --base_output_dir results/visualization` to see the visualization of attention as in the [webpage](https://easi3r.github.io/).


#### For the complete evaluation. Please refer to the [evaluation_script.md](data/evaluation_script.md) for more details.



## Acknowledgements
Our code is based on [DUSt3R](https://github.com/naver/dust3r), [MonST3R](https://github.com/Junyi42/monst3r), [DAS3R](https://github.com/kai422/DAS3R), [Spann3R](https://github.com/HengyiWang/spann3r), [CUT3R](https://github.com/CUT3R/CUT3R), [LEAP-VO](https://github.com/chiaki530/leapvo), [Shape of Motion](https://github.com/vye16/shape-of-motion/), [TAPVid-3D](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d), [CasualSAM](https://github.com/ztzhang/casualSAM) and [Viser](https://github.com/nerfstudio-project/viser). We thank the authors for their excellent work!

## Citation

If you find our work useful, please cite:

```bibtex
@article{chen2025easi3r,
    title={Easi3R: Estimating Disentangled Motion from DUSt3R Without Training},
    author={Chen, Xingyu and Chen, Yue and Xiu, Yuliang and Geiger, Andreas and Chen, Anpei},
    journal={arXiv preprint arXiv:2503.24391},
    year={2025}
    }
```
