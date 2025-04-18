## :book: Weakly-Supervised 3D Scene Graph Generation via Visual-Linguistic Assisted Pseudo-labeling
<image src="Model_Structure.jpg" width="100%">
<p align="center">
  <small>:fire: If you found the training scheme in 3D-VLAP is useful, please help to :star: it or recommend it to your friends. Thanks:fire:</small>
</p>


# Introduction
This is a release of the code of our paper **_Weakly-Supervised 3D Scene Graph Generation via Visual-Linguistic Assisted Pseudo-labeling_**.

Authors:

[[arxiv]](https://arxiv.org/abs/2404.02527)  [[code]]()

# Dependencies
```bash
conda create -n 3dvlap python=3.8
conda activate 3dvlap
pip install -r requirement.txt
```
# Prepare the data
A. Download 3Rscan and 3DSSG-Sub Annotation, you can follow [3DSSG](https://github.com/ShunChengWu/3DSSG#preparation)

B. Generate 2D Multi View Image
```bash
# you should motify the path in pointcloud2image_no_fea_match.py into your own path
python data/pointcloud2image_no_fea_match.py
```

C. You should arrange the file location like this
```
data
  3DSSG_subset
    relations.txt
    classes.txt
    
  3RScan
    0a4b8ef6-a83a-21f2-8672-dce34dd0d7ca
      multi_view_no_fea_match_top5
      labels.instances.align.annotated.v2.ply
    ...  
      
```

# Weights
You can download pretrained weights in:
Link: https://pan.baidu.com/s/11pEHmtKPu-pqO3RlR3MlyQ?pwd=x41c
Code: x41c 

# Run Code
```bash
# Train
python -m main --mode train --config <config_path> --exp <exp_name>
# Eval
python -m main --mode eval --config <config_path> --exp <exp_name>
```

# Paper

If you find the code useful please consider citing our [paper](https://arxiv.org/abs/2404.02527):
```
@article{wang2024weakly,
  title={Weakly-Supervised 3D Scene Graph Generation via Visual-Linguistic Assisted Pseudo-labeling},
  author={Wang, Xu and Li, Yifan and Zhang, Qiudan and Wu, Wenhui and Li, Mark Junjie and Ma, Lin and Jiang, Jianmin},
  journal={IEEE Transactions on Multimedia},
  year={2024},
  publisher={IEEE}
}
```


# Acknowledgement
This repository is partly based on [3DSSG](https://github.com/ShunChengWu/3DSSG)， [CLIP](https://github.com/openai/CLIP) and [VL-SAT](https://github.com/wz7in/CVPR2023-VLSAT) repositories.
