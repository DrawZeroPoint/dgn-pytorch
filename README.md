<div align="center">
<img src="https://github.com/DrawZeroPoint/dgn-pytorch/blob/master/dong1.png" width="800" alt="HoPE" />
</div>

# dgn-pytorch
Code for the paper: Depth Generation Network (DGN): Estimating Real World Depth from Stereo and Depth Images

# Introduction
Humans can perceive the depth of a scene even from
a photo of it by leveraging prior knowledge and infer
the unseen area by reasoning from the adjacent, or
even distant part of the scene.
Target at such ability, we design a DGN to incorporate
3D senses to predict the depth by inferring to a
variational Bayesian model contains a interpreter
network and a generation network. We maximizing
the likelihood of the distribution over the generated
depth, and given the synthetic ground truth, we follow
the Expectation-Maximization scheme to organize and
optimize the proposed model. The resulting model can estimate the relative depth of the scene with a pair of uncalibrated stereo images.

## Cite
If you referred or used DGN in your article, please consider citing:
```
@inproceedings{dong2019depth,
  title={Depth Generation Network: Estimating Real World Depth from Stereo and Depth Images},
  author={Dong, Zhipeng and Gao, Yi and Ren, Qinyuan and Yan, Yunhui and Chen, Fei},
  booktitle={2019 International Conference on Robotics and Automation (ICRA)},
  pages={7201--7206},
  year={2019},
  organization={IEEE}
}
```
