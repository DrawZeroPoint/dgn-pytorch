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
