# MVCNN_NeRF: Fusing Multi-view CNN and Neural Radiance Fields for 3D Object Classification

## Folder Overview

1. **COLMAP_PosesGenerationImages&PosesBoundsFiles**  
   This folder contains the original images used to generate camera poses and depth bounds using [COLMAP](https://colmap.github.io/). These outputs are essential for building the `poses_bounds.npy` file used in the training pipeline of NeRF-like models.

   The file `poses_bounds.npy` encodes three critical components:  
   - **Camera positions and orientations** – indicating where each image was captured and its viewing direction.  
   - **Viewing relationships** between training images – enabling the model to reason about multi-view geometry.  
   - **Near and far depth bounds** – controlling the range of volume rendering during ray sampling.

2. **MVCNN_origin**  
   This folder contains the original MVCNN implementation from [jongchyisu/mvcnn_pytorch](https://github.com/jongchyisu/mvcnn_pytorch), which was used to train the ScanObjectNN dataset and benchmark baseline classification performance.  
   ➤ *The original LICENSE file from the MVCNN repository is preserved here for compliance.*

3. **ObtainFeatures_from_NeRF**  
   This folder includes the modified version of NeRF from [kwea123/nerf_pl](https://github.com/kwea123/nerf_pl), adapted to extract features from the 8th layer of the MLP during NeRF’s training on the ScanObjectNN dataset. These features are saved locally as `.pt` files for subsequent fusion.  
   ➤ *This folder also preserves the LICENSE file of the original NeRF repository.*

4. **mvcnn_nerf**  
   This is the final integrated framework developed to fuse the features from MVCNN and NeRF. The combined features are used for the final classification task on ScanObjectNN. This module contains new scripts that load and fuse `.pt` features from NeRF and activations from MVCNN for joint training and evaluation.

## Citation of Reused Code

We respectfully acknowledge and reuse the following open-source repositories in our project:

- NeRF Implementation:  
  [kwea123/nerf_pl](https://github.com/kwea123/nerf_pl)  
  Licensed under the [MIT License](https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/LICENSE).

- MVCNN PyTorch Implementation:  
  [jongchyisu/mvcnn_pytorch](https://github.com/jongchyisu/mvcnn_pytorch)  
  Licensed under the [MIT License](https://github.com/jongchyisu/mvcnn_pytorch/blob/09a3b5134d92a35da31e4247b20c3c814b41f753/LICENSE).

Please refer to their repositories and licenses for original contributions.

## License Notice

This project is released for research and non-commercial use only. All third-party code is acknowledged and redistributed with the original license terms in their respective folders.

If you use this repository in your work, please consider citing the corresponding papers and repositories above.
