# SDSC-Net：Spectral Difference Encoding and Semantic Feature Calibration for Multispectral LiDAR Point Cloud Classification
In multispectral LiDAR point cloud classification, using spectral attributes implicitly through learned features may limit the ability of point cloud Transformers to distinguish land-cover classes with similar geometry. Local attention also restricts access to semantic context from spatially separated points. To address these limitations, we propose SDSC-Net, a Transformer network that combines spectral difference encoding with semantic feature calibration. A spectral difference-encoded self-attention (SDSA) module combines band-wise spectral differences with relative positions and feature relations to improve neighborhood modeling and distinguish classes with similar geometry. A semantic feature calibration (SFCM) module builds dynamic context prototypes from the current input and uses a gated residual mechanism to adaptively update point features. Together, these modules combine local spectral relationships with semantic context. On the Whitchurch–Stouffville (WS) and Nanjing University of Information Science and Technology (NUIST) datasets, SDSC-Net achieves overall accuracy (OA) scores of 95.84% and 80.35% and mean intersection over union (mIoU) scores of 82.44% and 61.27%, respectively, outperforming the compared methods. Visual comparisons further illustrate improved power line preservation on WS and more coherent building predictions on NUIST.

# ![Dataset](https://img.shields.io/badge/📁_-blue) Datasets
Nanjing University of Information Science and Technology campus (NUIST) and Canadian Whitchurch–Stouffville (WS) datasets [17]. NUIST has eight classes and eight input attributes (XYZ, RGB, Red Edge, IR); WS has five classes and six input attributes (XYZ, Green, SWIR, IR). NUIST regions 1 and 3–10 is used for training/validation, and regions 2 and 11 for testing. WS uses regions 1–8 for training, 11–12 for validation, and 9–10 for testing. 

<img width="803" height="554" alt="image" src="https://github.com/user-attachments/assets/444022d8-0704-4833-bdd0-a3fcbad651f1" />

<img width="725" height="538" alt="image" src="https://github.com/user-attachments/assets/28fc8f66-d9b7-44f5-a7b1-44dd2ba1a137" />

## 🚀 Method
We propose SDSC-Net, a U-Net-style encoder–decoder network for point cloud classification (Fig.1). Progressive downsampling extracts multiscale features, while upsampling and skip connections recover and fuse them. Spectral difference-encoded self-attention (SDSA) models neighborhood relationships using band-wise spectral differences. Semantic feature calibration (SFCM) builds point-level semantic references from dynamic context prototypes and adaptively calibrates point features through gated residual updates. A final fully connected classification layer outputs point-wise class prediction.

<img width="1838" height="665" alt="image" src="https://github.com/user-attachments/assets/b934b18d-e1fd-4293-a3a8-4e423478890c" />

# 📊 Experimental Results
Quantitative Results of Different Methods :
<img width="869" height="343" alt="image" src="https://github.com/user-attachments/assets/9e51bbaa-66ce-47eb-b162-3f4ac20eb938" />
Visualization Results of Different Methods :
<img width="884" height="478" alt="image" src="https://github.com/user-attachments/assets/56b4c933-27ff-4353-a3a3-1903d4234d35" />
Coming in future updates
