# S<sup>4</sup>DL: Shift-sensitive Spatial-Spectral Disentangling Learning for Hyperspectral Image Unsupervised Domain Adaptation

## Abstract
![PDF Image](/fig.png)

Unsupervised domain adaptation techniques, extensively studied in hyperspectral image (HSI) classification, aim to use labeled source domain data and unlabeled target domain data to learn domain invariant features for cross-scene classification. Compared to natural images, numerous spectral bands of HSIs provide abundant semantic information, but they also increase the domain shift significantly. In most existing methods, both explicit alignment and implicit alignment simply align feature distribution, ignoring domain information in the spectrum. We noted that when the spectral channel between source and target domains is distinguished obviously, the transfer performance of these methods tends to deteriorate.Additionally, their performance fluctuates greatly owing to the varying domain shifts across various datasets. To address these problems, a novel shift-sensitive spatial-spectral disentangling learning (S<sup>4</sup>DL) approach is proposed. In S<sup>4</sup>DL, gradient-guided spatial-spectral decomposition is designed to separate domain-specific and domain-invariant representations by generating tailored masks under the guidance of the gradient from domain classification. A shift-sensitive adaptive monitor is defined to adjust the intensity of disentangling according to the magnitude of domain shift. Furthermore, a reversible neural network is constructed to retain domain information that lies in not only semantic but also shallow-level detailed information. Extensive experimental results on several cross-scene HSI datasets consistently verified that S<sup>4</sup>DL is better than the state-of-the-art UDA methods. Our source code will be available.

## <a name="usage"></a> Usage

### <a name="usage-train"></a> Dataset
The dataset directory should look like this:
```bash
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Shanghai_Hangzhou
│   ├── DataCube_ShanghaiHangzhou.mat
└──  HyRANK
    ├── Dioni.mat
    └── Dioni_gt.mat
    ├── Loukia.mat
    └── Loukia_gt.mat
```


### <a name="usage-train"></a> Run

1. Download the dataset.
2. Change the dataset path (DATASET/PATH) in the yaml file to the download path.
3. Run the following command.

 ```shell
python train/s4dl/train.py configs/hyrank/s4dl/config.yaml \
            --path ./runs/hyrank/s4dl \
            --nodes 1 \
            --gpus 1 \
            --rank-node 0 \
            --backend gloo \
            --master-ip localhost \
            --master-port 9001 \
            --seed 1
```



## <a name="license"></a> License

This project is released under the MIT(LICENSE) license.