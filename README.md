# SCAM! Transferring humans between images with Semantic Cross Attention Modulation

Official PyTorch implementation of "SCAM! Transferring humans between images with Semantic Cross Attention Modulation", accepted at ECCV 2022 .

Arxiv link | Website link | Paper Link
<span color='red'>#TODO add links </span>


<p align="center">
<img src="medias/teaser.png" height="300px"/>
</p>



## Bibtex

If you happen to find this code or method useful in your research please cite this paper with

<span color='red'>#TODO add bibtex</span>

## Installation guide
To install this repo follow the following step:

1. Install Anaconda or MiniConda
2. Run `conda create -n scam python=3.9.12` 
3. Activate scam:  `conda activate scam`
4. Install pytorch 1.12 and torchvision 0.13 that match your device:
    - For cpu:
    ```
    conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cpuonly -c pytorch
    ```
    - For cuda 11.6:
    ```
    conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.6 -c pytorch -c conda-forge
    ```
5. Install dependencies:
```
conda env update --file environment.yml
```

## Prepare the data

### Download CelebAHQ

Download link: <span color='red'>#TODO add CelebAHQ and iDesigner postprocessed downloads</span>

Unzip inside the datasets folder

## Add a new dataset
If you want to add a new dataset you need to follow the same directory structure as the other datasets:

2 folders: train and test. 
Each folders contains 4 sub-folders: images, labels, vis and stats.
- images: contains the rgb images. can be jpg or png.
- labels: the segmentation labels. needs to be png where each pixels takes value between 0 and num_labels.
- vis: rgb visualization of the labels
- stats: Inception statistics of the train dataset. To do so, initialize a dataloader with the train dataloader and run metrics/fid/compute_fid_for_dataset.
## Run SCAM
Our code needs to be logged in wandb. Make sure to be logged in to wandb before starting. 
To run our code you just need to do:

```python train.py```

By default the code will be runned on cpu. If you want to run it on gpu you can do:

```python train.py compnode=single-gpu```

Other compute configs exists in `configs/compnode`. If none suit your needs you can easily create one thanks to the modularity of hydra.

In this paper, the main compute config we use is the 4 gpus (NVIDIA V100 32g with a total batch size of 32, 8 per GPUs) config called `1-node-cluster`.

By default, our method scam will be runned. If you want to try one of the baselines we compare to, you can run

```python train.py model=sean```

We include implementations for SEAN, SPADE, INADE, CLADE and GroupDNET.

To swap datasets, you can do:

```python train.py dataset=CelebAHQ```

we support 3 datasets, `CelebAHQ`, `idesigner` and `ADE20K`.

If you want to run different experiments, make sure to change `experiment_name_comp` or `experiment_name` when running. Otherwise or code will restart the training from the checkpointed weights for the experiment with the same name.