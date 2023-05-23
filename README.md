# Unoffical code for HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening (CVPR'22)

## What we have done for imporvment?
1) extend the hyperspectral datasets(over 100 bands) to multispctral datasets(only have 4 bands)
2) add Resnet50 pretrained model as backbone for spectral features extraction
3) remove origin cross-attention block as attention block in Deformable Detr
4) We will release datasets, checkpoints,comparision result and so on, beside, we want to make a google colab for easy implement and understand, please wait for our update!!!


For more information, please see
- **Paper**: [`CVPR-2022-Open-Access`](https://openaccess.thecvf.com/content/CVPR2022/html/Bandara_HyperTransformer_A_Textural_and_Spectral_Feature_Fusion_Transformer_for_Pansharpening_CVPR_2022_paper.html) or [`arxiv`](https://arxiv.org/abs/2203.02503).

## Flow chart



# Setting up a virtual conda environment
Setup a virtual conda environment using the provided ``environment.yml`` file or ``requirements.txt``.
```
conda env create --name HyperTransformer --file environment.yaml
conda activate HyperTransformer
```
or
```
conda create --name HyperTransformer --file requirements.txt
conda activate HyperTransformer
```

# Download datasets

We use three publically available HSI datasets for experiments, namely

1) **Pavia Center scene** [`Download the .mat file here`](https://www.dropbox.com/s/znykgpipttircdr/Pavia_centre.mat?dl=0), and save it in "./datasets/pavia_centre/Pavia_centre.mat".
2) **Botswana dataset**[`Download the .mat file here`](https://www.dropbox.com/s/w5bie03gaeuu6t9/Botswana.mat?dl=0), and save it in "./datasets/botswana4/Botswana.mat".
3) **Chikusei dataset** [`Download the .mat file here`](https://naotoyokoya.com/Download.html), and save it in "./datasets/chikusei/chikusei.mat".

# Processing the datasets to generate LR-HSI, PAN, and Reference-HR-HSI using Wald's protocol
 We use Wald's protocol to generate LR-HSI and PAN image. To generate those cubic patches,
  1) Run `process_pavia.m` in `./datasets/pavia_centre/` to generate cubic patches. 
  2) Run `process_botswana.m` in `./datasets/botswana4/` to generate cubic patches.
  3) Run `process_chikusei.m` in `./datasets/chikusei/` to generate cubic patches.
 
We also use multi-spectral pansharpening dataset acquired by WorldView4(WV4),QuickBird(QB),WorldView2(WV2) satalite
The number of bands in WV4 and QB is 4, and WV2 is 8
1) **WV4 WV2 QB** [`Download the .h5 file here`](https://pan.baidu.com/s/1ZdIHrQB93-ASi2zR5kJz0w?pwd=dffs)


# Training the Backbone of HyperTrasnformer
Use the following codes to train HyperTransformer on the three datasets.
 1) training on Pavia Center Dataset: 
    
    Change "train_dataset" to "pavia_dataset" in config_HSIT_PRE.json. 
    
    Then use following commad to train on Pavia Center dataset.
    `python train.py --config configs/config_HSIT_PRE.json`.
    
 4) training on Botswana Dataset:
     Change "train_dataset" to "botswana4_dataset" in config_HSIT_PRE.json. 
     
     Then use following commad to train on Pavia Center dataset. 
     `python train.py --config configs/config_HSIT_PRE.json`.
     
 6) training on Chikusei Dataset: 
     
     Change "train_dataset" to "chikusei_dataset" in config_HSIT_PRE.json. 
     
     Then use following commad to train on Pavia Center dataset. 
     `python train.py --config configs/config_HSIT_PRE.json`.
     
 
# Trained models and pansharpened results on test-set
You can download trained models and final prediction outputs through the follwing links for each dataset.
  1) Pavia Center: [`Download here`](https://www.dropbox.com/sh/9zg0wrbq6fzx1wa/AACH3mnRlqkVFmo6BF4wcDdaa?dl=0)
  2) Botswana: [`Download here`](https://www.dropbox.com/sh/e7og46hkn3wuaxr/AACrFOpOSFF2u0hG1CzNYVRxa?dl=0)
  3) Chikusei: [`Download here`](https://www.dropbox.com/sh/l6gaf723cb6asq4/AABPBUleyZ7aFX8POh_d5jC9a?dl=0)
  4) WV4: [`Download here`](https://pan.baidu.com/s/1rtO6g39PWOeK7kD0cqkrIw?pwd=qf3q)




