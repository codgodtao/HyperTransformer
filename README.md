# HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening (CVPR'22)

[`Wele Gedara Chaminda Bandara`](https://www.wgcban.com/), and [`Vishal M. Patel`](https://engineering.jhu.edu/vpatel36/sciencex_teams/vishalpatel/)


For more information, please see our 
- **Paper**: [`CVPR-2022-Open-Access`](https://openaccess.thecvf.com/content/CVPR2022/html/Bandara_HyperTransformer_A_Textural_and_Spectral_Feature_Fusion_Transformer_for_Pansharpening_CVPR_2022_paper.html) or [`arxiv`](https://arxiv.org/abs/2203.02503).
- **Poster**: [`view here`](https://www.dropbox.com/s/6gw64uo2j327yp1/poster.pdf?dl=0)
- **Video Presentation**: [`view here`](https://www.dropbox.com/s/twf90mbzjmev7yl/CVPR-HyperTransformer.mp4?dl=0)
- **Presentation Slides**: [`download here`](https://www.dropbox.com/s/odki2ikymkoh85r/Presentation.pptx?dl=0)

## Summary
<p align="center">
  <img src="/imgs/poster.jpg" />
</p>


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
 
# Training HyperTransformer 
We use two stage procedure to train our HyperTransformer. 

We first train the backbone of HyperTrasnformer and then fine-tune the MHFA modules. This way we get better results and faster convergence instead of training whole network at once.

## Training the Backbone of HyperTrasnformer
Use the following codes to pre-train HyperTransformer on the three datasets.
 1) Pre-training on Pavia Center Dataset: 
    
    Change "train_dataset" to "pavia_dataset" in config_HSIT_PRE.json. 
    
    Then use following commad to pre-train on Pavia Center dataset.
    `python train.py --config configs/config_HSIT_PRE.json`.
    
 4) Pre-training on Botswana Dataset:
     Change "train_dataset" to "botswana4_dataset" in config_HSIT_PRE.json. 
     
     Then use following commad to pre-train on Pavia Center dataset. 
     `python train.py --config configs/config_HSIT_PRE.json`.
     
 6) Pre-training on Chikusei Dataset: 
     
     Change "train_dataset" to "chikusei_dataset" in config_HSIT_PRE.json. 
     
     Then use following commad to pre-train on Pavia Center dataset. 
     `python train.py --config configs/config_HSIT_PRE.json`.
     

## Fine-tuning the MHFA modules in HyperTrasnformer
Next, we fine-tune the MHFA modules in HyperTransformer starting from pre-trained backbone from the previous step.
 1) Fine-tuning MHFA on Pavia Center Dataset: 

    Change "train_dataset" to "pavia_dataset" in config_HSIT.json. 
    
    Then use the following commad to train HyperTransformer on Pavia Center dataset. 
    
    Please specify path to best model obtained from previous step using --resume.
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/pavia_dataset/N_modules\(4\)/best_model.pth`.
   
 3) Fine-tuning on Botswana Dataset: 

    Change "train_dataset" to "botswana4_dataset" in config_HSIT.json. 
    
    Then use following commad to pre-train on Pavia Center dataset. 
    
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/botswana4/N_modules\(4\)/best_model.pth`.

 5) Fine-tuning on Chikusei Dataset: 

    Change "train_dataset" to "chikusei_dataset" in config_HSIT.json.
    
    Then use following commad to pre-train on Pavia Center dataset. 
    
    `python train.py --config configs/config_HSIT.json --resume ./Experiments/HSIT_PRE/chikusei_dataset/N_modules\(4\)/best_model.pth`.
    
# Trained models and pansharpened results on test-set
You can download trained models and final prediction outputs through the follwing links for each dataset.
  1) Pavia Center: [`Download here`](https://www.dropbox.com/sh/9zg0wrbq6fzx1wa/AACH3mnRlqkVFmo6BF4wcDdaa?dl=0)
  2) Botswana: [`Download here`](https://www.dropbox.com/sh/e7og46hkn3wuaxr/AACrFOpOSFF2u0hG1CzNYVRxa?dl=0)
  3) Chikusei: [`Download here`](https://www.dropbox.com/sh/l6gaf723cb6asq4/AABPBUleyZ7aFX8POh_d5jC9a?dl=0)

# Citation
If you find our work useful, please consider citing our paper.
```
@InProceedings{Bandara_2022_CVPR,
    author    = {Bandara, Wele Gedara Chaminda and Patel, Vishal M.},
    title     = {HyperTransformer: A Textural and Spectral Feature Fusion Transformer for Pansharpening},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {1767-1777}
}
```



