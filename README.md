# mri-to-ct

## Introduction

This repo contains the code from paper XXX. This guide is here to help you run the code on your device, and train a MRI-to-CT translation model. If you intend to use this program for your research, use the citation below:

@article{lundervold2019overview,
  title={An overview of deep learning in medical imaging focusing on MRI},
  author={Lundervold, Alexander Selvikv{\aa}g and Lundervold, Arvid},
  journal={Zeitschrift f{\"u}r Medizinische Physik},
  volume={29},
  number={2},
  pages={102--127},
  year={2019},
  publisher={Elsevier}
}

## Prerequesites

Make sure your Python environment verifies:
    
    Python >= 3.9
    Tensorflow >= 2.8.0
    CUDA >= 11.2
    cudnn >= XX

## Download

    git clone https://github.com/MatDagommer/mri-to-ct.git

## Data

In the folder ``` /data/raw_data ```, create a folder for each subject containing 4 NIfTI files:

* ``` CTlnT1_resliced.nii ``` (CT)
* ``` pCTlnT1_resliced.nii ``` (pseudo-CT)
* ``` skull_SAMSEG_resliced.nii ``` (Skull Mask)
* ``` T1_resliced.nii ``` (T1-MRI)

## References


