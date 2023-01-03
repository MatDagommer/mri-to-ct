# mri-to-ct

## Introduction

This repo contains the code from paper XXX. This guide is here to help you run the code on your device, and train a MRI-to-CT translation model. 

## Prerequesites

Make sure your Python environment verifies:
    
    Python >= 3.9
    Tensorflow >= 2.8.0

## Download

    git clone https://github.com/MatDagommer/mri-to-ct.git

## Data

In '''data/raw_data/''', create a folder for each subject containing 4 NIfTI files:

* CT (CTlnT1_resliced.nii)
* pseudo-CT (pCTlnT1_resliced.nii)
* Skull Mask (skull_SAMSEG_resliced.nii)
* T1-MRI (T1_resliced.nii)



