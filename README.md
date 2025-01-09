# Upper-body free-breathing Magnetic Resonance Fingerprinting applied to the quantification of water T1 and fat fraction

Motion-corrected (MoCo) MRF T1-FF is a framework for free-breathing quantitative MRI that estimates the respiration motion field using an optimized preliminary motion scan and uses it to correct the Magnetic Resonance Fingerprinting acquisition data before dictionary search for reconstructing motion-corrected fat fraction and water T1 parametric maps of the upper-body region.

![Alt text](image/Fig1_MoCoFramework_v2_formatted.jpeg?raw=true "Figure 1: MoCo MRF T1-FF framework")


## Requirements

After cloning the repository, one should create a conda environment using the requirements.txt.
```
conda env create --name <env> --file environment.yml
conda activate <env>
```

Weights & Biases is required to monitor the neural network training - create an account there: https://wandb.ai/site/ to get the API key and enter it when prompted during training.

## Data

Data from the motion scan can be pulled using:

```
python pull_example_data.py --url 'https://github.com/Stanrenko/mrf_moco_media_v2/releases/download/v1.0.0/inputs_motion.zip' --data_dir './data'
```

Data from the MRF scan can be pulled using:

```
python pull_example_data.py --url 'https://github.com/Stanrenko/mrf_moco_media_v2/releases/download/v1.0.0/inputs_mrf.zip' --data_dir './data'
```

MRF dictionaries can be pulled using:

```
python pull_example_data.py --url 'https://github.com/Stanrenko/mrf_moco_media_v2/releases/download/v1.0.0/dictionaries.zip'
```

## Snippet code for MoCo MRF T1-FF post-processing

4 blocks are illustrated here:

### 1. Displacement extraction from navigator images and binning (steps 1 and 3 in Figure 1)

This code extracts respiratory displacement from raw navigators motion scan k-space data.

On the motion scan, no seasonality adjustment is used. The displacements calculated on the motion scan are used to calculate the edges of the bins.
The channel with the best contrast for displacment extraction should be provided.

```
python extract_displacement.py --filenamenav data/motion_scan_raw_nav.npy --ch 17
```

On the Magnetic Resonance Fingerprinting (mrf) scan, seasonality adjustment is used to get greater homogeneity on the navigator  images. The bins calculated on the motion scan navigators are passed through filenamebins.

```
python extract_displacement.py --filenamenav data/mrf_scan_raw_nav.npy --ch 17 --filenamebins data/motion_scan_raw_bins.npy --seasonaladj True --hardinterp True
```

### 2. Estimating deformation field with Voxelmorph ([[1]](#1)) from motion scan volumes for all bins (step 2 in Figure 1)

The code outputs the gif showing the movie of registered volumes vs the movie of unregistered volumes, and saves the deformation map in motion_scan_volumes_allbins_deformation_map.npy

```
python estimate_deformation.py --filenamevol data/motion_scan_volumes_allbins.npy --fileconfig config/config_train_voxelmorph.json --nepochs 1 --keptbins "0,1,2,3"
```

### 3. Building motion-corrected mrf singular volumes and mask from mrf singular volumes for all bins and estimated deformation fields (step 4 in Figure 1)

The mrf singular volumes are obtained by projecting the acquired signals on the mrf temporal basis dictionary_phi_L0_6.npy. The weights are the gating weights. 

```
python build_moco_mrf_volumes.py --filenamevol data/mrf_scan_singular_volumes_allbins.npy --fileb1 data/coil_sensi.npy --fileweights data/mrf_scan_raw_weights.npy --filedef data/motion_scan_volumes_allbins_deformation_map.npy
```


### 4. Building parametric maps from motion-corrected mrf singular volumes (step 5 in Figure 1)

The pattern matching used is the bicomponent dictionary matching with clustering ([[2]](#2)), which uses first a coarse dictionary (dictionary_light.dict) to cluster the signals efficiently before matching on the full dictionary (dictionary.dict).

The dictionaries are stored in .dict files, the keys being the parameter values (water T1, fat T1, B1, df). FF is matched in a first step using a closed form formula as described in [[2]](#2).

```
python build_maps.py --filenamevol data/mrf_scan_singular_volumes_allbins_registered_ref0.npy --fileseqparams data/mrf_scan_seqParams.pkl --dictfile dictionary.dict --dictfilelight dictionary_light.dict --fileconfigmrf config/config_build_maps.json 
```

## References
<a id="1">[1]</a> 
Balakrishnan G, Zhao A, Sabuncu M. R., Guttag J and Dalca A. V., VoxelMorph: A Learning Framework for Deformable Medical Image Registration. IEEE Transactions on Medical Imaging, Aug. 2019. doi: 10.1109/TMI.2019.2897538

<a id="2">[2]</a> 
Slioussarenko C, Baudin P-Y, Reyngoudt H, Marty B. Bi-component dictionary matching for MR fingerprinting for efficient quantification of fat fraction and water T1 in skeletal muscle. Magn Reson Med. 2024; 91: 1179-1189. doi: 10.1002/mrm.29901