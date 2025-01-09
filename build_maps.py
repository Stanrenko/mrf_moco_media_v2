
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
import pickle
import dask.array as da
from tqdm import tqdm
from copy import copy
from PIL import Image
import os
import SimpleITK as sitk

from dictmodel import Dictionary

from utils import build_mask_from_volume,makevol
from utils_mrf import read_mrf_dict,SimpleDictSearch
# from utils_mha import write


import argparse


def build_mask_from_singular_volume(filename_volume,l=0,threshold=0.015,it=2):
    '''
        Builds mask from singular volumes
        inputs:
            filename_volume: .npy file name containing the singular volumes (nb singular volumes x nb slices x npoint x npoint)
            l: singular volume used for mask calculation 
            threshold: histogram threshold to define retained pixels for the mask
            it: binary closing iterations

        outputs:
            filename_mask: .npy file containing the mask (nb slices x npoint x npoint)
    '''

    filename_mask="".join(filename_volume.split(".npy"))+"_l{}_mask.npy".format(l)
    volumes=np.load(filename_volume)

    volume=volumes[l]
    mask=build_mask_from_volume(volume,threshold,it)
    
    np.save(filename_mask,mask)

    gif = []

    for i in range(mask.shape[0]):
        img = Image.fromarray(np.uint8(mask[i] / np.max(mask[i]) * 255), 'L')
        img = img.convert("P")
        gif.append(img)

    filename_gif = str.split(filename_mask, ".npy")[0] + ".gif"
    gif[0].save(filename_gif, save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)

    return filename_mask

def build_maps(filename_volume,filename_mask,filename_seqParams,dictfile,dictfile_light,optimizer_config,slices=None):
    '''
    Builds parametric maps from mrf singular volumes using bicomponent dictionary matching
    inputs:
        filename_volume: .npy containing mrf volumes (nb singular volumes x volume size)
        filename_mask :  .npy containing mask (volume size)
        filename_seqParams: .pkl file containing mrf sequence parameters
        dictfile: .dict containing dictionary file
        dictfile_light: .dict containing coarse dictionary file for preliminary clustering
        optimizer_config: .json bicomponent dictionary matching config file
        slices: string containing slices to reconstruct (e.g. "45,46,47")
    
    outputs:
        .pkl containing maps on mask
        .mha for each parameter containing the parametric maps of size volume size for each parameter (fat fraction ff, water T1 wT1, df inhomogeneity, B1 inhomogeneity attB1)

    '''

    file_map = "".join(filename_volume.split(".npy")) + "_MRF_map.pkl"
    volumes_all = np.load(filename_volume)

    mask=np.load(filename_mask)

    print("Volumes shape : {}".format(volumes_all.shape))

    ntimesteps=volumes_all.shape[0]
    print("There are {} volumes to match for the fingerprinting".format(ntimesteps))

    file = open(filename_seqParams, "rb")
    dico_seqParams = pickle.load(file)
    file.close()


    x_FOV = dico_seqParams["x_FOV"]
    y_FOV = dico_seqParams["y_FOV"]
    z_FOV = dico_seqParams["z_FOV"]

    npoint=2*volumes_all.shape[2]
    nb_slices=volumes_all.shape[1]
    dx = x_FOV / (npoint / 2)
    dy = y_FOV / (npoint / 2)
    dz = z_FOV / nb_slices

    if slices is not None:
        sl = np.array(slices.split(",")).astype(int)
        if not(len(sl)==0):
            mask_slice = np.zeros(mask.shape, dtype=mask.dtype)
            mask_slice[sl] = 1
            mask *= mask_slice
            sl=[str(s) for s in sl]
            file_map = "".join(filename_volume.split(".npy")) + "_sl{}_MRF_map.pkl".format("_".join(sl))

    print("Output map file name : {}".format(file_map))


    threshold_pca=optimizer_config["pca"]
    split=optimizer_config["split"]
    useGPU = optimizer_config["useGPU"]
    clustering=optimizer_config["clustering"]
    return_matched_signals=optimizer_config["return_matched_signals"]
    return_cost = optimizer_config["return_cost"]

    L0=ntimesteps
    filename_phi=str.split(dictfile,".dict") [0]+"_phi_L0_{}.npy".format(L0)

    if filename_phi not in os.listdir():
            #mrfdict = dictsearch.Dictionary()

        print("Generating mrf temporal basis : {}".format(filename_phi))
        keys,values=read_mrf_dict(dictfile,np.arange(0.,1.01,0.1))

        
        u,s,vh = da.linalg.svd(da.asarray(values))

        vh=np.array(vh)
        
        phi=vh[:L0]
        np.save(filename_phi,phi.astype("complex64"))
        del keys
        del values
        del u
        del s
        del vh
    else:
        print("Loading mrf temporal basis : {}".format(filename_phi))
        phi=np.load(filename_phi)

    mrfdict = Dictionary()
    mrfdict.load(dictfile, force=True)
    keys = mrfdict.keys
    array_water = mrfdict.values[:, :, 0]
    array_fat = mrfdict.values[:, :, 1]
    array_water_projected=array_water@phi.T.conj()
    array_fat_projected=array_fat@phi.T.conj()

    mrfdict_light = Dictionary()
    mrfdict_light.load(dictfile_light, force=True)
    keys_light = mrfdict_light.keys
    array_water = mrfdict_light.values[:, :, 0]
    array_fat = mrfdict_light.values[:, :, 1]
    array_water_light_projected=array_water@phi.T.conj()
    array_fat_light_projected=array_fat@phi.T.conj()


    optimizer = SimpleDictSearch(mask=mask,split=split,pca=True,threshold_pca=threshold_pca,useGPU_dictsearch=useGPU,threshold_ff=0.9,dictfile_light=(array_water_light_projected,array_fat_light_projected,keys_light),return_cost=return_cost,clustering=clustering,return_matched_signals=return_matched_signals)
    all_maps=optimizer.search_patterns_test_multi_2_steps_dico((array_water_projected,array_fat_projected,keys),volumes_all,retained_timesteps=None)




    curr_file=file_map
    file = open(curr_file, "wb")
    pickle.dump(all_maps,file)
    file.close()

    for iter in list(all_maps.keys()):

        map_rebuilt=all_maps[iter][0]
        mask=all_maps[iter][1]

        map_rebuilt["wT1"][map_rebuilt["ff"] > 0.7] = 0.0

        keys_simu = list(map_rebuilt.keys())
        values_simu = [makevol(map_rebuilt[k], mask > 0) for k in keys_simu]
        map_for_sim = dict(zip(keys_simu, values_simu))

        for key in ["ff","wT1","df","attB1"]:
            file_mha = "/".join(["/".join(str.split(curr_file,"/")[:-1]),"_".join(str.split(str.split(curr_file,"/")[-1],".")[:-1])]) + "_it{}_{}.mha".format(iter,key)
            curr_map=map_for_sim[key]
            curr_map=np.flip(np.moveaxis(curr_map,0,2),axis=(0,1,2))
            curr_volume=sitk.GetImageFromArray(curr_map)
            curr_volume.SetSpacing((dz,dx,dy))
            sitk.WriteImage(curr_volume,file_mha)

    return





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenamevol', type=str)
    parser.add_argument('--fileseqparams', type=str)
    parser.add_argument('--dictfile', type=str)
    parser.add_argument('--dictfilelight', type=str)
    parser.add_argument('--fileconfigmrf', type=str)
    parser.add_argument('--slices', nargs='?', const=None, type=str, default=None)

    args = parser.parse_args()

    filename_singular_volumes_registered=args.filenamevol
    filename_seqParams=args.fileseqparams
    dictfile=args.dictfile
    dictfile_light=args.dictfilelight
    slices=args.slices

    with open(args.fileconfigmrf, 'r') as file:
        optimizer_config = json.load(file)
    

    filename_mask=build_mask_from_singular_volume(filename_singular_volumes_registered)
    
    build_maps(filename_singular_volumes_registered,filename_mask,filename_seqParams,dictfile,dictfile_light,optimizer_config,slices=slices)