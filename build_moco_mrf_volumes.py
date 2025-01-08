
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import finufft
import pywt
import cv2
from tqdm import tqdm
from copy import copy
import os


from utils import apply_deformation_to_complex_volume, change_deformation_map_ref,calculate_inverse_deformation_map,Radial,undersampling_operator_singular_new

# from utils_mha import write


import argparse


def build_volumes_iterative_allbins_registered(filename_volume,filename_b1,filename_weights,file_deformation,niter=2,gating_only=True,dens_adj=True,index_ref=0,mu=0.1,lambda_wav=1e-5,axis=None):
    '''
        FISTA for minimizing the least-square reconstruction including motion field and wavelet regularization : argmin_U sum_i||WA(M_i)U - y_i||_2^2 + lambd ||WU||_1
        where A(M_i) = FSM with F fourier operator, S coil sensi and M_i deformation map for bin i

        inputs:
            filename_volume: .npy file containing respiratory volumes for all bins (size respiratory bins count x nb slices x npoint x npoint if motion scan data or respiratory bins count x nb singular volumes x nb slices x npoint x npoint if mrf scan data)
            filename_b1: .npy file containing coil sensitivities (nb channels x volume size)
            filename_weights: gating weights file .npy (1 x nb readouts per repetition x nb repetitions x 1)
            file_deformation: .npy containing the deformation field from all respiratory bins to bin 0, size (2 x resp bins count x nb slices x npoint x npoint)
            niter: number of FISTA iterations
            gating_only: use weights only for gating
            dens_adj: bool specifying whether to use radial density compensation for iterative reconstruction
            index_ref: int to change the reference respiratory bin
            mu: gradient descent step in FISTA
            lamda_wav: wavelet regularization penalty
            axis: int specifying whether the deformation map was trained on an axis other than 0

        outputs:
            filename_target : .npy file name containing the registered volumes (size (nb slices x npoint x npoint) if motion scan data or (nb singular volumes x nb slices x npoint x npoint) if mrf scan data)


    '''


    filename_target=filename_volume.split(".npy")[0] + "_registered_ref{}.npy".format(index_ref)

    interp=cv2.INTER_LINEAR
    print("Loading Volumes")
    volumes=np.load(filename_volume)

    volumes=volumes.astype("complex64")

    if volumes.ndim==4:
        #To fit the input expected by the undersampling function with L0=1 in our case
        volumes=np.expand_dims(volumes,axis=1)
        shift=0
    else:
        shift=1

    b1_all_slices_2Dplus1_pca=np.load(filename_b1)
    all_weights=np.load(filename_weights)

    if gating_only:
        print("Using weights only for gating")
        all_weights=(all_weights>0)*1

    
    print("Volumes shape {}".format(volumes.shape))
    print("Weights shape {}".format(all_weights.shape))

    nb_allspokes=all_weights.shape[2]
    npoint_image=volumes.shape[-1]
    npoint=2*npoint_image
    nbins=volumes.shape[0]

    incoherent=False
    mode="old"

    radial_traj = Radial(total_nspokes=nb_allspokes, npoint=npoint)

    deformation_map=np.load(file_deformation)
    deformation_map=change_deformation_map_ref(deformation_map,index_ref,axis)

    print("Calculating inverse deformation map")

    
    inv_deformation_map = np.zeros_like(deformation_map)
    for gr in tqdm(range(nbins)):
        inv_deformation_map[:, gr] = calculate_inverse_deformation_map(deformation_map[:, gr],axis)
        
    volumes_registered=np.zeros(volumes.shape[1:],dtype=volumes.dtype)

    print("Registering initial volumes")
    for gr in range(nbins):
        volumes_registered+=apply_deformation_to_complex_volume(volumes[gr].squeeze(),deformation_map[:,gr],interp=interp,axis=axis)
                                                                                    


    volumes0=copy(volumes_registered)

    volumes_registered=mu*volumes0

    print("volumes_registered.shape {}".format(volumes_registered.shape))

    
    wav_level = None
    wav_type="db4"

    lambd = lambda_wav

    print("Wavelet regularization penalty {}".format(lambd))

    if volumes_registered.ndim==3:
        volumes_registered=np.expand_dims(volumes_registered,axis=0)

    coefs = pywt.wavedecn(volumes_registered, wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
    u, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
    u0 = u
    y = u
    t = 1

    u = pywt.threshold(y, lambd * mu)

    print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

    t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
    y = u 
    t = t_next

    for i in range(niter):
        u_prev = u
        x = pywt.array_to_coeffs(y, slices)
        x = pywt.waverecn(x, wav_type, mode="periodization",axes=(1,2,3))

        
        for gr in tqdm(range(nbins)):
                
            volumesi=apply_deformation_to_complex_volume(x,inv_deformation_map[:,gr],interp=interp,axis=axis)
            if volumesi.ndim==3:
                volumesi=np.expand_dims(volumesi,axis=0)
 
            volumesi = undersampling_operator_singular_new(volumesi, radial_traj,
                                                               b1_all_slices_2Dplus1_pca, weights=all_weights[gr],
                                                               density_adj=dens_adj)
                   
            volumesi=apply_deformation_to_complex_volume(volumesi.squeeze(),deformation_map[:,gr],interp=interp,axis=axis)

            if volumesi.ndim==3:
                volumesi=np.expand_dims(volumesi,axis=0)

            if gr==0:
                final_volumesi=volumesi
            else:
                final_volumesi+=volumesi
        coefs = pywt.wavedecn(final_volumesi, wav_type, level=wav_level, mode="periodization",axes=(1,2,3))
        grad_y, slices = pywt.coeffs_to_array(coefs,axes=(1,2,3))
        grad = grad_y - u0
        y = y - mu * grad

        u = pywt.threshold(y, lambd * mu)

        print("Non zero percentage: {} ".format(np.count_nonzero(u)/np.prod(u.shape)))

        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t ** 2))
        y = u + (t - 1) / t_next * (u - u_prev)
        t = t_next

    volumes_registered = pywt.waverecn(pywt.array_to_coeffs(u, slices), wav_type, mode="periodization",axes=(1,2,3)).squeeze()
    np.save(filename_target,volumes_registered)

    return filename_target



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenamevol',type=str)
    parser.add_argument('--fileb1', type=str)
    parser.add_argument('--filedef', type=str)
    parser.add_argument('--fileweights', type=str)
    parser.add_argument('--niter', nargs='?', const=2, type=int, default=2)

    args = parser.parse_args()

    filename_volume=args.filenamevol
    filename_b1=args.fileb1
    filename_weights=args.fileweights
    file_deformation=args.filedef
    niter=args.niter

    filename_singular_volumes_registered=build_volumes_iterative_allbins_registered(filename_volume,filename_b1,filename_weights,file_deformation,niter)
