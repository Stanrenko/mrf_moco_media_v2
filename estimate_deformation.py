
import json
import numpy as np
from copy import copy
from skimage.transform import resize
import matplotlib 
import matplotlib.pyplot as plt
from PIL import Image

import voxelmorph as vxm
# import neurite as ne
import wandb
from wandb.keras import WandbMetricsLogger,WandbModelCheckpoint
from keras import backend
import tensorflow as tf
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'
print(tf.config.experimental.list_physical_devices("GPU"))

import argparse

from utils import format_input_voxelmorph,vxm_data_generator,scheduler,register_motionbin

def train_voxelmorph(filename_volumes,config_train,nepochs,init_weights=None,resolution=None,kept_bins=None,axis=None,us=None,excluded=5):
    '''
        Trains a model to learn deformation field of adjacent bins based on input volumes for all respiratory bins

        inputs:
            filename_volumes: .npy file containing respiratory volumes for all bins (size respiratory bins count x nb slices x npoint x npoint)
            config_train: .json config file for voxelmorph neural network
            nepochs: number of epochs overriding the default number of epochs in the config file
            init_weights: initialization file for the neural network weight if one wants to use a pretrained network
            resolution: int to downsample the input volumes from npoint x npoint to resolution x resolution
            kept_bins: str - respiratory bins to keep for deformation field estimation (e.g. "0,1,2,3")
            axis: int - if one wants to change the axis along which the deformation is estimated
            us: int - to keep only one in every us slice for training
            excluded: int - number of excluded slices on each extremity of the field of view

        outputs:
            file_model name
            Saves two files: 
            file_model: .h5 containing the trained weights of the network
            file_loss: .jpg showing the loss across epochs
    '''
    all_volumes = np.abs(np.load(filename_volumes))
    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")


    if kept_bins is not None:
        kept_bins_list=np.array(str.split(kept_bins,",")).astype(int)
        print(kept_bins_list)
        all_volumes=all_volumes[kept_bins_list]


    nb_gr,nb_slices,npoint,npoint=all_volumes.shape


    if axis is not None:
        all_volumes=np.moveaxis(all_volumes,axis+1,1)
        all_volumes=all_volumes[:,::int(npoint/nb_slices)]


    if us is not None:
        all_volumes=all_volumes[:,::us]

    print(all_volumes.shape)

    if nepochs is not None:
        config_train["nb_epochs"]=nepochs


    if resolution is not None:
        all_volumes=resize(all_volumes,(nb_gr,nb_slices,resolution,resolution))

    if resolution is None:
        file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights.h5"
        file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint.h5"
    else:
        file_model=filename_volumes.split(".npy")[0]+"_vxm_model_weights_res{}{}.h5".format(resolution)
        file_checkpoint="/".join(filename_volumes.split("/")[:-1])+"/model_checkpoint_res{}.h5".format(resolution)
    print(file_checkpoint)

    run=wandb.init(
        project="project_test",
        config=config_train
    )

    loss = config_train["loss"]
    decay=config_train["lr_decay"]

    #Finding the power of 2 "closest" and longer than  x dimension
    n = np.maximum(all_volumes.shape[-1],all_volumes.shape[-2])
    pad_1=2**(int(np.log2(n))+1)-n
    pad_2= 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2<0:
        pad=int(pad_1/2)
    else:
        pad = int(pad_2 / 2)

    # if n%2==0:
    #     pad=0
    pad_x=int((2*pad+n-all_volumes.shape[-2])/2)
    pad_y=int((2*pad+n-all_volumes.shape[-1])/2)

    pad_amount = ((0,0),(pad_x,pad_x), (pad_y,pad_y))
    print(pad_amount)
    nb_features=config_train["nb_features"]
    optimizer=config_train["optimizer"] #"Adam"
    lambda_param = config_train["lambda"] # 0.05
    nb_epochs = config_train["nb_epochs"] # 200
    batch_size=config_train["batch_size"] # 16
    lr=config_train["lr"]

    x_train_fixed,x_train_moving=format_input_voxelmorph(all_volumes,pad_amount,sl_down=excluded,sl_top=-excluded)
    

    # configure unet input shape (concatenation of moving and fixed images)
    inshape = x_train_fixed.shape[1:]

    vxm_model = vxm.networks.VxmDense(inshape, nb_features, int_steps=0)

    # loss classes

    if loss=="MSE":
        losses = [vxm.losses.MSE().loss, vxm.losses.Grad('l2').loss]
    elif loss == "NCC":
        losses = [vxm.losses.NCC().loss, vxm.losses.Grad('l2').loss]
    elif loss =="MI":
        losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    else:
        raise ValueError("Loss should be either MSE or Mutual Information (MI)")

    loss_weights = [1, lambda_param]

    print("Compiling Model")
    vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights)

    if lr is not None:
        backend.set_value(vxm_model.optimizer.learning_rate,lr)

    train_generator = vxm_data_generator(x_train_fixed,x_train_moving,batch_size=batch_size)


    nb_examples=x_train_fixed.shape[0]

    
    steps_per_epoch = int(nb_examples/batch_size)+1
    
    if "min_lr" in config_train:
        min_lr=config_train["min_lr"]

    curr_scheduler=lambda epoch,lr: scheduler(epoch,lr,decay,min_lr)
    Schedulecallback = tf.keras.callbacks.LearningRateScheduler(curr_scheduler)

    callback_checkpoint=WandbModelCheckpoint(filepath=file_checkpoint,save_best_only=True,save_weights_only=True,monitor="vxm_dense_transformer_loss")


    if init_weights is not None:
        vxm_model.load_weights(init_weights)

    hist = vxm_model.fit_generator(train_generator, epochs=nb_epochs, steps_per_epoch=steps_per_epoch, verbose=2,callbacks=[Schedulecallback,WandbMetricsLogger(),callback_checkpoint])

    vxm_model.save_weights(file_model)

    run.finish()

    plt.figure()
    plt.plot(hist.epoch, hist.history["loss"], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(file_model.split(".h5")[0]+"_loss.jpg")
    return file_model


def register_allbins_to_baseline(filename_volumes,file_model,config_train,niter=1,resolution=None,axis=None):
    '''
        Register all respiratory bins to bin 0 using trained voxelmorph neural network, and calculates the resulting deformation maps

        inputs:
            filename_volumes: .npy file containing respiratory volumes for all bins (size respiratory bins count x nb slices x npoint x npoint)
            file_model: trained voxelmorph network weights (.h5)
            config_train: .json config file for voxelmorph neural network
            niter: number of iteration for estimating deformation field
            resolution: int to specify trained resolution if model was trained on downsampled volumes
            axis: int - if the model was trained on a different axis than axis 0

        outputs:
            filename_registered_volumes file name
            Saves two files: 
            filename_deformation: .npy containing the deformation field from all respiratory bins to bin 0, size (2 x resp bins count x nb slices x npoint x npoint)
            filename_registered_volumes:  .npy file containing registered respiratory volumes for all bins (size respiratory bins count x nb slices x npoint x npoint)


    '''

    all_volumes = np.load(filename_volumes)

    if resolution is None:
        filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered.npy"
        filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map.npy"
    else:
        filename_registered_volumes=filename_volumes.split(".npy")[0]+"_registered_res{}.npy".format(resolution)
        filename_deformation = filename_volumes.split(".npy")[0] + "_deformation_map_res{}.npy".format(resolution)


    all_volumes = np.abs(all_volumes)
    
    deformation_map=None

    print("Volumes shape {}".format(all_volumes.shape))
    all_volumes=all_volumes.astype("float32")
    nb_gr,nb_slices,npoint,npoint=all_volumes.shape

    if resolution is not None:
        all_volumes=resize(all_volumes,(nb_gr,nb_slices,resolution,resolution))
    
    if axis is not None:
        all_volumes=np.moveaxis(all_volumes,axis+1,1)


    n = np.maximum(all_volumes.shape[-1],all_volumes.shape[-2])
    pad_1 = 2 ** (int(np.log2(n)) + 1) - n
    pad_2 = 2 ** int(np.log2(n) - 1) * 3 - n

    if pad_2 < 0:
        pad = int(pad_1 / 2)
    else:
        pad = int(pad_2 / 2)

    # if n%2==0:
    #     pad=0
    pad_x=int((2*pad+n-all_volumes.shape[-2])/2)
    pad_y=int((2*pad+n-all_volumes.shape[-1])/2)

    pad_amount = ((0, 0), (pad_x, pad_x), (pad_y, pad_y))
    print(pad_amount)

    nb_features=config_train["nb_features"]
    inshape=np.pad(all_volumes[0],pad_amount,mode="constant").shape[1:]
    vxm_model=vxm.networks.VxmDense(inshape, nb_features, int_steps=0)
    vxm_model.load_weights(file_model)
    registered_volumes=copy(all_volumes)
    mapxbase_all=np.zeros_like(all_volumes)
    mapybase_all = np.zeros_like(all_volumes)


    i=0
    while i<niter:
        print("Registration for iter {}".format(i+1))
        for gr in range(nb_gr):
            registered_volumes[gr],mapxbase_all[gr],mapybase_all[gr]=register_motionbin(vxm_model,all_volumes,gr,pad_amount,deformation_map)

        all_volumes=copy(registered_volumes)
        deformation_map=np.stack([mapxbase_all,mapybase_all],axis=0)
        print(deformation_map.shape)
        i+=1

    if axis is not None:
        registered_volumes=np.moveaxis(registered_volumes,1,axis+1)
        deformation_map=np.moveaxis(deformation_map,2,axis+2)

    if resolution is not None:
        deformation_map=resize(deformation_map,(2,nb_gr,nb_slices,npoint,npoint),order=3)
    np.save(filename_registered_volumes,registered_volumes)
    np.save(filename_deformation, deformation_map)
    return filename_registered_volumes


def generate_movement_gif(file_volume,sl,l=0):

    '''
        Returns dynamic movie of volumes for specified slice
        inputs:
            file_volume: .npy file containing respiratory volumes for all bins (size respiratory bins count x nb slices x npoint x npoint if motion scan data or respiratory bins count x nb singular volumes x nb slices x npoint x npoint if mrf scan data)
            sl: int specifying the slice 
            l : int specifying singular volume for mrf scan data 
        outputs:
            None
            saves filename_gif containing the movie (.gif)
    '''

    filename_gif = str.split(file_volume, ".npy")[0] + "_sl{}_moving.gif".format(sl)
    test_volume=np.load(file_volume.format(0)).squeeze()
    
    

    if test_volume.ndim==4:#file contains all phases for one singular volume
        all_matched_volumes=test_volume


    elif test_volume.ndim==5:#file contains all phases and all singular volumes
        all_matched_volumes=test_volume[:,l]
        filename_gif=filename_gif.replace("moving.gif","moving_l{}.gif".format(l))

    moving_image=np.concatenate([all_matched_volumes[:,sl],all_matched_volumes[1:-1,sl][::-1]],axis=0)
    
    
    gif=[]

    
    volume_for_gif = np.abs(moving_image)
    

    for i in range(volume_for_gif.shape[0]):
        min_value=np.min(volume_for_gif[i])
        max_value=np.max(volume_for_gif[i])
        img = Image.fromarray(np.uint8((volume_for_gif[i]-min_value)/(max_value-min_value)*255), 'L')
        img=img.convert("P")
        gif.append(img)


    gif[0].save(filename_gif,save_all=True, append_images=gif[1:], optimize=False, duration=100, loop=0)
    print(filename_gif)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenamevol',type=str)
    parser.add_argument('--fileconfig', type=str)
    parser.add_argument('--nepochs', type=int)
    parser.add_argument('--keptbins', nargs='?', const=None, type=str, default=None)

    args = parser.parse_args()

    filename_volumes=args.filenamevol
    with open(args.fileconfig, 'r') as file:
        config_train = json.load(file)
    nepochs=args.nepochs
    kept_bins=args.keptbins

    print("Training the neural network based on input volumes")
    file_model=train_voxelmorph(filename_volumes,config_train,nepochs,kept_bins=kept_bins)

    print("Evaluating the neural network to estimate deformation map and register moving volumes")
    filename_registered_volumes=register_allbins_to_baseline(filename_volumes,file_model,config_train)

    sl=46
    print("Building .gif files of unregistered and registered volumes for slice {}".format(sl))
    generate_movement_gif(filename_volumes,sl)
    generate_movement_gif(filename_registered_volumes,sl)
    

