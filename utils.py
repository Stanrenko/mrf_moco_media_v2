import numpy as np
import math
import finufft
from tqdm import tqdm
import pandas as pd
import cv2
import scipy as sp
from scipy import ndimage

class Trajectory(object):

    def __init__(self,applied_timesteps=None,**kwargs):
        self.paramDict=kwargs
        self.traj = None
        self.traj_for_reconstruction=None
        self.applied_timesteps=applied_timesteps
        self.reconstruct_each_partition = False #For 3D - whether all reps are used for generating the kspace data or only the current partition


    def get_traj(self):
        #Returns and stores the trajectory array of ntimesteps * total number of points * ndim
        raise ValueError("get_traj should be implemented in child")

    def get_traj_for_reconstruction(self,timesteps=175):
        if self.traj_for_reconstruction is not None:
            print("Warning : Outputting the stored reconstruction traj - timesteps input has no impact - please reset with self.traj_for_reconstruction=None")
            return self.traj_for_reconstruction

        else:
            traj = self.get_traj()
            return traj.reshape(timesteps,-1,traj.shape[-1])


    def adjust_traj_for_window(self,window):
        traj=self.get_traj()
        traj_shape=traj.shape
        traj=np.array(groupby(traj,window))
        traj=traj.reshape((-1,)+traj_shape[1:])
        self.traj=traj

class Radial(Trajectory):

    def __init__(self,total_nspokes=1400,npoint=512,**kwargs):
        super().__init__(**kwargs)

        self.paramDict["total_nspokes"]=total_nspokes #total nspokes per rep
        self.paramDict["npoint"] = npoint
        self.paramDict["nb_rep"]=1

    def get_traj(self):
        if self.traj is None:
            npoint = self.paramDict["npoint"]
            total_nspokes = self.paramDict["total_nspokes"]
            all_spokes = radial_golden_angle_traj(total_nspokes, npoint)
            #traj = np.reshape(groupby(all_spokes, nspoke), (-1, npoint * nspoke))
            traj = all_spokes
            traj=np.stack([traj.real,traj.imag],axis=-1)
            self.traj=traj

        return self.traj


class Radial3D(Trajectory):

    def __init__(self,total_nspokes=1400,nspoke_per_z_encoding=8,npoint=512,undersampling_factor=1,incoherent=False,is_random=False,mode="old",offset=0,golden_angle=True,nb_rep_center_part=1,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["total_nspokes"] = total_nspokes
        self.paramDict["nspoke"] = nspoke_per_z_encoding
        self.paramDict["npoint"] = npoint
        self.paramDict["undersampling_factor"] = undersampling_factor
        self.paramDict["nb_rep"]=math.ceil(self.paramDict["nb_slices"]/self.paramDict["undersampling_factor"])
        print(self.paramDict["nb_rep"])
        self.paramDict["random"]=is_random
        self.paramDict["incoherent"]=incoherent
        self.paramDict["mode"] = mode
        if self.paramDict["mode"]=="Kushball":
            self.paramDict["incoherent"]=True
        
        self.paramDict["offset"] = offset
        self.paramDict["golden_angle"]=golden_angle
        self.paramDict["nb_rep_center_part"] = nb_rep_center_part


    def get_traj(self):
        if self.traj is None:
            nspoke = self.paramDict["nspoke"]
            npoint = self.paramDict["npoint"]
            mode = self.paramDict["mode"]
            offset=self.paramDict["offset"]

            total_nspokes = self.paramDict["total_nspokes"]
            nb_slices=self.paramDict["nb_slices"]
            undersampling_factor=self.paramDict["undersampling_factor"]

            nb_rep_center_part=self.paramDict["nb_rep_center_part"]

            if self.paramDict["golden_angle"]:
                if self.paramDict["mode"]=="Kushball":
                    self.traj=self.traj=spherical_golden_angle_means_traj_3D(total_nspokes, npoint, nb_slices,undersampling_factor)

                
                else:
                    if self.paramDict["incoherent"]:
                        self.traj=radial_golden_angle_traj_3D_incoherent(total_nspokes, npoint, nspoke, nb_slices, undersampling_factor,mode,offset)
                    else:
                        self.traj = radial_golden_angle_traj_3D(total_nspokes, npoint, nspoke, nb_slices,
                                                                undersampling_factor,nb_rep_center_part)

            else:
                self.traj=distrib_angle_traj_3D(total_nspokes, npoint, nspoke, nb_slices,
                                                                undersampling_factor)


        return self.traj



class Navigator3D(Trajectory):

    def __init__(self,direction=[0.0,0.0,1.0],npoint=512,nb_slices=1,undersampling_factor=1,nb_gating_spokes=50,**kwargs):
        super().__init__(**kwargs)
        self.paramDict["total_nspokes"] = nb_gating_spokes
        self.paramDict["npoint"] = npoint
        self.paramDict["direction"] = direction
        self.paramDict["nb_slices"] = nb_slices
        self.paramDict["undersampling_factor"] = undersampling_factor
        self.paramDict["nb_rep"] = int(self.paramDict["nb_slices"] / self.paramDict["undersampling_factor"])
        self.reconstruct_each_partition=True

    def get_traj(self):
        if self.traj is None:
            npoint = self.paramDict["npoint"]
            direction=self.paramDict["direction"]
            #nb_rep=self.paramDict["nb_rep"]
            total_nspoke=self.paramDict["total_nspokes"]
            k_max=np.pi

            base_spoke=(-k_max+np.arange(npoint)*2*k_max/(npoint-1)).reshape(-1,1)*np.array(direction).reshape(1,-1)
            self.traj=np.repeat(np.expand_dims(base_spoke,axis=0),axis=0,repeats=total_nspoke)


        return self.traj



def radial_golden_angle_traj(total_nspoke,npoint,k_max=np.pi):
    golden_angle=111.246*np.pi/180
    base_spoke = (-k_max+k_max/(npoint)+np.arange(npoint)*2*k_max/(npoint))
    all_rotations = np.exp(1j * np.arange(total_nspoke) * golden_angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes

def distrib_angle_traj(total_nspoke,npoint,k_max=np.pi):
    angle=2*np.pi/total_nspoke
    base_spoke = -k_max+np.arange(npoint)*2*k_max/(npoint-1)
    all_rotations = np.exp(1j * np.arange(total_nspoke) * angle)
    all_spokes = np.matmul(np.diag(all_rotations), np.repeat(base_spoke.reshape(1, -1), total_nspoke, axis=0))
    return all_spokes


def radial_golden_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4,nb_rep_center_part=1):
    timesteps = int(total_nspoke / nspoke)

    nb_rep = math.ceil((nb_slices ) / undersampling_factor)+nb_rep_center_part-1
    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)

    k_z = np.zeros((timesteps, nb_rep))
    all_slices=np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)
    

    k_z[0, :] = all_slices[::undersampling_factor]


    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    if nb_rep_center_part>1:
        center_part=all_slices[int(nb_slices/2)]
        k_z_new= np.zeros((timesteps, nb_rep))
        for j in range( k_z.shape[0]):
            num_center_part=np.argwhere(k_z[j]==center_part)[0][0]
            k_z_new[j,:num_center_part]=k_z[j,:num_center_part]
            k_z_new[j,(num_center_part+nb_rep_center_part):]=k_z[j,(num_center_part+1):]
        print(k_z_new[0,:])
        k_z=k_z_new


    k_z=np.repeat(k_z, nspoke, axis=0)

    k_z = np.expand_dims(k_z, axis=-1)


    traj = np.expand_dims(all_spokes, axis=-2)

    k_z, traj = np.broadcast_arrays(k_z, traj)

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])


def distrib_angle_traj_3D(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=4):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = int(nb_slices / undersampling_factor)
    all_spokes = distrib_angle_traj(total_nspoke, npoint)

    k_z = np.zeros((timesteps, nb_rep))
    all_slices=np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)
    k_z[0, :] = all_slices[::undersampling_factor]

    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j)[::undersampling_factor])

    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    traj = np.expand_dims(all_spokes, axis=-2)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)
    return result.reshape(result.shape[0],-1,result.shape[-1])

def spherical_golden_angle_means_traj_3D(total_nspoke, npoint, npart, undersampling_factor=4,k_max=np.pi):
    
    phi1=0.46557123
    phi2=0.6823278

    theta=2*np.pi*np.mod(np.arange(total_nspoke*npart)*phi2,1)
    phi=np.arccos(np.mod(np.arange(total_nspoke*npart)*phi1,1))

    rotation=np.stack([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)],axis=1).reshape(-1,1,3)
    base_spoke = (-k_max+k_max/(npoint)+np.arange(npoint)*2*k_max/(npoint))

    base_spoke=base_spoke.reshape(-1,1)
    spokes=np.matmul(base_spoke,rotation).reshape(npart,total_nspoke,npoint,-1)
    spokes=np.moveaxis(spokes,0,1)
    return spokes.reshape(total_nspoke,-1,3)



def radial_golden_angle_traj_3D_incoherent(total_nspoke, npoint, nspoke, nb_slices, undersampling_factor=1,mode="old",offset=0):
    timesteps = int(total_nspoke / nspoke)
    nb_rep = math.ceil(nb_slices / undersampling_factor)

    golden_angle = 111.246 * np.pi / 180
    all_slices = np.arange(-np.pi, np.pi, 2 * np.pi / nb_slices)

    all_spokes = radial_golden_angle_traj(total_nspoke, npoint)
    if mode=="old":
        all_rotations = np.exp(1j * np.arange(nb_slices) * total_nspoke * golden_angle)
    elif mode=="new":
        all_rotations = np.exp(1j * np.arange(nb_slices) * golden_angle)
    else:
        raise ValueError("Unknown value for mode")

    all_spokes = np.repeat(np.expand_dims(all_spokes, axis=1), nb_slices, axis=1)
    traj = all_rotations[np.newaxis, :, np.newaxis] * all_spokes

    k_z=np.zeros((timesteps, nb_slices))
    k_z[0, :] = all_slices
    for j in range(1, k_z.shape[0]):
        k_z[j, :] = np.sort(np.roll(all_slices, -j))

    print(traj.shape)
    k_z=np.repeat(k_z, nspoke, axis=0)
    k_z = np.expand_dims(k_z, axis=-1)
    k_z, traj = np.broadcast_arrays(k_z, traj)

    result = np.stack([traj.real,traj.imag, k_z], axis=-1)


    if undersampling_factor>1:
        result = result.reshape(timesteps, nspoke, -1, npoint, result.shape[-1])

        result_us=np.zeros((timesteps, nspoke, nb_rep, npoint, 3),
                          dtype=result.dtype)
        
        #result_us[:, :, :, :, 1:] = result[:, :, :nb_rep, :, 1:]
        #print(result_us.shape)
        shift = offset

        for sl in range(nb_slices):

            if int(sl/undersampling_factor)<nb_rep:
                result_us[shift::undersampling_factor, :, int(sl/undersampling_factor), :, :] = result[shift::undersampling_factor, :, sl, :, :]
                shift += 1
                shift = shift % (undersampling_factor)
            else:
                continue

        result=result_us


    return result.reshape(total_nspoke,-1,3)




def simulate_nav_images_multi(kdata, trajectory, image_size=(400,), b1=None):
    traj = trajectory.get_traj()
    nb_channels = kdata.shape[0]
    npoint = kdata.shape[-1]
    nb_slices = kdata.shape[1]
    nb_gating_spokes = kdata.shape[2]
    npoint_image=image_size[0]

    if kdata.dtype == "complex64":
        traj=traj.astype("float64")

    if kdata.dtype == "complex128":
        traj=traj.astype("float128")

    if b1 is not None:
        if b1.ndim == 2:
            b1 = np.expand_dims(b1, axis=(1, 2))
        elif b1.ndim == 3:
            b1 = np.expand_dims(b1, axis = (1))

    traj = traj.astype(np.float32)

    kdata = kdata.reshape((nb_channels, -1, npoint))
    images_series_rebuilt_nav = np.zeros((nb_slices, nb_gating_spokes, npoint_image), dtype=np.complex64)

    for i in tqdm(range(nb_channels)):
        fk = finufft.nufft1d1(traj[0, :, 2], kdata[i, :, :], image_size)
        fk = fk.reshape((nb_slices, nb_gating_spokes, npoint_image))


        if b1 is None:
            images_series_rebuilt_nav += np.abs(fk) ** 2
        else:
            images_series_rebuilt_nav += b1[i].conj() * fk

    if b1 is None:
        images_series_rebuilt_nav = np.sqrt(images_series_rebuilt_nav)

    return images_series_rebuilt_nav


def groupby(arr, n, axis=0, mode="edge"):
    """ group array into groups of size 'n' """

    ngroup = -(-arr.shape[axis] // n)
    if arr.shape[axis] % n != 0:
        # pad array
        padding = [(0,0)] * arr.ndim
        nzero = n - np.mod(arr.shape[axis], n)
        padding[axis] = (nzero//2, -(-nzero//2))
        arr = np.pad(arr, padding, mode=mode)
    arr = np.moveaxis(arr, axis, 0)
    arr = arr.reshape((ngroup, -1) + arr.shape[1:])
    return list(np.moveaxis(arr, 1, axis + 1))




def correct_mvt_kdata_zero_filled(trajectory,cond,ntimesteps):
    '''
    Calculate density compensation weights for radial stack of stars trajectory with missing spokes (e.g. due to respiratory binning)
    inputs
    trajectory: Trajectory object containing the features of the sampling trajectory
    cond: boolean array for identifying retained spokes
    ntimesteps: total number of timesteps (if 1, all the spokes are retained for building the image so the density compensation is based on all the retained spokes)

    outputs
    weights: size nb readouts per mrf repetition x nb repetition 
    retained_ts: in the case multiple images are rebuilt, lists the retained time steps as some timesteps might not have any spokes

    '''

    traj=trajectory.get_traj()

    mode=trajectory.paramDict["mode"]
    incoherent=trajectory.paramDict["incoherent"]

    nb_rep = int(cond.shape[0]/traj.shape[0])
    npoint = int(traj.shape[1] / nb_rep)
    nspoke = int(traj.shape[0] / ntimesteps)


    traj_for_selection = np.array(groupby(traj, npoint, axis=1))
    traj_for_selection = traj_for_selection.reshape(cond.shape[0], -1, 3)
    indices = np.unravel_index(np.argwhere(cond).T,(nb_rep,ntimesteps,nspoke))
    retained_indices = np.squeeze(np.array(indices).T)
    retained_timesteps = np.unique(retained_indices[:, 1])
    ## DENSITY CORRECTION

    df = pd.DataFrame(columns=["rep", "ts", "spoke", "kz", "theta"], index=range(nb_rep * ntimesteps * nspoke))
    df["rep"] = np.repeat(list(range(nb_rep)), ntimesteps * nspoke)
    df["ts"] = list(np.repeat(list(range(ntimesteps)), (nspoke))) * nb_rep
    df["spoke"] = list(range(nspoke)) * nb_rep * ntimesteps

    df["kz"] = traj_for_selection[:, :, 2][:, 0]
    golden_angle = 111.246 * np.pi / 180

    if not(incoherent):
        df["theta"] = np.array(list(np.mod(np.arange(0, int(df.shape[0] / nb_rep)) * golden_angle, np.pi)) * nb_rep)
    elif incoherent:
        if mode=="old":
            df["theta"] = np.array(list(np.mod(np.arange(0, int(df.shape[0])) * golden_angle, np.pi)))
        elif mode=="new":
            df["theta"] = np.mod((np.array(list(np.arange(0, nb_rep) * golden_angle)).reshape(-1,1)+np.array(list(np.arange(0, int(df.shape[0] / nb_rep)) * golden_angle)).reshape(1,-1)).flatten(),np.pi)


    df_retained = df.iloc[np.nonzero(cond)]
    kz_by_timestep = df_retained.groupby("ts")["kz"].unique()
    theta_by_rep_timestep = df_retained.groupby(["ts", "rep"])["theta"].unique()

    df_retained = df_retained.join(kz_by_timestep, on="ts", rsuffix="_s")
    df_retained = df_retained.join(theta_by_rep_timestep, on=["ts", "rep"], rsuffix="_s")

    # Theta weighting
    df_retained["theta_s"] = df_retained["theta_s"].apply(lambda x: np.sort(x))
    #df_retained["theta_s"] = df_retained["theta_s"].apply(
    #    lambda x: np.concatenate([[x[-1] - np.pi], x, [x[0] + np.pi]]))
    df_retained["theta_s"] = df_retained["theta_s"].apply(
        lambda x: np.unique(np.concatenate([[0], x, [np.pi]])))
    diff_theta=(df_retained.theta - df_retained["theta_s"])
    theta_inside_boundary=(df_retained["theta"]!=0)*(df_retained["theta"]!=np.pi)
    df_retained["theta_inside_boundary"] = theta_inside_boundary

    min_theta = df_retained.groupby(["ts", "rep"])["theta"].min()
    max_theta = df_retained.groupby(["ts", "rep"])["theta"].max()
    df_retained = df_retained.join(min_theta, on=["ts", "rep"], rsuffix="_min")
    df_retained = df_retained.join(max_theta, on=["ts", "rep"], rsuffix="_max")
    is_min_theta = (df_retained["theta"]==df_retained["theta_min"])
    is_max_theta = (df_retained["theta"] == df_retained["theta_max"])
    df_retained["is_min_theta"] = is_min_theta
    df_retained["is_max_theta"] = is_max_theta

    df_retained["theta_weight"] = theta_inside_boundary*diff_theta.apply(lambda x: (np.sort(x[x>=0])[1]+np.sort(-x[x<=0])[1])/2 if ((x>=0).sum()>1) and ((x<=0).sum()>1) else 0)+ \
                                  (1-theta_inside_boundary)*diff_theta.apply(lambda x: np.sort(np.abs(x))[1]/2)

    df_retained["theta_weight_before_correction"] = df_retained["theta_weight"]

    df_retained["theta_weight"] = df_retained["theta_weight"]+ (theta_inside_boundary)* ((is_min_theta)*df_retained["theta"]+(is_max_theta)*(np.pi-df_retained["theta"]))/2

    df_retained.loc[df_retained["theta_weight"].isna(), "theta_weight"] = 1.0
    sum_weights = df_retained.groupby(["ts", "rep"])["theta_weight"].sum()
    df_retained = df_retained.join(sum_weights, on=["ts", "rep"], rsuffix="_sum")
    #df_retained["theta_weight"] = df_retained["theta_weight"] / df_retained["theta_weight_sum"]

    # KZ weighting
    #df_retained.loc[df_retained.ts == 138].to_clipboard()
    df_retained["kz_s"] = df_retained["kz_s"].apply(lambda x: np.unique(np.concatenate([[-np.pi], x, [np.pi]])))
    diff_kz=(df_retained.kz - df_retained["kz_s"])
    kz_inside_boundary=(df_retained["kz"].abs()!=np.pi)
    df_retained["kz_inside_boundary"]=kz_inside_boundary

    min_kz = df_retained.groupby(["ts"])["kz"].min()
    max_kz = df_retained.groupby(["ts"])["kz"].max()
    df_retained = df_retained.join(min_kz, on=["ts"], rsuffix="_min")
    df_retained = df_retained.join(max_kz, on=["ts"], rsuffix="_max")

    is_min_kz = (df_retained["kz"] == df_retained["kz_min"])
    is_max_kz = (df_retained["kz"] == df_retained["kz_max"])

    df_retained["is_min_kz"] = is_min_kz
    df_retained["is_max_kz"] = is_max_kz

    df_retained["kz_weight"] = kz_inside_boundary*diff_kz.apply(lambda x: (np.sort(x[x >= 0])[1] + np.sort(-x[x <= 0])[1]) / 2 if ((x >= 0).sum() > 1) and (
        ((x <= 0).sum() > 1)) else 0)+(1-kz_inside_boundary)*diff_kz.apply(lambda x: np.sort(np.abs(x))[1]/2)



    df_retained["kz_weight"] = df_retained["kz_weight"] + (kz_inside_boundary) * (
            (is_min_kz) * (df_retained["kz"]+np.pi) + (is_max_kz) * (np.pi - df_retained["kz"])) / 2

    df_retained.loc[df_retained["kz_weight"].isna(), "kz_weight"] = 1.0
    sum_weights = df_retained.drop_duplicates(subset=["kz","ts"])
    sum_weights = sum_weights.groupby(["ts"])["kz_weight"].apply(lambda x: x.sum())
    df_retained = df_retained.join(sum_weights, on=["ts"], rsuffix="_sum")
    #df_retained["kz_weight"] = df_retained["kz_weight"] / df_retained["kz_weight_sum"]
    theta_weight = df_retained["theta_weight"]
    kz_weight = df_retained["kz_weight"]
    weights = np.zeros(cond.shape[0])
    weights[cond] = theta_weight * kz_weight
    weights = weights.reshape(nb_rep, -1)
    weights = np.moveaxis(weights, 0, 1)
    weights = weights.reshape(ntimesteps, nspoke, -1)
    weights=weights[retained_timesteps]

    return weights,retained_timesteps




def format_input_voxelmorph(all_volumes,pad_amount,sl_down=5,sl_top=-5,normalize=True,all_groups_combination=False,exclude_zero_slices=True):

    nb_gr=all_volumes.shape[0]
    nb_slices=all_volumes.shape[1]
    fixed_volume=[]
    moving_volume=[]

    #Filtering out slices with only 0 as it seems to be buggy

    if exclude_zero_slices:
        sl_down_non_zeros = 0
        while not(np.any(all_volumes[:,sl_down_non_zeros])):
            sl_down_non_zeros+=1

        sl_top_non_zeros=nb_slices
        while not(np.any(all_volumes[:,sl_top_non_zeros-1])):
            sl_top_non_zeros-=1


        sl_down=np.maximum(sl_down,sl_down_non_zeros)
        sl_top=np.minimum(sl_top,sl_top_non_zeros)


    for gr in range(nb_gr-1):
        fixed_volume.append(all_volumes[gr,sl_down:sl_top])
        moving_volume.append(all_volumes[gr+1,sl_down:sl_top])

    if all_groups_combination:
        shift=2
        while shift<nb_gr:
            for gr in range(nb_gr - shift):
                fixed_volume.append(all_volumes[gr, sl_down:sl_top])
                moving_volume.append(all_volumes[gr + shift, sl_down:sl_top])
            shift+=1


    fixed_volume=np.array(fixed_volume)
    moving_volume=np.array(moving_volume)

    fixed_volume=fixed_volume.reshape(-1,fixed_volume.shape[-2],fixed_volume.shape[-1])
    moving_volume=moving_volume.reshape(-1,moving_volume.shape[-2],moving_volume.shape[-1])

    if normalize:
        fixed_volume/=np.max(fixed_volume,axis=(1,2),keepdims=True)
        moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)


    # fix data
    fixed_volume = np.pad(fixed_volume, pad_amount, 'constant')
    moving_volume = np.pad(moving_volume, pad_amount, 'constant')

    return fixed_volume,moving_volume



def vxm_data_generator(x_data_fixed, x_data_moving, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data_fixed.shape[1:]  # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data_moving.shape[0], size=batch_size)
        moving_images = x_data_moving[idx1, ..., np.newaxis]
        fixed_images = x_data_fixed[idx1, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)


def scheduler(epoch, lr,decay=0.005,min_lr=None):
  if epoch < 20:
    return lr
  else:
    if min_lr is None:
        return lr * tf.math.exp(-decay)
    else:
        return np.maximum(lr * tf.math.exp(-decay),min_lr)



def register_motionbin(vxm_model,all_volumes,gr,pad_amount,deformation_map=None):
    curr_gr=gr
    moving_volume=np.pad(all_volumes[curr_gr],pad_amount,mode="constant")
    nb_slices=all_volumes.shape[1]

    if deformation_map is None:
        mapx_base, mapy_base = np.meshgrid(np.arange(all_volumes.shape[-1]), np.arange(all_volumes.shape[-2]))
        mapx_base=np.tile(mapx_base,reps=(nb_slices,1,1))
        mapy_base = np.tile(mapy_base, reps=(nb_slices, 1, 1))
    else:
        
        mapx_base=deformation_map[0,gr]
        mapy_base=deformation_map[1,gr]

    while curr_gr>0:
        input=np.stack([np.pad(all_volumes[curr_gr-1],pad_amount,mode="constant"),moving_volume],axis=0)

        x_val_fixed,x_val_moving=format_input_voxelmorph(input,((0,0),(0,0),(0,0)),sl_down=0,sl_top=nb_slices,exclude_zero_slices=False)
        val_input=[x_val_moving[...,None],x_val_fixed[...,None]]

        val_pred=vxm_model.predict(val_input)
        moving_volume=val_pred[0][:,:,:,0]

        mapx_base=mapx_base+unpad(val_pred[1][:,:,:,1],pad_amount)
        mapy_base=mapy_base+unpad(val_pred[1][:,:,:,0],pad_amount)

        curr_gr=curr_gr-1

    if gr==0:
        moving_volume/=np.max(moving_volume,axis=(1,2),keepdims=True)
    unpadded_moving_volume=unpad(moving_volume,pad_amount)

          
    return unpadded_moving_volume,mapx_base,mapy_base



def unpad(x, pad_width):
    slices = []
    for c in pad_width:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]



def change_deformation_map_ref(deformation_map_allbins,index_ref,axis=None):
    '''
    changes the reference bin for the deformation map

    input :
    deformation_map_allbins nb_gr x 2 x nz x nx x ny

    output :
    deformation_map_allbins_new  nb_gr x 2 x nz x nx x ny -> deformation map to transform all bins to the bin index_ref

    '''

    
    if axis is not None:
        deformation_map_allbins=np.moveaxis(deformation_map_allbins,axis+2,2)

    ndim,nb_gr,nb_slices,nx,ny=deformation_map_allbins.shape
    deformation_map_allbins_new=np.zeros_like(deformation_map_allbins)

    mapx_base, mapy_base = np.meshgrid(np.arange(ny), np.arange(nx))

    mapx_base=np.tile(mapx_base,reps=(nb_slices,1,1))
    mapy_base = np.tile(mapy_base, reps=(nb_slices, 1, 1))
    deformation_base=np.stack([mapx_base,mapy_base],axis=0)
    
    for gr in range(nb_gr):
        deformation_map_allbins_new[:,gr]=deformation_map_allbins[:,gr]-deformation_map_allbins[:,index_ref]+deformation_base

    if axis is not None:
        deformation_map_allbins_new=np.moveaxis(deformation_map_allbins_new,2,axis+2)

    return deformation_map_allbins_new


def apply_deformation_to_complex_volume(volume, deformation_map,interp=cv2.INTER_LINEAR,axis=None):
    '''
    input :
    volume nz x nx x ny type complex
    deformation_map 2 x nz x nx x ny
    output :
    deformed volume nz x nx x ny type complex
    '''
    deformed_volume = apply_deformation_to_volume(np.real(volume), deformation_map,interp,axis) + 1j * apply_deformation_to_volume(
        np.imag(volume), deformation_map,interp,axis)
    return deformed_volume



def apply_deformation_to_volume(volume, deformation_map,interp=cv2.INTER_LINEAR,axis=None):
    '''
    input :
    volume nz x nx x ny type float
    deformation_map 2 x nz x nx x ny
    output :
    deformed volume nz x nx x ny type float
    '''
    if axis is not None:
        volume=np.moveaxis(volume,axis+volume.ndim-3,volume.ndim-3)
        deformation_map=np.moveaxis(deformation_map,axis+1,1)

    # print(volume.shape)
    # print(deformation_map.shape)
    deformed_volume = np.zeros_like(volume)
    
    if volume.ndim==4:
        L0=volume.shape[0]
        nb_slices = volume.shape[1]
        for sl in range(nb_slices):
            
            mapx = deformation_map[0, sl]
            mapy = deformation_map[1, sl]
            for l in range(L0):
                img = volume[l,sl].astype("float32")
                deformed_volume[l,sl] = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), interp)
    else:
        
        nb_slices = volume.shape[0]
        for sl in range(nb_slices):
            img = volume[sl].astype("float32")
            mapx = deformation_map[0, sl]
            mapy = deformation_map[1, sl]
            deformed_volume[sl] = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), interp)

    if axis is not None:
        deformed_volume=np.moveaxis(deformed_volume,volume.ndim-3,axis+volume.ndim-3)

    return deformed_volume


def calculate_inverse_deformation_map(deformation_map,axis=None):
    '''
    input :
    deformation_map 2 x nz x nx x ny
    output :
    inv_deformation_map 2 x nz x nx x ny
    '''
    if axis is not None:
        deformation_map=np.moveaxis(deformation_map,axis+1,1)
    inv_deformation_map = np.zeros_like(deformation_map)
    nb_slices = deformation_map.shape[1]
    for sl in range(nb_slices):
        inv_deformation_map[:, sl] = np.moveaxis(invert_map(np.moveaxis(deformation_map[:, sl], 0, -1)), -1, 0)

    if axis is not None:
        inv_deformation_map=np.moveaxis(inv_deformation_map,1,axis+1)

    return inv_deformation_map


def invert_map(F):
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:, :, 1], I[:, :, 0] = np.indices((h, w))  # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    return P



def undersampling_operator_singular_new(volumes,trajectory,b1_all_slices=None,ntimesteps=175,density_adj=True,weights=None,retained_timesteps=None):
    """
    returns A.H @ W @ A @ volumes where A=F@sampling@coils and W are the reconstruction weights (binning with or without dcomp)
    """

    L0=volumes.shape[0]
    size=volumes.shape[1:]

    if b1_all_slices is None:
        b1_all_slices=np.ones((1,)+size,dtype="complex64")

    nb_channels=b1_all_slices.shape[0]

    nb_slices=size[0]

 
    traj = trajectory.get_traj_for_reconstruction(1)
    if retained_timesteps is not None:
        traj=traj[retained_timesteps]
    if not((type(weights)==int)or(weights is None)):
        weights=weights.flatten()
    traj = traj.reshape(-1, 2).astype("float32")
    npoint = trajectory.paramDict["npoint"]

    num_k_samples = traj.shape[0]

    output_shape = (L0,) + size
    images_series_rebuilt = np.zeros(output_shape, dtype=np.complex64)


    if (weights is not None) and not(type(weights)==int):
        weights = np.expand_dims(weights, axis=(0, -1))


    for k in tqdm(range(nb_channels)):
        curr_volumes = volumes * np.expand_dims(b1_all_slices[k], axis=0)
        curr_kdata_slice=np.fft.fftshift(sp.fft.fft(
            np.fft.ifftshift(curr_volumes, axes=1),
            axis=1,workers=24), axes=1).astype("complex64")
        curr_kdata = finufft.nufft2d2(traj[:, 0],traj[:, 1],curr_kdata_slice.reshape((L0*nb_slices,)+size[1:])).reshape(L0,nb_slices,-1)
        if density_adj:
            curr_kdata=curr_kdata.reshape(L0,-1,npoint)
            density = np.abs(np.linspace(-1, 1, npoint))
            density = np.expand_dims(density, tuple(range(curr_kdata.ndim - 1)))
            curr_kdata*=density

        if weights is not None:
            curr_kdata = curr_kdata.reshape((L0,-1,npoint))
            curr_kdata *= weights

        curr_kdata = curr_kdata.reshape(L0, nb_slices,traj.shape[0])
        curr_kdata = np.fft.fftshift(sp.fft.ifft(np.fft.ifftshift(curr_kdata,axes=1),axis=1,workers=24),axes=1).astype(np.complex64)

        images_series_rebuilt+=np.expand_dims(b1_all_slices[k].conj(), axis=0)* (finufft.nufft2d1(traj[:, 0],traj[:, 1],curr_kdata.reshape(L0*nb_slices,-1),size[1:])).reshape((L0,)+size)


    images_series_rebuilt /= num_k_samples
    return images_series_rebuilt



def build_mask_from_volume(volumes,threshold_factor=0.05,iterations=3):
    mask = False
    unique = np.histogram(np.abs(volumes), 100)[1]
    mask = mask | (np.abs(volumes) > unique[int(len(unique) * threshold_factor)])
    mask = ndimage.binary_closing(mask, iterations=iterations)
    return mask*1

def makevol(values, mask):
    """ fill volume """
    values = np.asarray(values)
    new = np.zeros(mask.shape, dtype=values.dtype)
    new[mask] = values
    return new