import numpy as np
import argparse
import pickle
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from copy import copy
from utils import *




def build_navigator_images(filename_nav_save,seasonal_adj=False):
    '''
    Reconstructs navigator images from raw k-space data (helper for selecting the channel for displacement extraction)
    inputs:
        filename_nav_save - file containing navigators raw data channels x repetitions x gating spokes count per repetition x npoint (.npy)
        seasonal_adj - boolean to use deseasonalization on the navigator images

    outputs: 
        None

        saves filename_image_nav (.npy), filename_image_nav_plot (.jpg) and filename_image_nav_diff_plot (.jpg) which contain respectively the navigator images, the image showing the first 100 navigator images the image showing the gradient of the first 100 navigator images

    '''
    filename_image_nav= filename_nav_save.split("_nav.npy")[0] + "_image_nav.npy"
    filename_image_nav_plot = filename_nav_save.split("_nav.npy")[0] + "_image_nav.jpg"
    filename_image_nav_diff_plot = filename_nav_save.split("_nav.npy")[0] + "_image_nav_diff.jpg"

    data_for_nav = np.load(filename_nav_save)
    nb_channels = data_for_nav.shape[0]
    npoint_nav = data_for_nav.shape[-1]
    nb_gating_spokes = data_for_nav.shape[-2]
    nb_slices = data_for_nav.shape[1]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                           nb_gating_spokes=nb_gating_spokes)
    nav_image_size = (int(npoint_nav ),)

    image_nav_all_channels = []

    # for j in tqdm(range(nb_channels)):
    for j in tqdm(range(nb_channels)):
        images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[j], axis=0), nav_traj,
                                                                 nav_image_size, b1=None)
        image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
        image_nav_all_channels.append(image_nav_ch)

    image_nav_all_channels = np.array(image_nav_all_channels)
    if seasonal_adj:
        from statsmodels.tsa.seasonal import seasonal_decompose

        image_reshaped = image_nav_all_channels.reshape(-1, npoint_nav)
        decomposition = seasonal_decompose(image_reshaped,
                                           model='multiplicative', period=nb_gating_spokes)
        image=image_reshaped/decomposition.seasonal
        image=image.reshape(-1,nb_gating_spokes,npoint_nav)
        image_nav_all_channels=image
        print(image.shape)


    np.save(filename_image_nav,image_nav_all_channels)

    plot_image_grid(
        np.moveaxis(image_nav_all_channels.reshape(nb_channels, -1, int(npoint_nav)), -1, -2)[:, :, :100],
        nb_row_col=(6, 6),save_file=filename_image_nav_plot)

    plot_image_grid(
        np.moveaxis(np.diff(image_nav_all_channels.reshape(nb_channels, -1, int(npoint_nav)), axis=-1), -1, -2)[:,
        :, :100], nb_row_col=(6, 6),save_file=filename_image_nav_diff_plot)

    print("Navigator images plot file: {}".format(filename_image_nav_diff_plot))

    return

def calculate_displacement_weights(filename_nav_save,nb_segments,bottom,top,incoherent,ch,filename_bins,retained_categories,nbins,gating_only,pad,equal_spoke_per_bin,seasonal_adj,hard_interp,interp_bad_correl,soft_weight):

    '''
    Displacement calculation from raw navigator K-space data
    inputs:
        filename_nav_save - file containing navigators raw data channels x repetitions x gating spokes count per repetition x npoint (.npy)
        nb_segments - number of readouts per repetition in the mrf sequence
        bottom - minimum displacement (useful for initialization of the displacement calculation)
        top - maxium displacement
        ch - chosen channel for displacement calculation (based on navigator images)
        filename_bins - file containing the edges of the bins (used for using the same bins in the mrf scan and in the motion scan)
        retained_categories - string for selecting retained bins (might be useful if one wants to exclude one bin e.g. inspiration) e.g. "0,1,2,3,4"
        nbins - total number of bins
        gating_only - boolean, if true, the weights don't include any density compensation, and are only used for gating data according to respiratory bin
        pad - padding the navigator images for facilitating displacement calculation
        equal_spoke_per_bin - boolean selecting if the bins should be selected using equal spoke per bin or equal distance
        seasonal_adj - boolean to use deseasonalization on the navigator images
        hard_interp - boolean to use hard coded interpolation on mrf for the initial inversion
        interp_bad_correl - boolean to interpolate displacements with neighbours when poor correlation with baseline navigator image
        soft_weight - boolean to use soft weighing for extreme respiratory bins

    outputs: 
        None

        saves filename_displacement, filename_weights and filename_bins_output (.npy) which contain respectively the displacement (repetitions x gating spokes count per repetition), weighing (1 x nb_segments x repetitions x 1) and bins (nbins)

    '''

    ntimesteps =1


    filename_displacement=filename_nav_save.split("_nav.npy")[0] + "_displacement.npy"
    filename_weights=filename_nav_save.split("_nav.npy")[0] + "_weights.npy"
    filename_retained_ts=filename_nav_save.split("_nav.npy")[0] + "_retained_ts.pkl"
    filename_bins_output = filename_nav_save.split("_nav.npy")[0] + "_bins.npy"

    folder = "/".join(str.split(filename_nav_save, "/")[:-1])



    data_for_nav=np.load(filename_nav_save)




    displacements = calculate_displacements_singlechannel(data_for_nav, nb_segments, shifts=list(range(bottom, top)),ch=ch,pad=pad,seasonal_adj=seasonal_adj,interp_bad_correl=interp_bad_correl)



    nb_slices=data_for_nav.shape[1]
    nb_gating_spokes=data_for_nav.shape[2]


    if hard_interp:
        disp_interp=copy(displacements).reshape(-1,nb_gating_spokes)
        disp_interp[:, :8] = ((disp_interp[:, 7] - disp_interp[:, 0]) / (7 - 0))[:, None] * np.arange(8)[None,
                                                                                            :] + disp_interp[:, 0][:,
                                                                                                 None]
        displacements = disp_interp.flatten()
        np.save(filename_displacement, displacements)

    filename_displacement_plot = filename_displacement.replace(".npy",".jpg")
    plt.plot(displacements.flatten())
    plt.savefig(filename_displacement_plot)

    radial_traj=Radial3D(total_nspokes=nb_segments,undersampling_factor=1,npoint=800,nb_slices=nb_slices,incoherent=incoherent)

    if filename_bins is None:
        dico_traj_retained,dico_retained_ts,bins=estimate_weights_bins(displacements,nb_slices,nb_segments,nb_gating_spokes,ntimesteps,radial_traj,nb_bins=nbins,retained_categories=retained_categories,equal_spoke_per_bin=equal_spoke_per_bin,soft_weight=soft_weight)
        np.save(filename_bins_output,bins)
    else:
        bins=np.load(filename_bins)
        nb_bins=len(bins)+1
        print(nb_bins)
        dico_traj_retained,dico_retained_ts,_=estimate_weights_bins(displacements,nb_slices,nb_segments,nb_gating_spokes,ntimesteps,radial_traj,nb_bins=nb_bins,retained_categories=retained_categories,bins=bins,soft_weight=soft_weight)

    weights=[]
    for gr in dico_traj_retained.keys():
        weights.append(np.expand_dims(dico_traj_retained[gr],axis=-1))
    weights=np.array(weights)
    if gating_only:
        weights=(weights>0)*1
    np.save(filename_weights, weights)

    return


def calculate_displacements_singlechannel(data_for_nav, nb_segments, shifts=list(range(-5, 5)), ch=0,
                                          pad=10,seasonal_adj=False,
                                          interp_bad_correl=False):
    print("Processing Nav Data...")
    nb_allspokes = nb_segments
    nb_slices = data_for_nav.shape[1]
    nb_gating_spokes = data_for_nav.shape[2]
    npoint_nav = data_for_nav.shape[-1]

    all_timesteps = np.arange(nb_allspokes)
    nav_timesteps = all_timesteps[::int(nb_allspokes / nb_gating_spokes)]

    nav_traj = Navigator3D(direction=[0, 0, 1], npoint=npoint_nav, nb_slices=nb_slices,
                           applied_timesteps=list(nav_timesteps))

    nav_image_size = (int(npoint_nav),)
    print("Estimating Movement...")
    bottom = np.maximum(-shifts[0], int(nav_image_size[0] / 4))
    top = np.minimum(int(npoint_nav + 2 * pad) - shifts[-1], int(3 * nav_image_size[0] / 4))

    images_series_rebuilt_nav_ch = simulate_nav_images_multi(np.expand_dims(data_for_nav[ch], axis=0), nav_traj,
                                                             nav_image_size, b1=None)
    images_series_rebuilt_nav_ch = np.pad(images_series_rebuilt_nav_ch, pad_width=((0, 0), (0, 0), (pad, pad)),
                                          mode="edge")

    image_nav_ch = np.abs(images_series_rebuilt_nav_ch)
    displacements = calculate_displacement(image_nav_ch, bottom, top, shifts,seasonal_adj=seasonal_adj,
                                           interp_bad_correl=interp_bad_correl)

    max_slices = nb_slices
    displacements = displacements.reshape(nb_slices, -1)
    As_hat_normalized = np.zeros(displacements.shape)
    As_hat_filtered = np.zeros(displacements.shape)

    for sl in range(max_slices):
        signal = displacements[sl, :]
        if np.max(signal) == np.min(signal):
            signal = 0.5 * np.ones_like(signal)
        else:
            min = np.min(signal)
            max = np.max(signal)

            signal = (signal - min) / (max - min)
        As_hat_normalized[sl, :] = signal
        signal_filtered = savgol_filter(signal, 3, 2)
        signal_filtered = lowess(signal_filtered, np.arange(len(signal_filtered)), frac=0.1)[:, 1]
        As_hat_filtered[sl, :] = min + (max - min) * signal_filtered

    displacements = As_hat_filtered.flatten()

    return displacements



def calculate_displacement(image, bottom, top, shifts,seasonal_adj=False,interp_bad_correl=False):
    # np.save("./log/image_nav.npy",image)
    nb_gating_spokes = image.shape[1]
    nb_slices = image.shape[0]
    npoint_image = image.shape[-1]


    if seasonal_adj:
        from statsmodels.tsa.seasonal import seasonal_decompose

        image_reshaped = image.reshape(-1, npoint_image)
        decomposition = seasonal_decompose(image_reshaped,
                                           model='multiplicative', period=nb_gating_spokes)
        image=image_reshaped/decomposition.seasonal
        image=image.reshape(-1,nb_gating_spokes,npoint_image)
        print(image.shape)

    all_images = image
    if seasonal_adj:
        image_reshaped=image.reshape(-1, npoint_image)
        for ind in range(2, nb_gating_spokes):
            shifted_image = np.concatenate([image_reshaped[ind:], image_reshaped[:ind]],
                                           axis=0).reshape(nb_slices, nb_gating_spokes, -1)
            all_images = np.concatenate([all_images, shifted_image], axis=0)
    ft = np.mean(all_images, axis=0)

    image_nav_for_correl = image.reshape(-1, npoint_image)
    nb_images = image_nav_for_correl.shape[0]
    max_correl=0
    max_correls = []
    mvt = []

    all_correls=[]
    for j in tqdm(range(nb_images)):
        if (j % nb_gating_spokes == 0)or(max_correl<0.5):
            used_shifts=shifts
        else:
            used_shifts=np.arange(mvt[j - 1]-10,(mvt[j - 1]+10)).astype(int)
        # print(used_shifts)
        corrs = np.zeros(len(used_shifts))
        bottom = np.maximum(-used_shifts[0],int(npoint_image/4))
        top = np.minimum(int(npoint_image) -used_shifts[-1],int(3*npoint_image/4))

        for i, shift in enumerate(used_shifts):
            
            corr = np.corrcoef(np.concatenate([ft[j % nb_gating_spokes, bottom:top].reshape(1, -1),
                                               image_nav_for_correl[j, (bottom + shift):(top + shift)].reshape(1, -1)],
                                              axis=0))[0, 1]
            corrs[i] = corr

        J=corrs

        ind_max_J=np.argmax(J)
        current_mvt = used_shifts[ind_max_J]
        max_correl=J[ind_max_J]
        max_correls.append(max_correl)
        mvt.append(current_mvt)
        all_correls.append(corrs)

    
    displacement = np.array(mvt)
    max_correls=np.array(max_correls)

    if interp_bad_correl:
        ind_bad_correl=np.argwhere(max_correls<0.2)
        displacement_new=copy(displacement)
        for i in ind_bad_correl.flatten():
            if ((np.abs(displacement[i]-displacement[i-1])/np.abs(displacement[i])>0.5)and(np.abs(displacement[i]-displacement[i+1])/np.abs(displacement[i])>0.5)):
                displacement_new[i]=np.mean([displacement[i-1],displacement[i+1]])

        displacement=displacement_new


    return displacement

def estimate_weights_bins(displacements, nb_slices, nb_segments, nb_gating_spokes, ntimesteps, radial_traj, nb_bins=5,
                          retained_categories=None, bins=None, equal_spoke_per_bin=False, soft_weight=False):
    if soft_weight:
        alpha = 0.35
        tau = 0.5

    displacement_for_binning = displacements
    max_bin = np.max(displacement_for_binning)
    min_bin = np.min(displacement_for_binning)
    bin_width = (max_bin - min_bin) / nb_bins

    print("Min displacement {}".format(min_bin))
    print("Max displacement {}".format(max_bin))
    displacement_for_binning = displacement_for_binning.flatten()

    if bins is None:
        if not (equal_spoke_per_bin):
            min_std = 1000
            for offset in np.arange(0., bin_width, 0.1):
                nb_bins = 5
                max_bin = np.max(displacement_for_binning)
                min_bin = np.min(displacement_for_binning) + offset

                bin_width = (max_bin - min_bin) / nb_bins
                bins = np.arange(min_bin, max_bin + 0.9 * bin_width, bin_width)
                categories = np.digitize(displacement_for_binning, bins)
                df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T,
                                      columns=["displacement", "cat"])
                df_groups = df_cat.groupby("cat").count()
                min_curr = df_groups.displacement.iloc[:-1].std()
                if min_curr < min_std:
                    bins_final = bins
                    min_std = min_curr

            bins = bins_final
            if retained_categories is None:
                retained_categories = list(range(0, nb_bins + 1))
        else:
            disp_sorted_index = np.argsort(displacement_for_binning)


            count_disp = len(disp_sorted_index)
            disp_width = int(count_disp / nb_bins)

            displacement_for_bins = copy(displacement_for_binning)


            bins = []
            for j in range(1, nb_bins):
                bins.append(np.sort(displacement_for_bins)[j * disp_width])
            if retained_categories is None:
                retained_categories = list(range(0, nb_bins))
    else:
        nb_bins = len(bins) + 1
        if retained_categories is None:
            retained_categories = list(range(0, nb_bins))


    print(bins)

    categories = np.digitize(displacement_for_binning, bins, right=True)
    df_cat = pd.DataFrame(data=np.array([displacement_for_binning, categories]).T, columns=["displacement", "cat"])
    df_groups = df_cat.groupby("cat").count()

    print(categories)
    print(retained_categories)
    print(df_groups)

    groups = []

    for cat in retained_categories:
        groups.append(categories == cat)

    nb_part = nb_slices
    dico_traj_retained = {}
    dico_retained_ts = {}
    print(groups)


    if int(nb_segments / nb_gating_spokes) < (nb_segments / nb_gating_spokes):
        gating_spokes_step = int(nb_segments / (nb_gating_spokes - 1))
    else:
        gating_spokes_step = int(nb_segments / nb_gating_spokes)

    print("Gating spokes step : {}".format(gating_spokes_step))

    spoke_groups_onerep = np.argmin(np.abs(
            np.arange(0, nb_segments, 1).reshape(-1, 1) - np.arange(0, nb_segments, gating_spokes_step).reshape(1, -1
                                                                                                                )),
                                        axis=-1).reshape(1, -1)
    rep_increment = nb_gating_spokes * np.arange(nb_slices).reshape(-1, 1)

    spoke_groups = spoke_groups_onerep + rep_increment
    spoke_groups = spoke_groups.flatten()


    for j, g in tqdm(enumerate(groups)):
        print("######################  BUILDING FULL VOLUME AND MASK FOR GROUP {} ##########################".format(j))
        retained_nav_spokes_index = np.argwhere(g).flatten()
        # print(retained_nav_spokes_index)

        included_spokes = np.array([s in retained_nav_spokes_index for s in spoke_groups])
        included_spokes[::gating_spokes_step] = False
        

        if (j == (len(retained_categories) - 1)) and (soft_weight):
            alpha = -2 * np.log(tau) / (bins[-1] - bins[-2])
            print(alpha)
            print("Using Soft Weights for full inspiration phase")
            retained_nav_spokes_index_prev_bin = np.argwhere(groups[-2]).flatten()
            included_spokes_prev_bin = np.array([s in retained_nav_spokes_index_prev_bin for s in spoke_groups])
            included_spokes_prev_bin[::gating_spokes_step] = False
            disp_reshaped = np.array([displacement_for_binning[i] for i in spoke_groups])
            included_soft_weight = (np.exp(-alpha * np.abs(disp_reshaped - bins[-1])) > tau)
            included_spokes_prev_bin = (included_soft_weight & included_spokes_prev_bin)
            included_spokes_current_bin = included_spokes
            included_spokes = included_spokes | included_spokes_prev_bin
            disp_reshaped = disp_reshaped.reshape(nb_slices, nb_segments)

        if (j == 0) and (soft_weight):
            alpha = -2 * np.log(tau) / (bins[1] - bins[0])
            print(alpha)
            print("Using Soft Weights for full expiration phase")
            retained_nav_spokes_index_prev_bin = np.argwhere(groups[1]).flatten()
            included_spokes_prev_bin = np.array([s in retained_nav_spokes_index_prev_bin for s in spoke_groups])
            included_spokes_prev_bin[::gating_spokes_step] = False
            disp_reshaped = np.array([displacement_for_binning[i] for i in spoke_groups])
            included_soft_weight = (np.exp(-alpha * np.abs(disp_reshaped - bins[0])) > tau)
            included_spokes_prev_bin = (included_soft_weight & included_spokes_prev_bin)
            included_spokes_current_bin = included_spokes
            included_spokes = included_spokes | included_spokes_prev_bin
            disp_reshaped = disp_reshaped.reshape(nb_slices, nb_segments)

        

        weights, retained_timesteps = correct_mvt_kdata_zero_filled(radial_traj, included_spokes, ntimesteps)

        print(weights.shape)

        if (j == (len(retained_categories) - 1)) and (soft_weight):
            disp_reshaped = np.moveaxis(disp_reshaped, -1, 0)
            included_spokes_current_bin = 1 * np.expand_dims(
                included_spokes_current_bin.reshape(weights.shape[2], nb_segments).T, axis=0)
            included_spokes_prev_bin = 1 * np.expand_dims(
                included_spokes_prev_bin.reshape(weights.shape[2], nb_segments).T, axis=0)
            weights = weights * included_spokes_current_bin + weights * included_spokes_prev_bin * np.exp(
                -alpha * np.abs(disp_reshaped - bins[-1]))

        if (j == 0) and (soft_weight):
            disp_reshaped = np.moveaxis(disp_reshaped, -1, 0)
            included_spokes_current_bin = 1 * np.expand_dims(
                included_spokes_current_bin.reshape(weights.shape[2], nb_segments).T, axis=0)
            included_spokes_prev_bin = 1 * np.expand_dims(
                included_spokes_prev_bin.reshape(weights.shape[2], nb_segments).T, axis=0)
            weights = weights * included_spokes_current_bin + weights * included_spokes_prev_bin * np.exp(
                -alpha * np.abs(disp_reshaped - bins[0]))

        dico_traj_retained[j] = weights
        dico_retained_ts[j] = retained_timesteps

    return dico_traj_retained, dico_retained_ts, bins



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filenamenav',type=str)
    parser.add_argument('--ch', type=int)
    parser.add_argument('--segments', nargs='?', const=1400, type=int, default=1400)
    parser.add_argument('--bottom', nargs='?', const=-20, type=int, default=-20)
    parser.add_argument('--top', nargs='?', const=40, type=int, default=40)
    parser.add_argument('--retained', nargs='?', const="0,1,2,3,4", type=str, default="0,1,2,3,4")
    parser.add_argument('--filenamebins', nargs='?', const=None, type=str)
    parser.add_argument('--nbins', nargs='?', const=5, type=int, default=5)
    parser.add_argument('--gatingonly', nargs='?', const=False, type=bool, default=False)
    parser.add_argument('--equalspoke', nargs='?', const=True, type=bool, default=True)
    parser.add_argument('--seasonaladj', nargs='?', const=False, type=bool, default=False)
    parser.add_argument('--hardinterp', nargs='?', const=False, type=bool, default=False)
    parser.add_argument('--interpbadcorrel', nargs='?', const=True, type=bool, default=True)
    parser.add_argument('--softweight', nargs='?', const=True, type=bool, default=True)
    args = parser.parse_args()

    print(args)
    incoherent=False
    pad=10

    filename_nav_save=args.filenamenav
    ch=args.ch
    nb_segments=args.segments
    bottom=args.bottom
    top=args.top
    retained_categories=[int(item) for item in args.retained.split(',')]
    filename_bins=args.filenamebins
    nbins=args.nbins
    gating_only=args.gatingonly
    equal_spoke_per_bin=args.equalspoke
    seasonal_adj=args.seasonaladj
    hard_interp=args.hardinterp
    interp_bad_correl=args.interpbadcorrel
    soft_weight=args.softweight

    calculate_displacement_weights(filename_nav_save,nb_segments,bottom,top,incoherent,ch,filename_bins,retained_categories,nbins,gating_only,pad,equal_spoke_per_bin,seasonal_adj,hard_interp,interp_bad_correl,soft_weight)