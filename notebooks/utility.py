import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors as cl
from pathlib import Path
import flammkuchen as fl
from bouter import EmbeddedExperiment
from scipy.stats import binned_statistic, ranksums, wilcoxon
from scipy.signal import convolve2d
from scipy.optimize import curve_fit
import re
import json
from tqdm import tqdm

"""
To do: clean up un-used functions at the end!
"""


def config_rcparams():
    """
    Configure matplotlib so that the SVG outputs are in the actual size on the paper
    """
    plt.rcParams['svg.fonttype'] = 'none' # assume fonts are installed and do not convert text to path in svg
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.style'] = 'normal'
    plt.rcParams['font.size'] = 6
    
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['axes.labelsize'] = 6
    plt.rcParams['xtick.labelsize'] = 5
    plt.rcParams['ytick.labelsize'] = 5
    
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams["legend.frameon"] = False

def wrapped(theta, offset=np.pi, thresh=5):
    """ 
    Wrap radian time traces with specified offsets, while filling the 
    wrapping point with nan. For visualization purpose only!
    """
    wrapped = (theta+offset)%(2*np.pi) - np.pi
    wrapped[np.where(np.abs(np.diff(wrapped))>thresh)[0]] = np.nan
    return wrapped

def scaled_imshow(image, ax, levelrange=(1,99), *args, **kwargs):
    """ 
    Do imshow with scaling specified with percentiles. 
    """
    vmin = np.percentile(image, levelrange[0])
    vmax = np.percentile(image, levelrange[1])
    ax.imshow(image, vmin=vmin, vmax=vmax, *args, **kwargs)
    
def get_fish_list(protocol_path, silent=False):
    """ 
    Given the path to the protocol folder, display and the return the list of
    all fish directories under it. Assumes that the fish directories are named like
    20230101_f0.
    """
    master_path = Path(protocol_path)
    files = list(master_path.glob('*f*'))
    if not silent:
        for i, file in enumerate(files):
            print(i, file)
    return files

def nest_fish_list_per_fish(fish_list):
    """ 
    Make the fish list nested 
    """
    # first, cut out fish name from paths
    template = r'202\d\d\d\d\d_f\d'
    fish_name_list = [re.search(template, f.name)[0] for f in fish_list]
    # next, count all unique fish name
    unique_fish_name_list = []
    for fish_name in fish_name_list:
        if not fish_name in unique_fish_name_list:
            unique_fish_name_list.append(fish_name)
    
    nested_path_list = [[] for i in range(len(unique_fish_name_list))]
    for path in fish_list:
        for i, unique_fish_name in enumerate(unique_fish_name_list):
            if unique_fish_name in path.name:
                nested_path_list[i].append(path)
    return nested_path_list
        
    

def load_data(fish_path, cells_only=False):
    """ 
    Given the path to the fish directory, load and return 2p and behavioral data.
    Set cells_only to True to use s2p cell clasification (which I think misses too much stuff).
    """

    if cells_only:
        fname = '*data_from_suite2p_cellsonly.h5'
    else:
        fname = '*data_from_suite2p_allrois.h5'
    h5_path = list(fish_path.rglob(fname))[0] # assume there is only one
    
    s2p_data = fl.load(h5_path) # preprocessed data from suite2p    
    t_i = fl.load(fish_path / 'time.h5')['t'] # timestamps created by sidewinder
    t_i = t_i[:np.argmax(t_i)] # some experiments did not stop as intended because of a sidewinder bug

    # behavioral data
    exp = EmbeddedExperiment(fish_path / "behavior")
    
    return s2p_data, t_i, exp


def load_mask(rec_path, mask_name):
    mask_list = list(rec_path.glob('mask_'+mask_name+'*'))
    k = 0
    if len(mask_list) < 1: # no mask = skip
        print('No mask of the specified name was found!')
        return []
    if len(mask_list) > 1:
        print('Multiple masks of the specified name was found!')
        for i, found_mask in enumerate(mask_list):
            print(i, found_mask)
        k = int(input('Tell me which one to use! : '))
    with open(mask_list[k], 'r') as f:
        mask = json.load(f)
    return mask

def load_lsm_data(fish_path):
    """ Given the path to the fish directory, load and return 2p and behavioral data.
    Set cells_only to True to use s2p cell clasification (which I think misses too much stuff).
    """
    h5_path = fish_path / "data_from_suite2p_cellsonly.h5"  
    s2p_data = fl.load(h5_path) # preprocessed data from suite2p    

    # behavioral data
    exp = EmbeddedExperiment(fish_path / "behavior")
    
    return s2p_data, exp

def downsample_traces(t0, y, t1, func='mean'):
    """ Downsample time trace y(t) defined on time vector t0 to time vector t1.
    t0 and t1 are assumed to be both roughly evenly spaced, and t1 is coarcer than t0.
    """
    dt1 = np.nanmedian(np.diff(t1))
    t1bin = np.zeros(len(t1)+1)
    t1bin[:-1] = t1 - dt1/2
    t1bin[-1] = t1[-1] + dt1/2
    dsy = binned_statistic(t0, y, bins=t1bin, statistic=func)[0]
    return dsy


def calc_snip_correlation(trace, t, t_start, duration):
    """ Cut out N trace snippets starting at t_start with duration,
    calculate all pairwise correlations, and average them. Trace is 
    assumed to have ROI x time dimensions.
    """
    n_roi = trace.shape[0]
    n_rep = len(t_start)
    dt = np.nanmedian(np.diff(t))
    n_frames = int(duration / dt)
    
    snip = []
    for this_t_start in t_start:
        this_start_ind = np.argmax(t > this_t_start)
        snip.append(trace[:, this_start_ind:(this_start_ind + n_frames)])
    snip = np.asarray(snip)
    
    mean_corr = []
    for i in range(n_roi):
        C = np.corrcoef(snip[:, i, :])
        mean_corr.append(np.mean(C[np.triu(np.ones(n_rep),1)==1]))
    return np.asarray(mean_corr)
    
def scaled_shifted_cos(x, a, b, c):
    return a * np.cos(x - b) + c

def fit_transform_sinusoid(X, theta, b_range=(-np.pi,np.pi)):
    """ Fit a*cos(theta-b)+c to each row of X, and
    return (a,b,c), a*cos(theta-b)+c, and R2.
    """
    n_roi = X.shape[0]
    fit_param = np.zeros((n_roi, 3))
    Xhat = np.zeros(X.shape)
    
    # go through each raw, do curve fitting
    for i in range(n_roi):
        x = X[i ,:]
        not_nan = np.isnan(x)==0
        fit_param[i, :] = curve_fit(scaled_shifted_cos, theta[not_nan], x[not_nan], bounds=([0,b_range[0],-np.inf], [np.inf,b_range[1],np.inf]))[0]
        Xhat[i, :] = scaled_shifted_cos(theta, fit_param[i, 0], fit_param[i, 1], fit_param[i, 2])
        
    # calculate R2 (1-SSresidual/SStotal)
    SS_res = np.nansum((X - Xhat)**2, axis=1)
    SS_tot = np.nansum((X - np.nanmean(X, axis=1)[:, None])**2, axis=1)
    R2 = 1 - SS_res/SS_tot

    return fit_param, Xhat, R2
    
def calc_binned_pva(X, phi, n_bin=8):
    """ Given a N x T response matrix X, preferred angles phi, chunk phi into n_bin bins,
    calculate binned average, and creates population vector average based on it.
    """
    # phi is supposed to span from -pi to pi, but doubly make sure
    phi = (phi+np.pi) % (2*np.pi) - np.pi
    # convert phi into indices (= binning)
    phi_ind = ((phi + np.pi) / np.pi / 2 * n_bin).astype(int) # integer casting implicitly floor
    bin_counts = [np.sum(phi_ind==i) for i in range(n_bin)]
    # calculate vector for each bin (span -pi to pi)
    bin_center_angle = (np.arange(n_bin) + 0.5) / n_bin * 2.0 * np.pi - np.pi
    
    # Loop through the bins, calculate average within each bin and 
    # calculate PVA
    X_binned = np.empty((n_bin, X.shape[1]))
    X_binned[:,:] = np.nan
    PVA = np.zeros((2, X.shape[1]))
    for i in range(n_bin):
        if any(phi_ind==i):
            x = np.mean(X[phi_ind==i, :], axis=0)
            X_binned[i, :] = x
            PVA[0, :] += x * np.cos(bin_center_angle[i]) 
            PVA[1, :] += x * np.sin(bin_center_angle[i]) 
    PVA /= len(np.unique(phi_ind))
    
    theta = np.arctan2(PVA[1,:], PVA[0,:]) # arctan2(y, x) order
    amplitude = np.sqrt(np.sum(PVA**2, axis=0))
    
    return theta, amplitude, X_binned, phi_ind, bin_counts

def cut_snip(X, t, t_start, t_pre, t_post, subtract_pre=True):
    """ Given a data matrix X whose last dimension is time and a time vector t,
    cut snippets of data around t_start with pre/post lengths of t_pre/t_post.
    """
    snip = []
    dt = np.nanmedian(np.diff(t))
    n_sample_pre = int(np.floor(t_pre/dt))
    n_sample_post = int(np.ceil(t_post/dt))

    for this_t in t_start:
        i = np.argmax(t > this_t)
        if i-n_sample_pre>=0 and i+n_sample_pre<len(t):
            if subtract_pre:
                this_slice = X[..., (i-n_sample_pre):(i+n_sample_post+1)] - np.nanmean(X[..., (i-n_sample_pre):i], axis=-1)[..., None] 
            else:
                this_slice = X[..., (i-n_sample_pre):(i+n_sample_post+1)]
                
            snip.append(this_slice)
    snip = np.asarray(snip)
    
    t_snip = np.arange(n_sample_pre + n_sample_post + 1) * dt - t_pre
    
    return snip, t_snip
        
def scalar_to_color(x, cmap='coolwarm', vmax=1, vmin=np.nan):
    if np.isnan(vmin):
        vmin = -vmax
    x = (x-vmin)/(vmax-vmin)
    return cm.get_cmap(cmap)(x)[:3]

def scatter_correlation(x, y, ax, *args, **kwargs):
    C = np.corrcoef(x, y)
    ax.scatter(x, y, *args, **kwargs)
    ax.set_title('r = {0:0.2}'.format(C[0,1]))
    return C[0,1]
    
def von_Mises(theta, mu, kappa):
    I0 = np.sum(np.exp(kappa * np.cos(np.linspace(-np.pi,np.pi,1000))) * np.pi * 2 /1000)
    return np.exp(kappa * np.cos(theta - mu)) / I0

def set_pi_ticks(ax, vmin, vmax, spacing=1.0, is_x=True):
    vmin = int(vmin)
    vmax = int(vmax)
    tick_pos = np.arange(vmin, vmax, spacing) * np.pi
    tick_name = []
    for i in np.arange(vmin, vmax, spacing):
        tick_name.append("{}$\pi$".format(i))
    if is_x:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_name)
    else:
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_name)
        
def chuncked_correlation(t, x, y, t_start, d_pre, d_post):
    """ Given time traces x and y defined on t, calculate correation within [T-d_pre, T+d_post],
    where T is each element of t_start tuple.
    """
    out = []
    for T in t_start:
        is_this_chunck = (t>T-d_pre) * (t<T+d_post)
        C = np.corrcoef(x[is_this_chunck], y[is_this_chunck])
        out.append(C[0,1])
    return out

def chuncked_regression(t, x, y, t_start, d_pre, d_post):
    """ Given time traces x and y defined on t, do linear regression within [T-d_pre, T+d_post],
    where T is each element of t_start tuple.
    """
    intercept = []
    slope = []
    for T in t_start:
        is_this_chunck = (t>T-d_pre) * (t<T+d_post)
        n_dp = np.sum(is_this_chunck)
        regressor = np.vstack( (np.ones(n_dp), x[is_this_chunck])  ).T
        b0, b1 = np.linalg.lstsq(regressor, y[is_this_chunck], rcond=None)[0]
        intercept.append(b0)
        slope.append(b1)
    return intercept, slope
        
def plot_indiv_mean(ax, x, y, color=(0,0,0), *args, **kwargs):
    ax.plot(x, y.T, alpha=0.2, color=color, *args, **kwargs)
    ax.plot(x, np.nanmean(y, axis=0), linewidth=2, color=color, *args, **kwargs)
    
def plot_x_shade(ax, x0, x1, color=(0.2,0.2,0.2), alpha=0.2):
    ylim = ax.get_ylim()
    ax.fill_between([x0, x1],[ylim[0], ylim[0]],[ylim[1],ylim[1]], color=color, alpha=alpha, zorder=-10)
    
def angular_diff(t0, t1):
    return (t0 - t1 + np.pi) % (2 * np.pi) - np.pi

def create_rgb_filled_roi_mask(roi_mask, values, index=[], roi_index_start=1, colormap=cm.viridis, vmin=[], vmax=[]):
    # ROI mask retruned by suite2p is 1-indexed (0 is the "ground")
    # if index is explicitly provided, we will only draw those
    n_values = len(values)
    # if index is not provided, we need to make one
    if not np.any(index):
        index = np.arange(n_values) + roi_index_start
    
    # scaling
    if not vmin:
        vmin = np.min(values)
    if not vmax:
        vmax = np.max(values)
    # scale values from 0 to 255
    values = (values - vmin) / (vmax - vmin)   
    rgba_map = np.zeros((roi_mask.shape[0], roi_mask.shape[1], 4))
    for i, v in zip(index, values):
        rgba_map += (roi_mask==i)[:, :, None] * np.asarray(colormap(v))[None, None, :]
    return rgba_map
            
def scatter_bar(data, ax=[], color=cm.viridis, condnames=[], connect=False, **kwargs):
    # if axis is not explicitly provided, we use gcf
    if not ax:
        ax = plt.gca()
    
    # data can be 2d array (matched) or a list of 1d arrays
    islist = isinstance(data, list) or isinstance(data, tuple)
    isarray = isinstance(data, np.ndarray)
    if islist==isarray:
        print('wrong type')
        return
    
    if isarray and len(data.shape)!=2:
        print('wrong dimensions')
        return

    #  calculate mean and sem
    if islist:
        ncond = len(data)
        means = [np.mean(y) for y in data]
        sems = [np.std(y)/np.sqrt(len(y)) for y in data]
    else:
        ncond = data.shape[0]
        means = np.mean(data, axis=1)
        sems = np.std(data, axis=1) / np.sqrt(data.shape[1])

    # prepare color
    if isinstance(color, cl.Colormap): # colormap provided
        color_list = [color(x) for x in np.linspace(0, 1, ncond, endpoint=True)]
    else: # otherwise assume this is already a list of colors
        color_list = color
    
    x = np.arange(ncond)
    
    # bar plot
    ax.bar(x, means, color=color_list, alpha=0.5, zorder=0)
            
    # errorbars & individual data points
    all_jittered_x = []
    for i in range(ncond):
        y = data[i]
        n_sample = len(y)
        jittered_x = i + (np.random.rand(n_sample)-0.5) * 0.25
        all_jittered_x.append(jittered_x)
        ax.scatter(jittered_x, y, color=color_list[i % len(color_list)], zorder=3, **kwargs)
        ax.plot((i,i), (means[i]-sems[i], means[i]+sems[i]), 'k-', lw=3, zorder=5)
        
    # connecting dots (ignored for lists = unpaired data)
    if connect and isarray:
        all_jittered_x = np.asarray(all_jittered_x)
        for i in range(data.shape[1]):
            ax.plot(all_jittered_x[:, i], data[:, i], color=(0.5,0.5,0.5), alpha=0.5, zorder=2)
        
    if condnames:
        ax.set_xticks(x)
        ax.set_xticklabels(condnames)
        
    # statistics
    pval = np.zeros((ncond,ncond))
    for i in range(ncond):
            for j in range(ncond):
                if i!=j:
                    if islist:
                        _, pval[i, j] = ranksums(data[i], data[j])
                    else:
                        _, pval[i, j] = wilcoxon(data[i], data[j])
    np.set_printoptions(precision=3)
    return pval
        
def toggle_spines(ax, top, bottom, right, left):
    

    ax.spines.top.set_visible(top)
    ax.spines.bottom.set_visible(bottom)
    ax.spines.right.set_visible(right)
    ax.spines.left.set_visible(left)
    if not bottom:
        ax.set_xticks([])
    if not left:
        ax.set_yticks([])
    return 0
        

def remove_HPD_nosie(traces, noise_thresh=1000):
    
    d_traces = np.diff(traces, axis=1)
    noise_frame = (d_traces[:, :-1]>noise_thresh) * (d_traces[:, 1:]< -noise_thresh) 
    denoised = np.copy(traces)
    for i in tqdm(range(traces.shape[0])):
        for j in np.where(noise_frame[0, :])[0]:
            denoised[i, j+1] = (traces[i, j] + traces[i, j+2]) / 2
    
    return denoised

def split_time(t, t_start, duration, is_train=True):
    """
    Split time traces into training set and test set
    For now, always use the second half as a training set
    """
    if not is_train:
        out = (t>t_start)*(t<t_start+duration/2)
    else:
        out = (t>t_start+duration/2)*(t<t_start+duration)
    return out

def flatten_list_of_array(loa):
    out = []
    for a in loa:
        out.extend(list(a))
    return np.asarray(out)

def color_getter(ind):
    # I am writing this after compiling all the figures
    # so this is not really used
    # but if I have to re-generate the figures in the future
    # I can use this function to more easily unify colors
    # across figures (datasets)
    colors = (
        (1, 0.2, 0.6),   # 0 data (sun-and-bars) magenta
        (0.6,0.6,0.6),   # 1 control grey
        (0.9,0.5,0.2),   # 2 jump orange
        (0.2,0.5,0.9),   # 3 noise blue
        (0.2, 0.5, 0.2), # 4 "both" green
        (0.9, 0.7, 0.2), # 5 pre yellow 
        (0.8, 0.3, 0.6), # 6 learning pink
        (0.4, 0.7, 0.7), # 7 post green
        (0.6, 0.2, 0.65), # 8 treatment group purple
        (0.6, 0.2, 0.6),  # bottom purple
        (0.4, 0.8, 0.4)   # top green
    )
    colors = np.vstack(colors)
    return colors[ind, :]

def generate_sun_and_bars(wallpaper_shape, sun_elevation, sun_radius, bar_width, cylinder_gaze_angle=60):
    """
    Taken from E0084_v04
    """
    azimuth_mat, height_mat = np.meshgrid(np.arange(wallpaper_shape[1]), np.linspace(-1, 1, wallpaper_shape[0]))
    # convert azimuthal index to degrees, and center at 0
    azimuth_mat = azimuth_mat / wallpaper_shape[1] * 360 - 180
    # inverse transform height on the cylinder wall to gaze angle
    elevation_mat = np.arctan(height_mat * np.tan(cylinder_gaze_angle / 180 * np.pi)) / np.pi * 180
    # draw sun and bar
    sun_distance_mat = np.sqrt(azimuth_mat ** 2 + (elevation_mat - sun_elevation) ** 2)
    sun_gradient_mat = (sun_radius - sun_distance_mat) / sun_radius
    sun_gradient_mat = sun_gradient_mat * (sun_gradient_mat > 0)
    left_bar_mask = (azimuth_mat > -90 - bar_width * 0.5) * (azimuth_mat < -90 + bar_width * 0.5)
    right_bar_mask = (azimuth_mat > 90 - bar_width * 1.5) * (azimuth_mat < 90 + bar_width * 1.5) * \
                     ((azimuth_mat < 90 - bar_width * 0.5) + (azimuth_mat > 90 + bar_width * 0.5))
    bar_mask = left_bar_mask + right_bar_mask
    scene_mat = (sun_gradient_mat * (~bar_mask))
    return scene_mat

def generate_stonehenge(wallpaper_shape, cylinder_gaze_angle=60):
    azimuth_mat, height_mat = np.meshgrid(np.arange(wallpaper_shape[1]), np.linspace(-1, 1, wallpaper_shape[0]))
    # convert azimuthal index to degrees, and center at 0
    azimuth_mat = azimuth_mat / wallpaper_shape[1] * 360 - 180
    # inverse transform height on the cylinder wall to gaze angle
    elevation_mat = np.arctan(height_mat * np.tan(cylinder_gaze_angle / 180 * np.pi)) / np.pi * 180

    # draw bars
    bar_mask1 = (azimuth_mat > -127.5) * (azimuth_mat < -112.5)
    bar_mask2 = (azimuth_mat > -97.5) * (azimuth_mat < -82.5)
    bar_mask3 = (azimuth_mat > -7.5) * (azimuth_mat < 7.5)
    bar_mask4 = (azimuth_mat > 127.5) * (azimuth_mat < 142.5) * ((elevation_mat % 20) < 10)

    # combine them (boolean addition)
    temp = (bar_mask1 + bar_mask2 + bar_mask3 + bar_mask4).astype(float)

    return temp

def generate_noise(wallpaper_shape, smoothing_size=5):
    np.random.seed(seed=1)
    noise_mat = 1 - 2*np.random.rand(wallpaper_shape[0], wallpaper_shape[1])
    noise_mat = convolve2d(noise_mat, np.ones((smoothing_size, smoothing_size))/smoothing_size**2, mode='same')
    noise_mat = (noise_mat > 0).astype(float)
    return noise_mat

def generate_suns(wallpaper_shape, radius=60, elevation=35, mirror=False, cylinder_gaze_angle=60):
    azimuth_mat, height_mat = np.meshgrid(np.arange(wallpaper_shape[1]), np.linspace(-1, 1, wallpaper_shape[0]))
    # convert azimuthal index to degrees, and center at 0
    azimuth_mat = azimuth_mat / wallpaper_shape[1] * 360 - 180
    # inverse transform height on the cylinder wall to gaze angle
    elevation_mat = np.arctan(height_mat * np.tan(cylinder_gaze_angle / 180 * np.pi)) / np.pi * 180
    R = np.sqrt((azimuth_mat+90)**2 + (elevation_mat+35)**2)
    R = (radius - R) / radius # peak is 1
    sun_mat = R * (R>0) # cut off at 0 and scale
    if mirror:
        sun_mat[:,wallpaper_shape[1]//2:] = np.fliplr(sun_mat[:,:wallpaper_shape[1]//2])
    return sun_mat