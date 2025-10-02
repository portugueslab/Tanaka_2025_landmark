import numpy as np
from matplotlib import pyplot as plt
import colorcet as cc
from scipy.signal import convolve2d
from scipy.stats import wilcoxon
from matplotlib import cm
from matplotlib import colors as cl

class RingAttractor():
    def __init__(self, *args, **kwargs):
        # Prepare attributes to store relevant parameters
        self.param = dict() 
        self.W_fix = []
        self.W_pla = []
        self.resp = []
        self.taus = []
        
        self.init_params(*args, **kwargs)
        self.init_W()
        self.init_resp()
        
    def init_params(self, 
                    n_hd_units = 16,
                    n_vis_units = 16,
                    kappa_hd = 2.0, # width of connectivity
                    kappa_rc = 2.0,
                    rc_hd_offset = np.pi/2, # shift amount
                    w_hd_hd = -1.0, # weight scaling parameter
                    w_hd_rc = -1.0,
                    w_rc_hd = +1.0,
                    w_ahv_rc = +1.0,
                    w_vis_vis=-0.1,
                    max_w_vis_hd=1.0,
                    tau_hd  = 1.0, # time constants
                    tau_rc  = 1.0,
                    tau_ahv = 1.0,
                    tau_vis  = 1.0,
                    u_hd = 1.0, # constitutive input to HD units
                    u_rc = 0.0, # constitutive input to RC units
                    fwhm_vis = 36, # visual cell RF size, in degree
                    alpha=0.1 # weight learning rate
                    ):
        '''
        initilaize the parameters with default values (with an option to inject anything)
        '''
        param_name_list = self.init_params.__code__.co_varnames[1:-2]
        for pn in param_name_list:
            self.param[pn] = locals()[pn]
    
    def init_W(self):
        '''
        Create the initial (pre-learning) connectivity matrix 
        '''
        N = self.param['n_hd_units']
        M = self.param['n_vis_units']
        n_cell = N*3 + 2 + M        
        W = np.zeros((n_cell, n_cell))
        
        # preferred angle of HD cells
        phi = np.arange(self.param['n_hd_units'])/self.param['n_hd_units'] * np.pi * 2

        # build & scale hd cell output 
        hd_out = np.exp(self.param['kappa_hd']*np.cos(phi[None, :] - phi[:, None] + np.pi)) / np.exp(self.param['kappa_hd'])
        W_hd_hd = self.param['w_hd_hd'] * hd_out
        W_hd_rc = self.param['w_hd_rc'] * hd_out
        
        # build & scale rotation cell output
        cw_out =  np.exp(self.param['kappa_rc']*np.cos(phi[None, :] - phi[:, None] - self.param['rc_hd_offset'])) / np.exp(self.param['kappa_rc'])
        ccw_out = np.exp(self.param['kappa_rc']*np.cos(phi[None, :] - phi[:, None] + self.param['rc_hd_offset'])) / np.exp(self.param['kappa_rc'])
        W_cw_hd  = self.param['w_rc_hd'] * cw_out
        W_ccw_hd = self.param['w_rc_hd'] * ccw_out

        # combine
        W[:N, :3*N] = np.hstack((W_hd_hd, W_hd_rc, W_hd_rc))
        W[N:3*N, :N] = np.vstack((W_cw_hd, W_ccw_hd))
        W[3*N,   N:2*N] = self.param['w_ahv_rc']
        W[3*N+1, 2*N:3*N] = self.param['w_ahv_rc']
        W[-M:,-M:] = (1-np.eye(M)) * self.param['w_vis_vis']
        
        self.W_fix = W
        self.W_pla = W*0
        self.reset_W_pla()
        
    def init_resp(self, fix_seed=False):
        '''
        Initialize response as a 1D array with the appropriate length
        Also store tau as an array for convenience
        '''
        if fix_seed:
            np.random.seed(seed=1)
        n_cell = self.W_fix.shape[0]
        self.resp = -np.random.rand(n_cell) # this is the only randomness, I suppose
        self.taus = np.hstack((np.tile(self.param['tau_hd'], self.param['n_hd_units']),
                               np.tile(self.param['tau_rc'], self.param['n_hd_units']*2),
                               np.tile(self.param['tau_ahv'], 2),
                               np.tile(self.param['tau_vis'], self.param['n_vis_units'])))
        
    def reset_W_pla(self, sigma=0):
        N = self.param['n_hd_units']
        M = self.param['n_vis_units']
        self.W_pla *= 0
        self.W_pla[-M:, :N] = np.abs(np.random.normal(size=(M, N)) * sigma)
        
    def run_onestep(self, dt, ahv=0, scene=[]):
        '''
        Move the simulation forward by dt amount.
        '''
        N = self.param['n_hd_units']
        M = self.param['n_vis_units']
        
        # leak term (for all cells)
        leak = -self.resp
        
        # calculate network input to the HD/RC units
        rectified_output = self.rectify_resp()
        network_input = (rectified_output[None, :] @ (self.W_fix + self.W_pla)).flatten()
        constitutive_input = np.hstack((np.tile(self.param['u_hd'], N), np.tile(self.param['u_rc'], N*2),))

        # calculate inputs to the sensory neurons
        ahv_input = self.calc_ahv_input(ahv)
        
        if type(scene)==np.ndarray:
            vis_input = self.calc_vis_input(scene)
        else:
            vis_input = 0
        
        # combine and scale
        dr = leak + network_input
        dr[:N*3] += constitutive_input
        dr[(N*3):(N*3+2)] += ahv_input
        dr[-M:] += vis_input

        
        self.resp += dr / self.taus * dt
        self.update_W(dt)
        
    def rectify_resp(self):
        return self.resp * (self.resp > 0)
        
    def calc_ahv_input(self, ahv):
        '''
        Given a scaler AHV signal, return inputs to CW and CCW AHV cells.
        Some form of nonlinearity should be implemented here to allow gain 1 integration
        '''
        return np.asarray((ahv*(ahv>0), -ahv*(ahv<0)))
    
    def calc_vis_input(self, scene):
        '''
        Given the visual scene, calculate the inputs to the visual cells.
        Consider implementing 2D verison etc.
        '''
        vertical_mean = np.mean(scene, axis=0)

        # convert FWHM to kappa
        hwhm_rad = self.param['fwhm_vis'] / 2 / 180 * np.pi
        kappa = np.log(2) / (1-np.cos(hwhm_rad))
        
        # prepare receptive fields
        peaks = np.arange(self.param['n_vis_units']) / self.param['n_vis_units'] *np.pi * 2.0
        x = np.arange(scene.shape[1]) / scene.shape[1] * np.pi * 2.0
        RFs = np.exp(kappa * np.cos(x[None, :] + peaks[:, None])) / np.exp(kappa)

        # convolve
        dx = 2.0 * np.pi / scene.shape[1]
        out = np.sum(RFs * vertical_mean[None, :], axis=1) * dx

        # rectify
        out = out * (out > 0)
                
        return out
    
    def update_W(self, dt):
        '''
        Implement hebbian-like update of the viusal-to-hd learning
        '''
        pass
        
    def show_W(self, figsize=(6,4)):
        fig, ax = plt.subplots(1,1,figsize=figsize)
        W = self.W_fix + self.W_pla
        vmax = np.max(np.abs(W))
        im = ax.imshow(W, vmax=vmax, vmin=-vmax, cmap=cc.cm.CET_D1)
        cb = plt.colorbar(im, ax=ax, shrink=0.5, location='top')
        cb.set_label('synaptic weight')
        ax.set_xlabel('postsynaptic cells')
        ax.set_ylabel('presynaptic cells')
        
        # prepare ticks
        N = self.param['n_hd_units']
        tick_pos = (0, N, 2*N, 3*N, 3*N+2)
        tick_labels = ('HD', 'rotation (cw)', 'rotation (ccw)', 'ahv', 'visual')
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels, rotation=90)
        
        return fig, ax, cb
        
    def decode(self):
        phi = np.arange(self.param['n_hd_units'])/self.param['n_hd_units'] * np.pi * 2
        pva_x = np.mean(self.resp[:self.param['n_hd_units']] * np.cos(phi))
        pva_y = np.mean(self.resp[:self.param['n_hd_units']] * np.sin(phi))
        pva_theta = np.arctan2(pva_y, pva_x)
        pva_length = np.sqrt(pva_x**2 + pva_y**2)
        return pva_theta, pva_length
    
    def read_resp(self):
        return self.resp.copy()

    
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

def bout_generator(duration, dt, tau, frequency, mu, sigma, tau_sign, fix_seed=False):
    
    if fix_seed:
        np.random.seed(seed=1)
    n_step = int(duration / dt)
    omega = []
    bias = 0
    last_omega=0
    
    for i in range(n_step):
        this_omega = last_omega * np.exp(-dt/tau)
        
        # Draw bout onset
        # Exponentially smoothed delta function with
        # specified AUC
        # exp(-t/tau) integrates to tau (from 0 to +Inf)
        # So initial speed should be AUC / tau
        if np.random.rand() < dt * frequency:
            amp = np.abs(np.random.normal(mu, sigma)) # this is the angle change
            sign = 2*(np.random.rand() > (0.5 + bias)) - 1
            bias += sign 
            this_omega += amp * sign / tau
        omega.append(this_omega)
        last_omega = this_omega
        bias *= np.exp(-dt/tau_sign)
    return np.asarray(omega)

def polar_wrap(x):
    """
    Because polar plot does not automatically close the circle...
    """
    return np.hstack((x, x[0]))