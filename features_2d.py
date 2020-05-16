#######################################################################################
# Computes two-dimensional representations of statistical EEG features by interpolating
# the feature values over the electrode positions within the scalp
# Written by Daniel Joongwon Kim
# University of Pennsylvania, Department of Computer and Information Science
#######################################################################################

# Import libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from features import EEGFeatures, EEG_FEATS

# List of electrode coordinates to be used for image construction
EEG_X = [-0.4, 0.4, 0.0, -0.32, 0.32, -0.64, 0.64, 0.0, -0.24, 0.24, -0.8, 0.8, -0.64, 0.64,
         0.0, -0.32, 0.32, -0.24, 0.24]

EEG_Y = [0.0, 0.0, 0.0, 0.4, 0.4, 0.48, 0.48, 0.4, 0.76, 0.76, 0.0, 0.0, -0.48, -0.48, -0.48,
         -0.5, -0.5, -0.76, -0.76]


# A class that allows the user to obtain images of feature distribution across the brain surface
class EEGMap:

    # Generates a map of EEG feature distribution over the brain surface
    # Inputs
    #   patient_id: ID of the patient
    #   input_feats: an array of EEG features with shape N* x C x F, where N* is the number of
    #                clean EEG segments from the patient's dataset, C is the number of channels
    #                and F is the number of features
    #   channel_info: an array of binary values with shape N* x C with the same description.
    #                 Contains 0's if the channel is good and 1's if the channel is bad
    #   normalize: whether to normalize the feature inputs for map generation
    # Outputs
    #   all_outputs: a multidimensional array of feature maps with shape (N* x F x W x H), where
    #                N* and F share the same definitions as above and W, H denote the image size
    @staticmethod
    def generate_map(patient_id, input_feats, channel_info, normalize=False):
        # Initialize HDF file to store the output array
        file = h5py.File('data/%s_data.h5' % patient_id, 'a')
        all_outputs = file.create_dataset('maps', (np.size(input_feats, axis=0), np.size(input_feats, axis=-1), 48, 48)
                                          , compression='gzip', chunks=True)
        # Obtain coordinates for rectangular grid with pre-allocated size
        grid_x, grid_y = np.mgrid[-1:1:48j, -1:1:48j]
        x_zero, y_zero, zeros = EEGMap.zero_coordinates(grid_x, grid_y, rad=0.85)
        # Normalize features globally if indicated so
        if normalize:
            input_feats = EEGFeatures.normalize_feats(input_feats, option='default')
        # Iterate over all samples
        for ii in range(np.size(input_feats, axis=0)):
            channels_to_zero = channel_info[ii]
            # Iterate over all features
            for jj in range(np.size(input_feats, axis=-1)):
                # Check whether channel contains NaN features
                x_coords = np.array([x for idx, x in enumerate(EEG_X) if channels_to_zero[idx] == 0])
                y_coords = np.array([y for idx, y in enumerate(EEG_Y) if channels_to_zero[idx] == 0])
                map_coords = np.c_[np.r_[x_coords, x_zero], np.r_[y_coords, y_zero]]
                # Build inputs for EEG map
                map_inputs = np.array([feat for idx, feat in enumerate(input_feats[ii, :, jj])
                                       if channels_to_zero[idx] == 0])
                map_inputs = np.r_[np.abs(map_inputs), zeros]
                # Perform linear interpolation upon the map inputs
                map_outputs = griddata(map_coords, map_inputs, (grid_x, grid_y), method='linear')
                # Return map-generated outputs for the EEG sample
                all_outputs[ii, jj, :, :] = map_outputs
        return all_outputs

    # Returns a list of coordinates outside the scalp region to be zeroed for interpolation.
    # Note that x_coords and y_coords are synchronized in order
    # Inputs
    #   grid_x: list of x-coordinates from the input grid
    #   grid_y: list of y-coordinates from the input grid
    #   rad: the radius of the circular scalp region
    # Outputs
    #   x_coords: list of all x-coordinates to be zeroed
    #   y_coords: list of all y-coordinates to be zeroed
    #   zeros: list of zeros for all given coordinates
    @staticmethod
    def zero_coordinates(grid_x, grid_y, rad):
        x_coords, y_coords = list(), list()
        # Iterate over every combination of grid values for x and y axes
        for x in grid_x[:, 0]:
            for y in grid_y[0, :]:
                # Add the coordinate iff its distance exceeds the radius from the origin.
                if np.sqrt((x**2 + y**2)) > rad:
                    x_coords.append(x)
                    y_coords.append(y)
        # Create a list of zeros to be used for suppression outside
        zeros = np.zeros(len(x_coords))
        return x_coords, y_coords, zeros

    # Visualizes the feature distribution across the brain surface
    # Inputs
    #   input_maps: a multidimensional array of feature maps with shape (N* x F x W x H), where
    #               N* and F share the same definitions as above and W, H denote the image size
    #   feat_idx: index of the feature to be visualized, as given in EEG_FEATS in features.py
    # Outputs
    #   returns a plot of sample distribution of the designated feature over the scalp
    @staticmethod
    def visualize_map(input_maps, feat_idx):
        sample_idx = np.random.randint(0, np.size(input_maps, axis=0))
        feat_map = input_maps[sample_idx][feat_idx]
        plt.imshow(feat_map, extent=(-1, 1, -1, 1), origin='lower', cmap='jet')
        plt.title('Interpolated Map - %s' % EEG_FEATS[feat_idx])
        plt.show()
        return
