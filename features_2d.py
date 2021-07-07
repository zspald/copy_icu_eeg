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
from features import EEG_FEATS

# List of electrode coordinates to be used for image construction
EEG_X = [-0.4, 0.4, 0.0, -0.32, 0.32, -0.64, 0.64, 0.0, -0.24, 0.24, -0.8, 0.8, -0.64, 0.64,
         0.0, -0.32, 0.32, -0.24, 0.24]

EEG_Y = [0.0, 0.0, 0.0, 0.4, 0.4, 0.48, 0.48, 0.4, 0.76, 0.76, 0.0, 0.0, -0.48, -0.48, -0.48,
         -0.5, -0.5, -0.76, -0.76]

# map indicating channel subtractions based on indices in the channel list at the top of features.py
# (assuming channel order in data is same as order in list)
# BIPOLAR_MAP = {7:5, 5:15, 15:17, 17:10, 7:3, 3:0, 0:12, 12:10, 9:2, 2:14, 8:4, 4:1, 1:13, 13:11, 8:6, 6:16, 16:18, 18:11}
BIPOLAR_MAP = {8: [5, 3], 5: [10],  3: [0], 10: [12], 0: [15], 12: [17], 15: [17], 7: [2], 2: [14], 9: [4, 6], 4: [1],
               6: [11], 1: [16], 11: [13], 16: [18], 13: [18]}
# calculate new coordinates as midpoint of channels being subtracted
BIPOLAR_X = []
BIPOLAR_Y = []
for electrode in BIPOLAR_MAP:
    for linked in BIPOLAR_MAP[electrode]:
        new_x = (EEG_X[electrode] + EEG_X[linked]) / 2
        new_y = (EEG_Y[electrode] + EEG_Y[linked]) / 2
        BIPOLAR_X.append(new_x)
        BIPOLAR_Y.append(new_y)



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
    # Outputs
    #   all_outputs: a multidimensional array of feature maps with shape (N* x F x W x H), where
    #                N* and F share the same definitions as above and W, H denote the image size
    @staticmethod
    def generate_map(patient_id, input_feats, channel_info, bipolar=False):
        # Check whether the set of input features is valid
        if input_feats is None:
            return None
        # determine which montage-associated coordinates to use
        if bipolar:
            x_list = BIPOLAR_X
            y_list = BIPOLAR_Y
            # propagate channel rejections to corresponding bipolar channels
            channel_info = EEGMap.channel_removal_bipolar(channel_info)
            # Initialize HDF file to store the output array
            file = h5py.File('data/%s_data_bipolar.h5' % patient_id, 'a')
        else:
            x_list = EEG_X
            y_list = EEG_Y
            # Initialize HDF file to store the output array
            file = h5py.File('data/%s_data.h5' % patient_id, 'a')
        all_outputs = file.create_dataset('maps', (np.size(input_feats, axis=0), np.size(input_feats, axis=-1), 48, 48)
                                          , compression='gzip', chunks=True)
        # Obtain coordinates for rectangular grid with pre-allocated size
        grid_x, grid_y = np.mgrid[-1:1:48j, -1:1:48j]
        x_zero, y_zero, zeros = EEGMap.zero_coordinates(grid_x, grid_y, rad=0.85)
        # Iterate over all samples
        for ii in range(input_feats.shape[0]):
            channels_to_zero = channel_info[ii]
            # Iterate over all features
            for jj in range(input_feats.shape[-1]):
                # Check whether channel contains NaN features
                x_coords = np.array([x for idx, x in enumerate(x_list) if channels_to_zero[idx] == 0])
                y_coords = np.array([y for idx, y in enumerate(y_list) if channels_to_zero[idx] == 0])
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
    #   sample_idx: index of the sample to be visualized
    # Outputs
    #   returns a plot of sample distribution of the designated feature over the scalp
    @staticmethod
    def visualize_map(input_maps, feat_idx, sample_idx=None, bipolar=False):
        if sample_idx is None:
            sample_idx = np.random.randint(0, input_maps.shape[0])
        feat_map = input_maps[sample_idx, feat_idx, :, :]
        plt.imshow(feat_map, extent=(-1, 1, -1, 1), origin='lower', cmap='jet')
        if bipolar:
            plt.scatter(BIPOLAR_X, BIPOLAR_Y, c='k')
        else:
            plt.scatter(EEG_X, EEG_Y, c='k')
        if bipolar:
            plt.title('Bipolar Interpolated Map - %s' % EEG_FEATS[feat_idx])
        else:
            plt.title('Referential Interpolated Map - %s' % EEG_FEATS[feat_idx])
        plt.show()
        return

    @staticmethod
    def channel_removal_bipolar(channel_info):
        # create array to store channel removal info for bipolar montage
        bipolar_channel_info = np.zeros((channel_info.shape[0], sum([len(val) for _, val in BIPOLAR_MAP.items()])))
        # print(channel_info.shape)
        # print(bipolar_channel_info.shape)

        # map linking referential montage channels to affected bipolar montage channels to propagate removal
        # see features.py for ordering/indices of channels in referential and bipolar montages
        # In referential: 0 = C3, 1 = C4, 2 = Cz, 3 = F3, 4 = F4, 5 = F7, 6 = F8, 7 = Fz, 8 = Fp1, 9 = Fp2,
        # 10 = T3, 11 = T4, 12 = T5, 13 = T6, 14 = Pz, 15 = P3, 16 = P4, 17 = O1, 18 = O2,
        # In bipolar: 0 = Fp1-F7, 1 = Fp1-F3, 2 = F7-T3, 3 = F3-C3, 4 = T3-T5, 5 = C3-P3, 6 = T5-O1,   
        # 7 = P3-O1, 8 = Fz-Cz, 9 = Cz-Pz, 10 = Fp2-F4, 11 = Fp2-F8, 12 = F4-C4, 13 = F8-T4, 14 = C4-P4, 
        # 15 = T4-T6, 16 = P4-O2, 17 = T6-O2 
        bipolar_affected_channels_map = {0: [3, 5], 1: [12, 14], 2: [8, 9], 3: [1, 3], 4: [10, 12], 5: [0, 2], 
                                        6: [11, 13], 7: [8], 8: [0, 1], 9: [10, 11], 10: [2, 4], 11: [13, 15], 
                                        12: [4, 6], 13: [15, 17], 14: [9], 15: [5, 7], 16: [14, 16], 17: [6, 7], 
                                        18: [16, 17]}

        # iterate through all accepted segments
        for i in range(channel_info.shape[0]):
            channel_list = channel_info[i, :]
            # iterate through list of channels statuses
            for idx, chan in enumerate(channel_list):
                # if channel is flagged for removal
                if chan == 1:
                    # propagate channel removal to affected bipolar montage channels using map above
                    # (i.e. if C3 should be removed in referential, F3-C3 and C3-P3 should be removed in bipolar)
                    for bi_chans in bipolar_affected_channels_map[idx]:
                        bipolar_channel_info[i, bi_chans] = 1

        print("Channel removals propagated to bipolar montage")
        return bipolar_channel_info


