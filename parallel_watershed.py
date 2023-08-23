# Import a range of python libraries used in this notebook:
import iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import iris.plot as iplt
import iris.quickplot as qplt
import shutil
import datetime
from six.moves import urllib
from pathlib import Path
###########################################################
# Import tobac itself:
import tobac
print('using tobac version', str(tobac.__version__))
###########################################################
# Disable a few warnings:
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)
warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)
warnings.filterwarnings('ignore', category=FutureWarning, append=True)
warnings.filterwarnings('ignore',category=pd.io.pytables.PerformanceWarning)
###########################################################
data_out=Path('../')
###########################################################
# Download the data: This only has to be done once for all tobac examples and can take a while
data_file = list(data_out.rglob('data/Example_input_OLR_model.nc'))
if len(data_file) == 0:
    file_path='https://zenodo.org/record/3195910/files/climate-processes/tobac_example_data-v1.0.1.zip'
    #file_path='http://zenodo..'
    tempfile=Path('temp.zip')
    print('start downloading data')
    request=urllib.request.urlretrieve(file_path, tempfile)
    print('start extracting data')
    shutil.unpack_archive(tempfile, data_out)
    tempfile.unlink()
    print('data extracted')
    data_file = list(data_out.rglob('data/Example_input_OLR_model.nc'))
###########################################################
#Load Data from downloaded file:
OLR=iris.load_cube(str(data_file[0]),'OLR')
###########################################################
#Set up directory to save output and plots:
savedir=Path("Save")
if not savedir.is_dir():
    savedir.mkdir()
plot_dir=Path("../plots")
if not plot_dir.is_dir():
    plot_dir.mkdir()
###########################################################
# Determine temporal and spatial sampling:
dxy,dt=tobac.get_spacings(OLR)
###########################################################
# Dictionary containing keyword arguments for feature detection step (Keywords could also be given directly in the function call).
parameters_features={}
parameters_features['position_threshold']='weighted_diff'
parameters_features['sigma_threshold']=0.5
parameters_features['n_min_threshold']=4
parameters_features['target']='minimum'
parameters_features['threshold']=[250,225,200,175,150]

# Dictionary containing keyword options for the segmentation step:
parameters_segmentation={}
parameters_segmentation['target']='minimum'
parameters_segmentation['method']='watershed'
parameters_segmentation['threshold']=250
###########################################################
da = xr.DataArray.from_iris(OLR)

parallelism_degree = 2
start_counter = 0
end_counter = int((da.south_north.size / parallelism_degree))
_step = end_counter
masks_list = []
features_list = []
flag = False

features = tobac.feature_detection_multithreshold(OLR, dxy, **parameters_features)
frame_section_cube_list = []

for i in range(parallelism_degree):
    frame_section = da.isel(south_north=slice(start_counter, end_counter))
    feature_section = features.loc[(features["hdim_1"] >= start_counter) & (features["hdim_1"] < end_counter)]
    if flag:
        feature_section.loc[:, "hdim_1"] -= _step
    frame_section_cube = frame_section.to_iris()
    frame_section_cube_list.append(frame_section_cube)
    start_counter += _step
    end_counter += _step

    if features is not None:
        mask, features_mask = tobac.segmentation_2D(feature_section, frame_section_cube, dxy, **parameters_segmentation)
        mask_array = xr.DataArray.from_iris(mask)
        features_list.append(feature_section)
        firsts = []
        lasts = []
        for a in range(mask_array.T.time.size):
            test = mask_array.isel(time=a).to_pandas()
            if flag:
                first = [x for x in list(map(set, test.head(n=1).values))[0]]
                first.remove(0)
                if first:
                    firsts.append(first)
            if i != (parallelism_degree - 1):
                last = [x for x in list(map(set, test.tail(n=1).values))[0]]
                last.remove(0)
                if last:
                    lasts.append(last)
        firsts = list(set([elem for sublist in firsts for elem in sublist]))
        lasts = list(set([elem for sublist in lasts for elem in sublist]))
        features_first = feature_section.loc[feature_section["feature"].isin(firsts)]
        features_last = feature_section.loc[feature_section["feature"].isin(lasts)]
        #if flag:
        #    features_last.loc[:, "hdim_1"] -= _step
        #features_first.loc[:, "hdim_1"] += _step
        if i != 0:
            mask, features_mask2 = tobac.segmentation_2D(features_first, frame_section_cube_list[i-1], dxy, **parameters_segmentation)
            mask_array2 = xr.DataArray.from_iris(mask)
            masks_list.append(mask_array2)

        masks_list.append(mask_array)
    else:
        zero_array = frame_section.where(frame_section == 0, other=0)
        masks_list.append(zero_array)

    frame_array = xr.DataArray.from_iris(frame_section_cube)
    flag = True

Mask_OLR = xr.concat(masks_list, dim="south_north")
Mask_OLR = Mask_OLR.to_iris()
Features = pd.concat(features_list)
print('feature detection performed and saved')
###########################################################
# Arguments for trajectory linking:
parameters_linking={}
parameters_linking['v_max']=20
parameters_linking['stubs']=2
parameters_linking['order']=1
parameters_linking['extrapolate']=0
parameters_linking['memory']=0
parameters_linking['adaptive_stop']=0.2
parameters_linking['adaptive_step']=0.95
parameters_linking['subnetwork_size']=100
parameters_linking['method_linking']= 'predict'
###########################################################
# Perform linking and save results to file:
Track=tobac.linking_trackpy(features, OLR, dt=dt, dxy=dxy, **parameters_linking)
Track.to_hdf(savedir / 'Track.h5', 'table')
###########################################################
# Set extent of maps created in the following cells:
axis_extent = [-95, -89, 28, 32]
###########################################################
# Create animation of tracked clouds and outlines with OLR as a background field
animation_test_tobac=tobac.animation_mask_field(Track,features,OLR,Mask_OLR,
                                          axis_extent=axis_extent,#figsize=figsize,orientation_colorbar='horizontal',pad_colorbar=0.2,
                                          vmin=80,vmax=330,
                                          plot_outline=True,plot_marker=True,marker_track='x',plot_number=True,plot_features=True)
###########################################################
# Save animation to file:
savefile_animation = savedir /'olr_tracking_model_animation.mp4'
animation_test_tobac.save(savefile_animation,dpi=200)
print(f'animation saved to {savefile_animation}')
