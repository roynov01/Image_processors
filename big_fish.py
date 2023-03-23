# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 15:49:58 2022

@author: royno
"""
# https://github.com/fish-quant/big-fish-examples
# https://big-fish.readthedocs.io/en/stable/stack/io.html
import os
import numpy as np
import bigfish
import bigfish.stack as stack
import bigfish.multistack as multistack
import bigfish.plot as plot
import pandas as pd


path = r"C:\Users\royno\Desktop\py\bigfish\test"

path_input = path + '/data/input'
path_output = path + '/data/output'


SINGLE_STACK = 10



# stack.check_input_data(path_input, input_segmentation=True)

# import images
nuc = stack.read_image(os.path.join(path_input, "ex1_dapi_fov1.tif"))
rna = stack.read_image(os.path.join(path_input, "ex1_fish_fov1.tif"))
gfp = stack.read_image(os.path.join(path_input, "ex1_gfp_fov1.tif"))
rna_stacked = stack.read_image(os.path.join(path_input, "ex1_fish_fov1_temp_max.tif"))

# convert to 2D
rna_stacked_2d = rna_stacked[0]
nuc_2d = nuc[SINGLE_STACK]
rna_2d = rna[SINGLE_STACK]
gfp_2d = gfp[SINGLE_STACK]


# change brightness/contrasts
rna_2d_stretched = stack.rescale(rna_2d, channel_to_stretch=0)
nuc_2d_stretched = stack.rescale(nuc_2d, channel_to_stretch=0)

# plot image
plot.plot_images(rna_2d, framesize=(5, 5), contrast=True)
plot.plot_images([rna_2d_stretched, nuc_2d_stretched], titles=["RNA","DAPI"])
plot.plot_images([rna_stacked_2d,rna_2d_stretched], titles=["RNA_stacked","RNA"], contrast=True)


#%% filters 

DAPI_TRESHOLD = 920

nuc_2d_mean = stack.mean_filter(nuc_2d, kernel_shape="square", kernel_size=30)
nuc_2d_median = stack.median_filter(nuc_2d, kernel_shape="square", kernel_size=30)
nuc_2d_min = stack.minimum_filter(nuc_2d, kernel_shape="square", kernel_size=30)
nuc_2d_max = stack.maximum_filter(nuc_2d, kernel_shape="square", kernel_size=30)
nuc_2d_gaussian = stack.gaussian_filter(nuc_2d, sigma=5)

images = [nuc_2d, nuc_2d_mean, nuc_2d_median, nuc_2d_min, nuc_2d_max, nuc_2d_gaussian]
titles = ["original image", "mean filter", "median filter", "minimum filter", "maximum filter", "gaussian filter"]
plot.plot_images(images, rescale=True, titles=titles)

nuc_bool = nuc_2d > DAPI_TRESHOLD
# np.max(nuc_2d,axis=1) # fust to get an idea of values
nuc_dilated = stack.dilation_filter(nuc_bool, kernel_shape="square", kernel_size=30)
nuc_eroded = stack.erosion_filter(nuc_bool, kernel_shape="square", kernel_size=30)
images = [nuc_bool, nuc_dilated, nuc_eroded]
titles = ["masked image", "binary dilation", "binary erosion"]
plot.plot_images(images, rescale=True, titles=titles)

rna_log = stack.log_filter(rna_2d, sigma=3)
rna_background_mean = stack.remove_background_mean(rna_stacked_2d, kernel_shape="square", kernel_size=31)
rna_background_gaussian = stack.remove_background_gaussian(rna_stacked_2d, sigma=3)
images = [rna_log, rna_background_gaussian, rna_background_mean]
titles = ["LoG filter", "remove gaussian background", "remove mean background"]
plot.plot_images(images, contrast=True, titles=titles)



#%% projections
nuc_max = stack.maximum_projection(nuc)
nuc_mean = stack.mean_projection(nuc, return_float=False)
nuc_median = stack.median_projection(nuc)
images = [nuc_max, nuc_mean, nuc_median, nuc_2d]
titles = ["maximum projection", "mean projection", "median projection", "original"]
plot.plot_images(images, rescale=True, titles=titles)

#%% focus
rna_mip = stack.maximum_projection(rna) # max projection

# blurred RNA (we simulate some z-slices out-of-focus)
# rna_blurred = rna.copy()
# rna_blurred[-4:, ...] = stack.gaussian_filter(rna_blurred[-4:, ...], sigma=5) * 1.5
# rna_blurred_mip = stack.maximum_projection(rna_blurred)
# images = [rna_mip, rna_blurred_mip]
# titles = ["FISH channel (MIP)", "blurred FISH channel (MIP)"]
# plot.plot_images(images, titles=titles, framesize=(10, 5), contrast=True)

# check which Z stacks are out focus - takes a lot of time!

# focus = stack.compute_focus(rna, neighborhood_size=31)
# measures = [focus.mean(axis=(1, 2))]
# plot.plot_sharpness(measures, labels=["original"], title="Sharpness measure over smFISH channel")
# can remove them:
# nb_to_keep = rna.shape[0] - 4 # remove 4 stacks from the end

# in_focus_image = stack.in_focus_selection(rna, focus, proportion=nb_to_keep)
# in_focus_image_mip = stack.maximum_projection(in_focus_image)
# images = [rna_mip, in_focus_image_mip]
# titles = ["FISH channel (MIP)", "FISH channel (in-focus selection + MIP)"]
# plot.plot_images(images, titles=titles, framesize=(10, 5), contrast=True)

#%% segmentation

import bigfish.segmentation as segmentation

# nuc = stack.read_image(os.path.join(path_input, "example_nuc_full.tif"))
# cell = stack.read_image(os.path.join(path_input, "example_cell_full.tif"))

nuc = nuc_2d
cell = rna_stacked_2d

# by Thresholding
nuc_mask = segmentation.thresholding(nuc, threshold=DAPI_TRESHOLD)
nuc_mask = segmentation.clean_segmentation(nuc_mask, small_object_size=2000, fill_holes=True)
nuc_label = segmentation.label_instances(nuc_mask)
plot.plot_segmentation(nuc, nuc_label, rescale=True,title="mask")

# by U-net based model 
model_nuc = segmentation.unet_3_classes_nuc()
nuc_label = segmentation.apply_unet_3_classes(model_nuc, nuc, target_size=256, test_time_augmentation=True)
plot.plot_segmentation(nuc, nuc_label, rescale=True,title="U-net based model")


#%% Cells segmentation
cell_label = segmentation.cell_watershed(cell, nuc_label, threshold=500, alpha=0.9)
plot.plot_segmentation_boundary(cell, cell_label, nuc_label, contrast=True, boundary_size=4)

# or, step by step:
TRESHOLD = 650
    
watershed_relief = segmentation.get_watershed_relief(cell, nuc_label, alpha=0.9)
cell_mask = segmentation.thresholding(cell, threshold=TRESHOLD)
cell_mask[nuc_label > 0] = True
cell_mask = segmentation.clean_segmentation(cell_mask, small_object_size=5000, 
                                            fill_holes=True)
cell_label = segmentation.apply_watershed(watershed_relief, nuc_label, cell_mask)
plot.plot_images([watershed_relief, cell_mask, cell_label], 
                 titles=["Watershed relief", "Binary mask", "Labelled cells"])
 
# U-net based model


model_cell = segmentation.unet_distance_edge_double()
# instance segmentation
cell_label = segmentation.apply_unet_distance_double(
    model_cell, 
    nuc=nuc, 
    cell=cell, 
    nuc_label=nuc_label, 
    target_size=256, test_time_augmentation=True)
plot.plot_segmentation(cell, cell_label, rescale=True)

plot.plot_segmentation_boundary(cell, cell_label, nuc_label, contrast=True, boundary_size=4,title="non post processed", framesize=(5,5))
# post processing segmentation
nuc_label = segmentation.clean_segmentation(nuc_label, delimit_instance=True)
cell_label = segmentation.clean_segmentation(cell_label, smoothness=7, delimit_instance=True)
nuc_label, cell_label = multistack.match_nuc_cell(nuc_label, cell_label, single_nuc=False, cell_alone=True)
plot.plot_images([nuc_label, cell_label], titles=["Labelled nuclei", "Labelled cells"])

plot.plot_segmentation_boundary(nuc_2d, cell_label, nuc_label, contrast=True, boundary_size=3, title="with post processed")

#%% save labels
stack.save_image(nuc_label, os.path.join(path_output, "nuc_label.tif"))
stack.save_image(cell_label, os.path.join(path_output, "cell_label.tif"))

#%% spot detection

import bigfish.detection as detection


# rna = stack.read_image(os.path.join(path_input, "experiment_1_smfish_fov_1.tif"))
rna = stack.read_image(os.path.join(path_input, "ex1_fish_fov1.tif"))

rna_mip = stack.maximum_projection(rna)
plot.plot_images(rna_mip, framesize=(5, 5), contrast=True)



PIXEL_SIZE = 115 #nm/pixel
PIXEL_SIZE_Z = 300 # nm/Z stack


# spot radius
VOXEL = (PIXEL_SIZE_Z, PIXEL_SIZE, PIXEL_SIZE)
RADIUS = (3*PIXEL_SIZE_Z, 3*PIXEL_SIZE, 3*PIXEL_SIZE)
# strict: VOXEL=(150, 50, 50), radius=(350, 150, 150)
# medium: VOXEL=(300, 103, 103), radius=(160, 120, 120)
# non-strict: size=(300, 103, 103), radius=(350, 150, 150)
spot_radius_px = detection.get_object_radius_pixel(
    voxel_size_nm=VOXEL, 
    object_radius_nm=RADIUS, 
    ndim=3)



# LoG filter
rna_log = stack.log_filter(rna, sigma=spot_radius_px)

# local maximum detection
mask = detection.local_maximum_detection(rna_log, min_distance=spot_radius_px)

# thresholding
threshold = detection.automated_threshold_setting(rna_log, mask)
spots, _ = detection.spots_thresholding(rna_log, mask, threshold)
plot.plot_detection(rna_mip, spots, contrast=True)

plot.plot_elbow(images=rna, voxel_size=VOXEL, spot_radius=RADIUS)


# for breaking dense spots into individual:
spots_post_decomposition, dense_regions, reference_spot = detection.decompose_dense(
    image=rna, 
    spots=spots, 
    voxel_size=VOXEL, 
    spot_radius=RADIUS, 
    alpha=0.7,  # alpha impacts the number of spots per candidate region
    beta=1,  # beta impacts the number of candidate regions to decompose
    gamma=5)  # gamma the filtering step to denoise the image
print("detected spots before decomposition: {spots.shape}")
print(f"detected spots after decomposition: {spots_post_decomposition.shape}")

plot.plot_detection(rna_mip, spots_post_decomposition, contrast=True)



#%% finding clusters of dots:
CLUSTER_RADIUS = PIXEL_SIZE*3
    
spots_post_clustering, clusters = detection.detect_clusters(
    spots=spots_post_decomposition, 
    voxel_size=VOXEL, 
    radius=CLUSTER_RADIUS, 
    nb_min_spots=6)
print(f"detected spots after clustering: {spots_post_clustering.shape[0]}")
print(f"detected clusters: {clusters.shape[0]}")

plot.plot_detection(rna_mip, 
                    spots=[spots_post_decomposition, clusters[:, :3]], 
                    shape=["circle", "polygon"], 
                    radius=[3, 6], 
                    color=["red", "blue"],
                    linewidth=[1, 2], 
                    fill=[False, True], 
                    contrast=True)

#%% save spots and clusters

path = os.path.join(path_output, "spots.npy")
stack.save_array(spots_post_clustering, path)
path = os.path.join(path_output, "clusters.npy")
stack.save_array(clusters, path)


# path = os.path.join(path_output, "spots.csv")
# np.savetxt(path,spots_post_clustering,delimiter = ",",fmt="%-1i")
# path = os.path.join(path_output, "clusters.csv")
# np.savetxt(path,clusters,delimiter = ",",fmt="%-1i")

path = os.path.join(path_output, "spots.csv")
stack.save_data_to_csv(spots_post_clustering, path)
path = os.path.join(path_output, "clusters.csv")
stack.save_data_to_csv(clusters, path)

#%% extract cell level results:

# nuc = stack.read_image(os.path.join(path_input, "ex1_dapi_fov1.tif"))
nuc_mip = stack.maximum_projection(nuc)
spots = spots_post_clustering
# RNA in nucleus - foci or transcription sites
image_contrasted = stack.rescale(rna, channel_to_stretch=0)
image_contrasted = stack.maximum_projection(image_contrasted)

spots_no_ts, foci, ts = multistack.remove_transcription_site(spots, clusters, nuc_label, ndim=3)
spots_in, spots_out = multistack.identify_objects_in_region(nuc_label, spots, ndim=3)
print(f"detected spots: {spots.shape[0]}")
print(f"detected spots (without transcription sites):{spots_no_ts.shape[0]}")
print(f"detected spots (inside nuclei): {spots_in.shape[0]}")
print(f"detected spots (outside nuclei): {spots_out.shape[0]}")

#%% summarize data per cell

fov_results = multistack.extract_cell(
    cell_label=cell_label, 
    ndim=3, 
    nuc_label=nuc_label, 
    rna_coord=spots_no_ts, 
    others_coord={"foci": foci, "transcription_site": ts},
    image=image_contrasted,
    others_image={"dapi": nuc_mip, "smfish": rna_mip})

print("number of cells identified: {0}".format(len(fov_results)))

for i, cell_results in enumerate(fov_results[64:67]):
    print("cell {0}".format(i))
    
    # get cell results
    cell_mask = cell_results["cell_mask"]
    cell_coord = cell_results["cell_coord"]
    nuc_mask = cell_results["nuc_mask"]
    nuc_coord = cell_results["nuc_coord"]
    rna_coord = cell_results["rna_coord"]
    foci_coord = cell_results["foci"]
    ts_coord = cell_results["transcription_site"]
    image_contrasted1 = cell_results["image"]
    print("\r number of rna {0}".format(len(rna_coord)))
    print("\r number of foci {0}".format(len(foci_coord)))
    print("\r number of transcription sites {0}".format(len(ts_coord)))

    # plot.plot_cell(
    #     ndim=3, cell_coord=cell_coord, nuc_coord=nuc_coord, 
    #     rna_coord=rna_coord, foci_coord=foci_coord, other_coord=ts_coord, 
    #     image=cell_results["smfish"], cell_mask=cell_mask, nuc_mask=nuc_mask, contrast=True,
    #     title="Cell {0}".format(i),show=True)
    # plot.plot_cell(
    #     ndim=3, cell_coord=cell_coord, nuc_coord=nuc_coord, 
    #     rna_coord=rna_coord, foci_coord=foci_coord, other_coord=ts_coord, 
    #     image=cell_results["dapi"], cell_mask=cell_mask, nuc_mask=nuc_mask, contrast=True,
    #     title="Cell {0}".format(i),show=True)
    
    

plot.plot_cell_coordinates(
    ndim=3,
    cell_coord=[cell["cell_coord"] for cell in fov_results[0:12]],
    nuc_coord=[cell["nuc_coord"] for cell in fov_results[0:12]],
    rna_coord=[cell["rna_coord"] for cell in fov_results[0:12]])
    # titles=[f'cell {i}' for i in range(len(fov_results[0:12]))])
    



summary = multistack.summarize_extraction_results(fov_results, ndim=3)
summary.to_csv(path_or_buf=os.path.join(path_output, "summary.csv"))

# or save individual cells
for i, cell_results in enumerate(fov_results):
    path = os.path.join(path_output, "results_cell_{0}.npz".format(i))
    stack.save_cell_extracted(cell_results, path)

filenames = [None] * len(fov_results)
for i, cell_results in enumerate(fov_results):
    path = os.path.join(path_output, "results_cell_{0}.npz".format(i))
    stack.save_cell_extracted(cell_results, path)
    filenames[i] = path

#%% downstream analysis
dataframes = [None] * len(filenames)
for filename in filenames:
    # load single cell data
    data = stack.read_cell_extracted(filename)
    cell_mask = data["cell_mask"]
    nuc_mask = data["nuc_mask"]
    rna_coord = data["rna_coord"]
    foci_coord = data["foci"]
    smfish = data["smfish"]
    
    # compute features
    features, features_names = classification.compute_features(
    cell_mask, nuc_mask, ndim=3, rna_coord=rna_coord,
    smfish=smfish, voxel_size_yx=103,
    foci_coord=foci_coord,
    centrosome_coord=None,
    compute_distance=True,
    compute_intranuclear=True,
    compute_protrusion=True,
    compute_dispersion=True,
    compute_topography=True,
    compute_foci=True,
    compute_area=True,
    return_names=True)

    # build dataframe
    features = features.reshape((1, -1))
    df_cell = pd.DataFrame(data=features, columns=features_names)
    dataframes.append(df_cell)
df = pd.concat(dataframes)
df.reset_index(drop=True, inplace=True)
path = os.path.join(path_output, "df_features.csv")
df.to_csv(path, sep=',')
# stack.save_data_to_csv(df, path)

#%%
# cellpose, stardist / stardist3d