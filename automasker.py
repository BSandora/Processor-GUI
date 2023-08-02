"""
Mask and analyze quot-track trajectories.
TO-DO:
- handle multiple masks
- plot number of trajs in each mask
"""
# Filepaths
import os
from glob import glob


# image reader
from quot.read import ImageReader

# Filtering, thresholding, refining masks
from scipy.ndimage import gaussian_filter
from skimage.filters.thresholding import threshold_isodata
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.filters import try_all_threshold


# Mask calculations
from quot.helper import get_edges, get_ordered_mask_points
from quot.gui.masker import apply_masks
from matplotlib.path import Path

# Arrays, DataFrames
import numpy as np
import pandas as pd

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize 

# Progress bar
from tqdm import tqdm


def make_apply_mask(image_path: str,
                    csv_path: str, 
                    assignment_mode: str='all_points',
                    in_mask_out_dir: str=None,
                    show_plot: bool=False, 
                    out_fig: str=None,
                    out_csv: str=None) -> pd.DataFrame:
    """
    Auto-generate and apply a mask to a trajs.csv file, modifying 
    it with a mask_column and saving files with trajectories only 
    within masks or outside masks.
    
    Inputs:
    -------
    image_path      :   str, path to image or movie to use to mask
    csv_path        :   str, path to trajs.csv file to mask
    assignment_mode :   str, how to assign trajectories to masks
    mask_dir        :   str, a folder to save the masked trajectories 
                        in for convenience. If None, saves to 
                        same directory as trajs.csv file.
    show_plot       :   bool, if True, show the results of masking.
    out_fig         :   str, path to save mask outputs. If None, 
                        no Figure is made.
    out_csv         :   str, path to save mask details. If None,
                        no CSV is saved.

    Outputs:
    --------
    return          :   pd.DataFrame of mask vertices 
                        and details about mask
    show            :   show plot, if desired
    write           :   overwrite CSV at csv_path provided with
                        "mask_index" column. Additionally, save
                        a CSV file with trajectories inside a mask and 
                        a CSV file with trajectories outside all masks.
    """
    assert assignment_mode in ["by_localization", "single_point", "all_points"], \
        "assignment_mode must be 'by_localization', 'single_point', or 'all_points'."

    # Read in image or movie, max-intensity project and blur
    image = ImageReader(image_path)
    proj = image.get_frame(0)
    gauss_blur = gaussian_filter(proj, sigma=1.5)

    # Mask, remove small bits, fill holes
    mask = gauss_blur > threshold_isodata(gauss_blur)
    removed = remove_small_objects(mask, 3000)
    filled = remove_small_holes(removed, 500)

    # Get coordinates for polygon bounding this refined mask
    mask_coords = get_ordered_mask_points(get_edges(filled))

    # Mask the DF and save to the same path
    tracks = pd.read_csv(csv_path)
    tracks['mask_index'] = apply_masks([mask_coords], tracks, mode=assignment_mode)
    tracks.to_csv(csv_path, index=False)

    # Save trajectories outside mask
    out_csv_outside = f"{os.path.splitext(csv_path)[0]}_outside_mask.csv"
    tracks[tracks['mask_index'] == 0].to_csv(out_csv_outside, index=False)

    # Save trajectories inside mask to user-provided folder
    if in_mask_out_dir is not None:
        if os.path.isdir(in_mask_out_dir):
            out_csv_inside = os.path.join(in_mask_out_dir, 
                f"{os.path.splitext(os.path.basename(csv_path))[0]}.csv")
        else:
            print(f"Out directory for masked CSVs is not valid, \
                {in_mask_out_dir} passed. Saving to {os.path.dirname(csv_path)} instead.")
            out_csv_inside = f"{os.path.splitext(csv_path)[0]}.csv"
    else:
        out_csv_inside = f"{os.path.splitext(csv_path)[0]}.csv"
    tracks[tracks['mask_index'] > 0].to_csv(out_csv_inside, index=False)

    # Make a mask summary DF
    if out_csv is not None:
        # Calculate mean total intensity, which is the sum intensity of 
        # a frame within the mask averaged over all the frames a movie.
        sum_int = image.get_frame(0).astype(float)
        sum_int *= filled
        sum_int[sum_int == 0] = np.nan
        mean_sum_intensity = np.nanmean(sum_int)

        df = pd.DataFrame(index=range(mask_coords.shape[0]),
                          columns=["filename", "mask_index", "y", "x", 
                                   "vertex", "area", "mean_sum_intensity",
                                   "sum_intensity"])
        df['filename'] = image_path        
        c = 0
        for mask_index, arr in enumerate([mask_coords]):
            l = arr.shape[0]
            df.loc[c:c+l-1, "mask_index"] = mask_index 
            df.loc[c:c+l-1, "y"] = arr[:,0]
            df.loc[c:c+l-1, "x"] = arr[:,1]
            df.loc[c:c+l-1, "vertex"] = np.arange(l)
            df.loc[c:c+l-1, "area"] = filled.sum()
            df.loc[c:c+l-1, "mean_sum_intensity"] = mean_sum_intensity
            df.loc[c:c+l-1, "sum_intensity"] = mean_sum_intensity * filled.sum()
            c += l 
        df.to_csv(out_csv, index=False)

    # Make summary masking figure, if desired
    if (show_plot) or (out_fig is not None):
        _, ax = plt.subplots(3, 3)

        # Mask processing steps
        ax[0,0].imshow(image.get_frame(0), cmap=plt.cm.gray)
        ax[0,0].set_title("1. Still frame", fontsize='small')
        ax[0,1].imshow(proj, cmap=plt.cm.gray)
        ax[0,1].set_title("2. Max-intensity projection", fontsize='small')
        ax[0,2].imshow(gauss_blur, cmap=plt.cm.gray)
        ax[0,2].set_title("3. Gaussian blur", fontsize='small')
        ax[1,0].imshow(mask, cmap=plt.cm.gray)
        ax[1,0].set_title("4. Naive mask", fontsize='small')
        ax[1,1].imshow(removed, cmap=plt.cm.gray)
        ax[1,1].set_title("5. Pruned mask", fontsize='small')
        ax[1,2].imshow(filled, cmap=plt.cm.gray)
        ax[1,2].set_title("6. Filled mask", fontsize='small')
        
        ####################################
        ## MASKING PLOTS STOLEN FROM QUOT ##
        ####################################
        # Only consider points that do not have the error flag set
        tracks = tracks[tracks["error_flag"] == 0.0].copy()

        # Estimate the size of the ROI
        y_max = int(np.ceil(tracks["y"].max()))
        x_max = int(np.ceil(tracks["x"].max()))

        # Generate coordinates for each pixel
        Y, X = np.indices((y_max, x_max))
        YX = np.asarray([Y.ravel(), X.ravel()]).T 

        # Generate an image where each pixel is assigned to a mask
        mask_im = np.zeros((y_max, x_max), dtype=np.int64)
        for i, point_set in enumerate([mask_coords]):
            path = Path(point_set, closed=True)
            mask_im[path.contains_points(YX).reshape((y_max, x_max))] = i+1

        # Generate localization density
        y_bins = np.arange(y_max+1)
        x_bins = np.arange(x_max+1)
        H, _, _ = np.histogram2d(tracks['y'], tracks['x'], bins=(y_bins, x_bins))
        H = gaussian_filter(H, 5.0)

        # The set of points to use for the scatter plot
        L = np.asarray(tracks[["y", "x", 'mask_index']])

        # Categorize each localization as either (1) assigned or (2) not assigned
        # to a mask
        inside = L[:,2] > 0
        outside = ~inside 

        # Localization density in the vicinity of each spot
        yx_int = L[:,:2].astype(np.int64)
        densities = H[yx_int[:,0], yx_int[:,1]]
        norm = Normalize(vmin=0, vmax=densities.max())

        ax[2,0].imshow(mask_im, cmap='gray')
        ax[2,1].imshow(H, cmap='gray')
        ax[2,2].scatter(
            L[inside, 1],
            y_max-L[inside, 0],
            c=densities[inside],
            cmap="viridis",
            norm=norm,
            s=1
        )
        ax[2,2].scatter(
            L[outside, 1],
            y_max-L[outside, 0],
            cmap="magma",
            c=densities[outside],
            norm=norm,
            s=1
        )
        ax[2,2].set_xlim((0, x_max))
        ax[2,2].set_ylim((0, y_max))
        ax[2,2].set_aspect('equal')

        # Subplot labels
        ax[2,0].set_title("Applied mask", fontsize='small')
        ax[2,1].set_title("Total localization density", fontsize='small')
        ax[2,2].set_title("Inside/outside mask", fontsize='small')
        
        ax = ax.ravel()
        for a in ax:
            a.axis('off')

        # Save plots with editable text in Adobe
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42
        plt.savefig(out_fig, dpi=800, transparent=True)
        
        if show_plot:
            plt.show()

        plt.close()

    # Return binary mask
    return filled


def runMasker (fileNames):
    # Get directory of TIFs to use to mask
    parentDir = os.path.split(fileNames[0])[0]
    overallDir = os.path.split(parentDir)[0]
    mask_tif_dir = os.path.join(overallDir, 'snaps3')
    if not os.path.isdir(mask_tif_dir) and os.name == 'posix':
        if os.name == 'posix':
            mask_tif_dir = mask_tif_dir.replace("\\ ", " ")
    
    # Sanity check
    tif_files = glob(os.path.join(mask_tif_dir, "*.tif*"))
    assert len(tif_files) > 0, f"No TIF/TIFF files found in {mask_tif_dir}!"

    # Get directory where trajs.csv files are, sanity check
    trajs_dir = parentDir
    if not os.path.isdir(trajs_dir) and os.name == 'posix':
        if os.name == 'posix':
            trajs_dir = trajs_dir.replace("\\ ", " ")
    assert len(glob(os.path.join(trajs_dir, "*.csv"))) > 0, \
        f"No CSV files found in {trajs_dir}!"

    # Get directory to save in_mask trajs
    out_dir = os.path.join(overallDir, 'masked')
    if not os.path.isdir(out_dir) and os.name == 'posix':
        if os.name == 'posix':
            out_dir = out_dir.replace("\\ ", " ")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    for tif in tqdm(tif_files):
        # Make and apply masks
        f = os.path.splitext(os.path.basename(tif))[0]
        csv_path = glob(os.path.join(trajs_dir, f + "_trajs.csv"))
        if len(csv_path) != 1 or csv_path[0] not in fileNames:
            continue
        out_fig = os.path.splitext(tif)[0] + "_masked.png"
        out_csv = os.path.splitext(tif)[0] + "_masks.csv"
        bin_mask = make_apply_mask(
                tif, 
                *csv_path, 
                in_mask_out_dir=out_dir,
                out_fig=out_fig,
                out_csv=out_csv)
        
