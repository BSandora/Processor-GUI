import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg') 
import matplotlib.gridspec as grd
import time
from matplotlib.widgets import RangeSlider
from matplotlib.patches import Rectangle

import os
from glob import glob
from tifffile import imread
from tqdm import tqdm
from typing import Tuple
import numpy as np

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QWidget

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

def display_cell_prompt(images: list[Tuple[str, str]], 
                        rois: Tuple=None, 
                        vmax=65536, 
                        pbar=True) -> dict:
    
    # Output dictionary
    labeled = dict(keep=[], cull=[])
    resolved = 0

    # Check that ROIs are correct shape before using
    if rois is not None:
        if not (rois.shape[1] == 4 and rois.shape[0] == len(images)):
            print(f"ROIs must be of shape (n_images, 4). {rois.shape} passed.")
            rois = None

    # Progress bar if requested
    if pbar:
        iterations = enumerate(tqdm(images))
    else:
        iterations = enumerate(images)
    
    figArray = []
    target = -1
    # Loop over all tuples
    for i, pair in iterations:
        # Sanity checks, cull if there's an error
        if len(pair) != 2:
            print(f"Problem with passed images {pair}: non-length-2 tuple passed; skipping...")
            labeled['cull'].append(pair[0])
            continue
        if not (os.path.isfile(pair[0]) and os.path.isfile(pair[1])):
            print(f"Problem with passed images {pair}: invalid filepath; skipping...")
            labeled['cull'].append(pair[0])
            continue

        # Figure layout
        fig = plt.figure(constrained_layout=True)
        figArray.append(fig)
        plt.ion()
        gs = grd.GridSpec(4, 2, figure=fig, height_ratios=(6, 2, 1, 1))
        axs = [fig.add_subplot(ax) \
               for ax in [gs[0,0], gs[0,1], gs[1,:], gs[2,:], gs[3,0], gs[3,1]]]
        
        # Kill all ticks for image Axes
        for ax in [axs[0], axs[1]]:
            ax.set_yticks([])
            ax.set_xticks([])
        
        # Kill y-ticks for LUT and slider
        axs[2].set_yticks([])
        axs[3].set_yticks([])

        # Show images
        im = imread(pair[1])
        im_zoomed = imread(pair[0])
        image = axs[0].imshow(im, vmin=0, vmax=vmax/4)
        zoomed = axs[1].imshow(im_zoomed, vmin=0, vmax=vmax/4)

        # Add ROI square, if provided and valid
        if rois is not None:
            axs[0].add_patch(Rectangle(xy=(rois[i, 0], rois[i, 1]),
                                       width=rois[i, 2] - rois[i, 0],
                                       height=rois[i, 3] - rois[i, 1],
                                       color='r', fill=False))

        # Show LUT and slider
        slider = RangeSlider(axs[3], label=None, valmin=0, valmax=vmax, valinit=(0, vmax/4))
        slider.valtext.set_visible(False)
        axs[2].hist(im.flatten(), bins='auto')
        lower_limit_line = axs[2].axvline(slider.val[0], color='k')
        upper_limit_line = axs[2].axvline(slider.val[1], color='k')
        axs[2].set_xlim((0, vmax))

        # Button functions
        def keep_button_clicked(event):
            labeled['keep'].append(pair[0])
            resolve()

        def cull_button_clicked(event):
            labeled['cull'].append(pair[0])
            resolve()
            
        def resolve():
            nonlocal resolved, target
            resolved += 1
            print("kept")
            plt.close()
            print(target)
            if target > 0:
                target -= 1
                figArray[target].canvas.show()
            loop.quit()


        # Slider function
        def slider_update(val):
            # The val passed to a callback by the RangeSlider will
            # be a tuple of (min, max)

            # Update the image's colormap
            for im in [image, zoomed]:
                im.norm.vmin = val[0]
                im.norm.vmax = val[1]
            
            # Update vertical lines on LUT
            lower_limit_line.set_xdata([val[0], val[0]])
            upper_limit_line.set_xdata([val[1], val[1]])

            # Redraw the figure to ensure it updates
            print("slid")

        slider.on_changed(slider_update)

        # Create buttons and define functionality
        keep_button = plt.Button(ax=axs[4], label='Keep', color='green')
        keep_button.on_clicked(keep_button_clicked)
        cull_button = plt.Button(ax=axs[5], label='Cull', color='red')
        cull_button.on_clicked(cull_button_clicked)
        
        # Show plot. Button click closes.
        plt.show()
        fig.canvas.show()
        target += 1
        loop = QtCore.QEventLoop()
        loop.exec()

    
       
    return labeled


def run_cell_culler (parent_dir: str, selectedFiles: list[int]):
    # Get parent folder, replacing escaped space if on Unix
    parent_folder = parent_dir
    if not os.path.isdir(parent_folder) and os.name == 'posix':
        parent_folder = parent_folder.replace("\\ ", " ")
    
    # Sanity checks and filepaths
    assert os.path.isdir(parent_folder), f"{parent_folder} is not a folder!"
    snaps2_dir = os.path.join(parent_folder, "snaps2")
    snaps3_dir = os.path.join(parent_folder, "snaps3")
    spt_folder = os.path.join(parent_folder, "spt")
    rois = np.loadtxt(os.path.join(parent_folder, "rois.txt"), delimiter=',')
    zoomed = glob(os.path.join(snaps3_dir, "*.tif*"))
    assert len(zoomed) > 0, "No TIFs found in snaps3 folder."
    assert len(glob(os.path.join(spt_folder, "*.nd2"))) > 0, \
        "No ND2 files found in spt folder."
    
    # Sort zoomed by integer, not as strings
    oldZoomed = sorted(zoomed, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    zoomed = []
    newRois = np.empty((len(selectedFiles), 4))
    counter = 0
    for i in range(len(oldZoomed)):
        if (i+1 in selectedFiles):
            zoomed.append(oldZoomed[i])
            newRois[counter] = rois[i]
            counter += 1
            
            
    # Make culled folder if it doesn't exist
    cull_folder = os.path.join(spt_folder, "culled")
    if not os.path.exists(cull_folder):
        os.mkdir(cull_folder)
    
    # Pair up zoomed and snaps
    zoomed_pairs = [(i, os.path.join(snaps2_dir, os.path.basename(i))) \
                    for i in zoomed]
    labels = display_cell_prompt(zoomed_pairs, rois=newRois)
    
    for file in labels['cull']:
        nd2_name = os.path.splitext(os.path.basename(file))[0] + ".nd2"
        old_path = os.path.join(spt_folder, nd2_name)
        try:
            os.rename(old_path, os.path.join(cull_folder, nd2_name))
        except:
            print(f"Moving {old_path} to culled folder failed. It may not exist. Continuing...")
            continue
        
    return labels