# -*- coding: utf-8 -*-
"""

@author: bswin
"""

from PySide6.QtCore import QSize, Qt, QPointF
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QComboBox, QCheckBox, QLabel, QLineEdit, QMainWindow, QPushButton, QGridLayout, QFileDialog, QVBoxLayout, QHBoxLayout
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from saspt.io import load_detections
from saspt import StateArrayDataset
from saspt import StateArray
from os import path
from PIL import Image
from cell_culler import run_cell_culler


import sys
import csv
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


class MainWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("SASPT Processing")
        self.setFixedSize(QSize(1400,700))
        
        self.layout = QGridLayout()
        
        self.selectedFile = 0
        #Keys corresponding to processed stats
        self.keys = ['n_tracks', 'n_jumps', 'n_detections', 'mean_track_length', 'max_track_length', 'mean_jumps_per_track', 'mean_detections_per_frame', 'max_detections_per_frame', 'fraction_of_frames_with_detections']
        self.polyPaths = []
        self.slashKey = self.findSlashKey()
        
        #File Uploading Interface
        self.fileWidget = self.initUploadInterface()
        
        #Parameters Interface
        self.paramWidget = self.initParamInterface()
        self.paramWidget.hide()
        
        #Statistics Display
        self.statsWidget = self.initStatDisplay()
        self.statsWidget.hide()
        
        #Statistics Graph
        self.statGraphWidget = self.initStatGraph()
        self.statGraphWidget.hide()
        
        #Images Display
        self.imageWidget = ImageWindow(self)
        self.imageWidget.hide()
        
        #Selection Interface
        self.selectWidget = self.initSelectInterface()
        self.selectWidget.hide()
        

        
        self.layout.addWidget(self.fileWidget, 0, 0)
        self.layout.addWidget(self.paramWidget, 1, 0)
        self.layout.addWidget(self.statsWidget, 0, 1)
        
        
        widget = QWidget()
        widget.setLayout(self.layout)
        self.setCentralWidget(widget)
        
        
    #Determines the correct 'slash' or file separator on the given os    
    def findSlashKey(self):
        key = path.join('a', 'dir')[1]
        #Safety mechanism
        if (key == '\\'):
            key = '\\'
        return key
        
    #Initialized the interface for uploading files
    def initUploadInterface(self):
        fileLayout = QVBoxLayout()
        uploadBarLayout = QHBoxLayout()
        fileButton = QPushButton("Upload Files")
        fileButton.clicked.connect(self.uploadFiles)
        fileButton.setFixedSize(QSize(100,50))
        uploadPickleButton = QPushButton("Upload Pickle")
        uploadPickleButton.clicked.connect(self.uploadPickle)
        uploadPickleButton.setFixedSize(QSize(100,50))
        uploadBarLayout.addWidget(fileButton)
        uploadBarLayout.addWidget(uploadPickleButton)
        uploadBarWidget = QWidget()
        uploadBarWidget.setLayout(uploadBarLayout)
        uploadBarWidget.setFixedSize(QSize(200,100))
        self.fileConf = QLabel()
        fileLayout.addWidget(uploadBarWidget)
        fileLayout.addWidget(self.fileConf)
        fileWidget = QWidget()
        fileWidget.setLayout(fileLayout)
        return fileWidget
    
    #Initialize the interface for inputting dataset parameters
    def initParamInterface(self):
        self.paramLayout = QGridLayout()
        paramLabels = self.getParamLabels()
        for i in range(len(paramLabels)):
            self.paramLayout.addWidget(paramLabels[i], i, 0)
            #Dummy widgets to clamp QLineEdits
            self.paramLayout.addWidget(QWidget(), i, 2)
        helpButton = QPushButton("?")
        helpButton.clicked.connect(self.showParamHelp)
        self.paramLayout.addWidget(helpButton, 0, 1)
        self.paramInputs = self.getParamInputs()
        inputPos = (1, 2, 3, 5, 6, 7, 8, 9, 11, 12)
        for i in range(len(self.paramInputs)):
            self.paramLayout.addWidget(self.paramInputs[i], inputPos[i], 1)
        self.progressCheck = QCheckBox()
        self.progressCheck.setChecked(True)
        self.paramLayout.addWidget(self.progressCheck, 10, 1)
        submitParams = QPushButton("Submit Settings")
        submitParams.clicked.connect(self.confirmSettings)
        self.paramConf = QLabel()
        self.paramLayout.addWidget(submitParams, 14, 0)
        self.paramLayout.addWidget(self.paramConf, 15, 0)
        paramWidget = QWidget()
        paramWidget.setLayout(self.paramLayout)
        return paramWidget
    
    #Shows clarifying information for parameters
    def showParamHelp(self):
        info = ["", " Length of one edge of a square pixel in micrometers", "The length from one frame to the next in seconds", 
                "The axial detection range of the microscope in micrometers", "", "Treat trajectories with more jumps than this variable as separate trajectories",
                "Ignore trajectories that start before this frame number", "Maximum number of trajectories to feed into the inference routine",
                "Maximum number of inference iterations to perform", "Strength of the prior; Lower values here will make your output more sensitive to variations in your trajectories, and will make plots look less smeary",
                "Shows progress", "Number of processor threads to use for computation", "Analysis Model"]
        for i in range(len(info)):
            self.paramLayout.addWidget(QLabel(info[i]), i, 2)
    
    #Initialize the display for relevant statistics
    def initStatDisplay(self):
        statsLayout = QVBoxLayout()
        statsTitle = QLabel("STATISTICS")
        font = statsTitle.font()
        font.setPointSize(25)
        statsTitle.setFont(font)
        self.fractionBound = QLabel()
        self.cumulativeMedian = QLabel()
        statsLayout.addWidget(statsTitle)
        statsLayout.addWidget(self.fractionBound)
        statsLayout.addWidget(self.cumulativeMedian)
        self.statTrackLabels = []
        for i in range(9):
            label = QLabel()
            self.statTrackLabels.append(label)
            statsLayout.addWidget(label)
        self.meanIntensity = QLabel()
        self.totalIntensity = QLabel()
        self.altMeanIntensity = QLabel()
        self.altTotalIntensity = QLabel()
        graphButton = QPushButton("Graph Statistics")
        graphButton.clicked.connect(self.seeGraph)
        statsLayout.addWidget(self.meanIntensity)
        statsLayout.addWidget(self.totalIntensity)
        statsLayout.addWidget(self.altMeanIntensity)
        statsLayout.addWidget(self.altTotalIntensity)
        statsLayout.addWidget(graphButton)
        statsWidget = QWidget()
        statsWidget.setLayout(statsLayout)
        return statsWidget
    
    #Initialize the graphing interface for comparing statistics
    def initStatGraph(self):
        statGraphLayout = QGridLayout()
        self.hoveredPoint = QLabel()
        self.statSelectY = self.getStatSelect()
        self.graphWidget = GraphWidget(self)
        self.graphWidget.setBackground('w')
        leftBarLayout = QVBoxLayout()
        self.statSelectX = self.getStatSelect()
        clearRoi = QPushButton("Clear ROI")
        clearRoi.clicked.connect(self.clearRoi)
        setRoi = QPushButton("Create ROI")
        setRoi.clicked.connect(self.graphWidget.setRoi)
        showSpectrum = QPushButton("Show Diffusion Spectrum")
        showSpectrum.clicked.connect(self.showSpectrum)
        leftBarLayout.addWidget(self.hoveredPoint)
        leftBarLayout.addWidget(self.statSelectY)
        leftBarLayout.addWidget(QWidget())
        leftBarLayout.addWidget(clearRoi)
        leftBarLayout.addWidget(setRoi)
        leftBarLayout.addWidget(showSpectrum)
        leftBarWidget = QWidget()
        leftBarWidget.setLayout(leftBarLayout)
        returnButton = QPushButton("Return")
        returnButton.clicked.connect(self.hideGraph)
        roiStatsLayout = QVBoxLayout()
        self.roiStatsSelected = QLabel()
        self.roiFractionBound = QLabel()
        roiStatsLayout.addWidget(self.roiStatsSelected)
        roiStatsLayout.addWidget(self.roiFractionBound)
        self.roiStatLabels = []
        for i in range(10):
            label = QLabel()
            self.roiStatLabels.append(label)
            roiStatsLayout.addWidget(label)
        roiStatsWidget = QWidget()
        roiStatsWidget.setLayout(roiStatsLayout)
        statGraphLayout.addWidget(leftBarWidget, 0, 0)
        statGraphLayout.addWidget(self.graphWidget, 0, 1)
        statGraphLayout.addWidget(returnButton, 1, 0)
        statGraphLayout.addWidget(self.statSelectX, 1, 1)
        statGraphLayout.addWidget(roiStatsWidget, 0, 2)
        statGraphWidget = QWidget()
        statGraphWidget.setLayout(statGraphLayout)
        return statGraphWidget
    
    #Initialize the interface for selecting a file and additional visuals
    def initSelectInterface(self):
        selectLayout = QVBoxLayout()
        pickleButton = QPushButton("Pickle Dataset")
        pickleButton.clicked.connect(self.savePickle)
        self.selectBar = QComboBox()
        self.selectBar.currentIndexChanged.connect(self.index_changed)
        cellBarLayout = QGridLayout()
        cellBarLayout.addWidget(QLabel("Protein View"), 0, 0)
        cellBarLayout.addWidget(QLabel("Cropped View"), 0, 1)
        cellBarLayout.addWidget(QLabel("Alternate Cell View"), 0, 2)
        self.cellDisplay = []
        for i in range(3):
            label = QLabel()
            self.cellDisplay.append(label)
            cellBarLayout.addWidget(label, 1, i)
        cellBar = QWidget()
        cellBar.setLayout(cellBarLayout)
        self.singleGraph = QLabel()
        selectLayout.addWidget(pickleButton)
        selectLayout.addWidget(self.selectBar)
        selectLayout.addWidget(cellBar)
        selectLayout.addWidget(self.singleGraph)
        selectWidget = QWidget()
        selectWidget.setLayout(selectLayout)
        return selectWidget

    # Accepts one or more user selected files and loads them into detections folder
    def uploadFiles(self):
        self.fileNames, self._filter = QFileDialog.getOpenFileNames(self, "Select Files", ".", "Quot Traj Files (*.csv)")
        if self.fileNames is not None:   
            for i in range(len(self.fileNames)):
                self.fileNames[i] = self.fileNames[i].replace("/",self.slashKey)
            self.detections = []
            for file in self.fileNames:
                self.detections.append(load_detections(file))
            self.fileConf.setText("Files uploaded successfully")
            self.targetFolder = path.split(path.split(self.fileNames[0])[0])[0]
            culling = True
            fileNums = self.getFileNum(self.getShortNames(self.fileNames))
            print(fileNums)
            for i in range(len(fileNums)):
                try:
                    fileNums[i] = int(fileNums[i])
                except:
                    culling = False
                    break
            if culling:
                kept = self.getFileNum(self.getShortNames(run_cell_culler(self.targetFolder, fileNums)['keep']))
                oldFileNames = self.fileNames.copy()
                oldFileNums = self.getFileNum(self.getShortNames(self.fileNames))
                self.fileNames = []
                for i in range(len(oldFileNames)):
                    if oldFileNums[i] in kept:
                        self.fileNames.append(oldFileNames[i])
                print(kept)
                print(oldFileNames)
                print(oldFileNums)
                print(self.fileNames)
            self.shortNames = self.getShortNames(self.fileNames)
            self.paramWidget.show()
        
    # Accepts a .pickle file and loads it into a dataset
    def uploadPickle(self):
        fileNames, _filter = QFileDialog.getOpenFileName(self, "Select Files", ".", "Pickle Files (*.pickle, *.pkl)")
        if fileNames is not None:        
            with open(fileNames, 'rb') as f:
                self.dataset = pickle.load(f)
            self.fileNames = self.dataset.paths['filepath']
            self.shortNames = self.getShortNames(self.fileNames)
            self.targetFolder = path.split(path.split(self.fileNames[0])[0])[0]
            self.initGUI()
        
    # Saves the current dataset state to a .pickle file
    def savePickle(self):
        fileName, _filter = QFileDialog.getSaveFileName(self, 'Save File', '', '*.pickle')
        if fileName != '':    
            with open(fileName, 'wb') as f:
                pickle.dump(self.dataset, f, pickle.HIGHEST_PROTOCOL)
         
    #Display progress bars in terms of percent
    def displayProgress(self, percent : float):
        self.paramConf.setText("Dataset loading... please wait...(may take a while)  " + str(percent) + "%")
        self.paramConf.repaint()
    
    #Registers settings and utilizes them to initialize all relevant data
    def confirmSettings(self):
        self.settings = dict(pixel_size_um = float(self.paramInputs[0].text()),
                frame_interval = float(self.paramInputs[1].text()),
                focal_depth = float(self.paramInputs[2].text()),

                splitsize = int(self.paramInputs[3].text()),
                start_frame = int(self.paramInputs[4].text()),
                sample_size = int(self.paramInputs[5].text()),
                max_iter = int(self.paramInputs[6].text()),
                conc_param =  float(self.paramInputs[7].text()),
                progress_bar = self.progressCheck.checkState() == Qt.CheckState.Checked,
                num_workers = int(self.paramInputs[8].text()),

                likelihood_type = self.paramInputs[9].text())
        self.displayProgress('0')
        self.paths = dict(filepath=self.fileNames, condition=self.shortNames)
        self.stateArrays = []
        for detect in self.detections:
            self.stateArrays.append(StateArray.from_detections(detect, **self.settings))
        self.dataset = StateArrayDataset.from_kwargs(pd.DataFrame(self.paths), path_col = 'filepath', condition_col = 'condition', **self.settings)
        self.initGUI()
        
    #Initializes analysis and preloads all necessary information to run the GUI
    def initGUI(self):    
        self.mpo = []
        self.stats = []
        for i in range(len(self.fileNames)):
            self.fileNames[i] = self.fileNames[i].replace("/",self.slashKey)
        if (not self.checkFile()):
            self.paramConf.setText("INVALID FILE - PLEASE RETRY")
            return
        try:
            self.rois = self.loadRois()
        except:
            self.rois = None
        progress = 0
        for file in self.fileNames:
            occs = self.dataset.marginal_posterior_occs
            for fileOcc in occs:
                counter = 0
                for n in fileOcc:
                    counter += n
                self.mpo.append(fileOcc/n)
            progress += 100 / len(self.fileNames)
            self.displayProgress(progress)
        processedStats = self.dataset._get_processed_track_statistics().set_index("condition")
        for i in range(len(self.fileNames)):
            self.stats.append(Stats(self.mpo[i], self.dataset, self.targetFolder, self.getFileNum(self.shortNames)[i], self.rois, processedStats.loc[self.shortNames[i]]))
        self.paramConf.setText("Dataset Initialized")
        self.paramConf.repaint()
        self.statsWidget.show()
        self.registerImages()
        self.imageWidget.show()
        self.selectBar.addItems(self.shortNames)
        self.layout.replaceWidget(self.fileWidget, self.selectWidget)
        self.fileWidget.hide()
        self.paramWidget.hide()
        self.selectWidget.show()
        self.updateDisplay()
        
    def checkFile(self) -> bool:
        if not path.isdir(self.targetFolder):
            return False
        return True
    
    #Returns short versions of a list of fileNames
    def getShortNames(self, fileNames):
        shortNames = []
        for file in fileNames:
            shortNames.append(path.split(file)[1])
        return shortNames
    
    #Given a list of short file names, returns the numbers corresponding to each file
    def getFileNum(self, fileNames):
        nums = []
        for file in fileNames:
            split = re.split('\.', file)
            if split[1] == 'csv':
                nums.append(re.split('_', file)[0])
            else:
                nums.append(split[0])
        return nums
    
    #Reads all rois into an array
    def loadRois(self):
        roisFile = path.join(self.targetFolder, "rois.txt")
        rois = []
        with open(roisFile, 'r') as fd:
            reader = csv.reader(fd)
            for row in reader:
                rois.append(list(map(int,row)))
        return rois
        
    #Load all graph images that remain stagnant
    def registerImages (self):
        self.graphDiffusionSpectra()
        
    #Graphs all diffusion spectra on a single line graph
    def graphDiffusionSpectra (self):
        for i in range(self.dataset.n_files):
             plt.plot(self.dataset.likelihood.diff_coefs,
             self.mpo[i] / self.mpo[i].sum(),
             label=self.dataset.paths['condition'][i])
        plt.xscale("log")
        plt.legend()
        plt.savefig("diffusion_spectra.png", dpi=600)
        plt.close()
    
    #Graphs single diffusion spectrum on a  line graph
    def graphDiffusionSpectrum (self):
        plt.plot(self.dataset.likelihood.diff_coefs, self.mpo[self.selectedFile] / self.mpo[self.selectedFile].sum(), label=self.dataset.paths['condition'][self.selectedFile])
        plt.xscale("log")
        plt.ylim([0,0.06])
        plt.savefig("diffusion_spectrum.png", dpi=600)
        plt.close()
        
    #Returns a set of labels corresponding to each parameter
    def getParamLabels(self):
        labels = []
        labelText = ("IMAGING PARAMETERS (usually keep default)",
                     "Pixel Size (um)","Frame Interval","Focal Depth",
                     "INFERENCE PARAMETERS", "Split Size", "Start Frame", 
                     "Sample Size", "Max Iterations", "conc_param", "Progress Bar", 
                     "Number of Workers (Threads)", "Model Type")
        for i in range(13):
            labels.append(QLabel(labelText[i]))
        return labels
    
    #Returns a set of lineEdits corresponding to each parameter
    def getParamInputs(self):
        inputs = []
        inputText = ('0.16','0.00748','0.7','10','0','10000','200','1.0','1','rbme')
        for i in range(10):
            inputs.append(QLineEdit())
            inputs[i].setText(inputText[i])
            inputs[i].setFixedSize(QSize(100,20))
        return inputs
    
    #SelectBar index changed
    def index_changed(self, index):
        self.selectedFile = index
        self.updateDisplay()
        self.imageWidget.changeSelect(index)
        
    #Stat Select index changed
    def stat_changed(self, index):
        self.graphWidget.clear()
        self.graphX = self.getArrayOfStats(self.statSelectX.currentIndex())
        self.graphY = self.getArrayOfStats(self.statSelectY.currentIndex())
        self.graphWidget.plot(self.graphX, self.graphY, pen=None, symbol='o', symbolPen=None, symbolSize=10, symbolBrush=(200, 100, 255))
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.loadNew(self.graphX, self.graphY)
        
    #Returns an array of stats of a given type
    def getArrayOfStats(self, index : int):
        arr = []
        if (index == 0):
            for stat in self.stats:
                arr.append(float(stat.fractionBound))
        elif (index == 1):
            for stat in self.stats:
                arr.append(float(stat.cumulativeMedian))
        else:
            for stat in self.stats:
                arr.append(stat.processedStats.loc[self.keys[index-2]])
        return arr
    
    #Generates and sets the cell-level pictures
    def setPicture(self):
        targetNum = self.getFileNum(self.shortNames)[self.selectedFile]
        cellFig = plt.figure(constrained_layout=True)
        cellAx = plt.axes()
        cellFig.add_subplot(cellAx)
        cellAx.set_yticks([])
        cellAx.set_xticks([])
        image = Image.open(path.join(self.targetFolder, "snaps2", targetNum + ".tif"))
        cellAx.imshow(image)
        cellAx.add_patch(plt.Rectangle(xy=(self.rois[int(targetNum)-1][0], self.rois[int(targetNum)-1][1]),
                                      width=self.rois[int(targetNum)-1][2] - self.rois[int(targetNum)-1][0],
                                      height=self.rois[int(targetNum)-1][3] - self.rois[int(targetNum)-1][1],
                                      color='r', fill=False))
        plt.savefig("annotated_cell.tif", dpi=1400, bbox_inches='tight', pad_inches=0)
        plt.close()
        cellPicture = QPixmap()
        cellPicture.load("annotated_cell.tif")
        cellPicture = cellPicture.scaled(QSize(200, 200), Qt.KeepAspectRatio)
        self.cellDisplay[0].setPixmap(cellPicture)
        
        for i in range(1,3):
            cellPicture = QPixmap()
            subfolders = ["snaps3", "snaps4"]
            subfolder = subfolders[i-1]
            cellPicture.load(path.join(self.targetFolder, subfolder, targetNum))
            cellPicture = cellPicture.scaled(QSize(200, 200), Qt.KeepAspectRatio)
            self.cellDisplay[i].setPixmap(cellPicture)
    
    #Updates display based on selected image
    def updateDisplay(self):
        invalidNumber = False
        try:
            self.setPicture()
        except:
            self.cellDisplay[0].setText("Images Not Present or Inaccessible")
            self.cellDisplay[1].clear()
            self.cellDisplay[2].clear()
            invalidNumber = True
        self.graphDiffusionSpectrum()
        graph = QPixmap()
        graph.load("diffusion_spectrum.png")
        graph = graph.scaled(QSize(600, 300), Qt.KeepAspectRatio)
        self.singleGraph.setPixmap(graph)
        self.fractionBound.setText("Fraction Bound (Probability of diffusion coefficient < 0.1): " + str(self.stats[self.selectedFile].fractionBound))
        self.cumulativeMedian.setText("Median Diffusion Coefficient: " + str(self.stats[self.selectedFile].cumulativeMedian))
        self.updateStatTracks()
        if not invalidNumber:    
            self.meanIntensity.setText("Mean Pixel Intensity of Target Area: " + str(self.stats[self.selectedFile].meanIntensity))
            self.totalIntensity.setText("Total Pixel Intensity of Target Area: " + str(self.stats[self.selectedFile].totalIntensity))
            self.altMeanIntensity.setText("Mean Pixel Intensity of Target Area (Alternate View): " + str(self.stats[self.selectedFile].altMeanIntensity))
            self.altTotalIntensity.setText("Total Pixel Intensity of Target Area (Alternate View): " + str(self.stats[self.selectedFile].altTotalIntensity))
        if (self.imageWidget.imageSelect.currentIndex() == 1):
            self.loadTemporalPlot()
    
    #Update statistics derived from processed stat tracks
    def updateStatTracks(self):
        for i in range(9):
            label = self.statTrackLabels[i]
            key = self.keys[i]
            label.setText(key + ": " + str(self.stats[self.selectedFile].processedStats.loc[key]))
        
    #Loads the Temporal Plot and registers that in ImageWindow
    def loadTemporalPlot(self):
        self.stateArrays[self.selectedFile].plot_temporal_assignment_probabilities("assignment_probabilities_by_frame.png")
        self.imageWidget.loadTemporalPlot()
    
    #Switches view to graph
    def seeGraph(self):
        self.layout.replaceWidget(self.selectWidget, self.statGraphWidget)
        self.roi = None
        self.selectWidget.hide()
        self.statsWidget.hide()
        self.statGraphWidget.show()
        self.stat_changed(0)
        
    #Leaves graph view
    def hideGraph(self):
        self.layout.replaceWidget(self.statGraphWidget, self.selectWidget)
        self.selectWidget.show()
        self.statsWidget.show()
        self.statGraphWidget.hide()
        
    #Returns from graph while selecting targeted file at index
    def escapeGraph(self, index):
        self.selectBar.setCurrentIndex(index)
        self.index_changed(index)
        self.hideGraph()
    
    # Returns a select box that allows choice between given statistics
    def getStatSelect(self) -> QComboBox:
        select = QComboBox()
        select.addItems(['Fraction Bound', 'Median Diffusion Coefficient'])
        select.addItems(self.keys)
        select.currentIndexChanged.connect(self.stat_changed)
        return select
    
    # Destroys current roi
    def clearRoi(self):
        self.graphWidget.clearRoi()
        if self.roi is not None:
            self.graphWidget.removeItem(self.roi)
        self.roiStatsSelected.clear()
        self.roiFractionBound.clear()
        for i in range(10):
            self.roiStatLabels[i].clear()
        
    # Creates a polyline roi from marked points
    def buildRoi(self, points : np.ndarray):
        if self.roi is not None:
            self.graphWidget.removeItem(self.roi)
        self.roi = pg.PolyLineROI(points, closed = True, movable=False, rotatable=False, resizable=False, pen = pg.mkPen(pg.mkColor(100, 10, 200), cosmetic=True, width = 3), handlePen = pg.mkPen(pg.mkColor(0, 100, 100), cosmetic=True, width = 2))
        self.graphWidget.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self.roiMoved)
        self.roiPoints = []
        self.roiMoved(self.roi)
        
    #Determines if new points added/removed when roi moved, recalculates analysis from there 
    def roiMoved(self, roi: pg.PolyLineROI):
        path = roi.shape()
        newRoiPoints = []
        for i in range(len(self.graphX)):
            if path.contains(QPointF(self.graphX[i], self.graphY[i])):
                newRoiPoints.append(i)
        if (newRoiPoints != self.roiPoints or len(self.roiPoints) == 0):
            fileList = ""
            self.roiPoints = newRoiPoints
            for ref in self.roiPoints:
                fileList += self.shortNames[ref] + "\n"
            self.roiStatsSelected.setText(fileList)
            self.roiFractionBound.clear()
            for label in self.roiStatLabels:
                label.clear()
            if (len(self.roiPoints) > 0):
                sum_unnormalized = np.zeros((100,))
                for ref in self.roiPoints:
                    sum_unnormalized += self.mpo[ref]*self.stats[ref].processedStats.loc['n_jumps']
                self.normalized = sum_unnormalized / np.sum(sum_unnormalized)
                self.roiFractionBound.setText("Weighted Fraction Bound: " + str(self.normalized[self.dataset.likelihood.diff_coefs < 0.1].sum() / self.normalized.sum()))
                self.generateSpectrum(self.roiPoints)
                sums = np.zeros(10)
                for ref in self.roiPoints:
                    stats = self.stats[ref]
                    for i in range(10):
                        if (i == 0):
                            sums[0] += stats.cumulativeMedian
                        else:
                            sums[i] += stats.processedStats.loc[self.keys[i-1]]
                
                sums /= len(self.roiPoints)
                keys = ['Cumulative Median']
                keys = keys + self.keys
                for i in range(10):
                    self.roiStatLabels[i].setText(keys[i] + ": " + str(sums[i]))
            
        self.graphWidget.clearLastPoint()
       
    #Generates an aggregate spectrum for all roipoints
    def generateSpectrum(self, roiPoints):
        self.spectrumWidget = QWidget()
        spectrumLayout = QVBoxLayout()
        spectrumPic = QLabel()
        plt.plot(self.dataset.likelihood.diff_coefs, self.normalized)
        plt.xscale("log")
        plt.savefig("aggregateRoiSpectrum.png", dpi=600)
        plt.close()
        pixmap = QPixmap()
        pixmap.load("aggregateRoiSpectrum.png")
        pixmap = pixmap.scaled(QSize(500, 500), Qt.KeepAspectRatio)
        spectrumPic.setPixmap(pixmap)
        spectrumLayout.addWidget(spectrumPic)
        self.spectrumWidget.setLayout(spectrumLayout)
        
    #Show aggregate spectrum in additional window
    def showSpectrum(self):
        self.spectrumWidget.show()
        
    #Updates text displaying hovered target
    def setHoveredText(self, index):
        if (index >= 0):
            self.hoveredPoint.setText("HOVERED: " + self.shortNames[index])
        else:
            self.hoveredPoint.clear()
        
        
class ImageWindow(QWidget):
    
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.selectedFile = 0
        self.loadedTemporalPlot = False
        
        self.setWindowTitle("Graph Images")
        imageLayout = QVBoxLayout()
        self.imageSelect = QComboBox()
        self.imageSelect.addItems(["", "Temporal Plot", "Diffusion Spectrum"])
        self.imageSelect.currentIndexChanged.connect(self.index_changed)
        self.imageDisplay = QLabel()
        saveImageButton = QPushButton("Save Graph")
        saveImageButton.clicked.connect(self.saveImage)
        imageLayout.addWidget(self.imageSelect)
        imageLayout.addWidget(self.imageDisplay)
        imageLayout.addWidget(saveImageButton)
        self.setLayout(imageLayout)
        
    #Reloads image based on selected graph
    def index_changed(self, index):
         self.image = QPixmap()
         if (index == 1):
             if (self.loadedTemporalPlot):
                 self.loadedTemporalPlot = False
             else:
                 self.parent.loadTemporalPlot()
                 return
             self.image.load("assignment_probabilities_by_frame.png")
             self.image = self.image.scaled(QSize(900, 600), Qt.KeepAspectRatioByExpanding)
         elif (index == 2):
            self.image.load("diffusion_spectra.png")
            self.image = self.image.scaled(QSize(900, 600), Qt.KeepAspectRatioByExpanding)
         self.imageDisplay.setPixmap(self.image)
         
    #Saves image to folder
    def saveImage(self):
        fileName, _filter = QFileDialog.getSaveFileName(self, 'Save File', '', '*.png')
        self.image.save(fileName, "PNG")
    
    #Graph SelectBar changed
    def changeSelect(self, index):
        self.selectedFile = index
        self.index_changed(self.imageSelect.currentIndex())
    
    #Registers temporal Plot as loaded and reloads if necessary
    def loadTemporalPlot(self):
        self.loadedTemporalPlot = True
        if (self.imageSelect.currentIndex() == 1):
            self.index_changed(1)
            
class GraphWidget(PlotWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.scatterItem = None
        #self.polyPaths = []
        #self.pathStarted = False;
        
    
    #Resets data upon loading a new plot
    def loadNew(self, graphX, graphY):
        self.clearRoi()
        self.proxy = pg.SignalProxy(self.scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        self.graphX = graphX
        self.graphY = graphY
    
    #On mouse pressed down
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.pressPos = event.position()
        super().mousePressEvent(event)

    #On mouse released, determines if clicked or escape
    def mouseReleaseEvent(self, event):
        if (self.selectedFile >= 0):
            self.parent.escapeGraph(self.selectedFile)
        elif (self.pressPos is not None and 
            event.button() == Qt.LeftButton):
                self.clicked()
        self.pressPos = None
        super().mouseReleaseEvent(event)
        
    #Stores location in local graph coordinates as mouse moves
    def mouseMoved(self, evt):
        self.mousePoint = self.p.vb.mapSceneToView(evt[0])
        mouseX = self.mousePoint.x()
        mouseY = self.mousePoint.y()
        self.selectedFile = -1
        for i in range(len(self.graphX)):
            if (math.dist((mouseX, mouseY), (self.graphX[i], self.graphY[i])) < self.getPlotItem().getViewBox().viewPixelSize()[0]*10):
                self.selectedFile = i
                break
        self.parent.setHoveredText(self.selectedFile)
        
    #Adds a roi bounding point on click
    def clicked(self):
        self.scatterItem.addPoints([self.mousePoint.x()], [self.mousePoint.y()])
        self.points.append((self.mousePoint.x(), self.mousePoint.y()))
        self.update()
        
    #Removes roi bounding points from view
    def clearRoi(self):
        self.points = []
        if self.scatterItem is not None:
            self.getPlotItem().removeItem(self.scatterItem)
        brush = pg.mkBrush(color=(255,0,0))
        self.scatterItem = pg.ScatterPlotItem(pen=None, size=10, brush=brush, symbol='s')
        self.p = self.getPlotItem()
        self.p.addItem(self.scatterItem)
    
    #Establishes roi based on current bound points
    def setRoi(self):
        self.parent.buildRoi(self.points)
        self.clearRoi()
        
    #Clears last placed point, used to delete accidental points from manipulating roi with mouse
    def clearLastPoint(self):
        if (len(self.points) > 0):
            self.points.pop()
        x, y = self.scatterItem.getData()
        self.scatterItem.setData(x[:-1], y[:-1])

        
class Stats():
    
    def __init__(self, mpo, dataset, targetFolder, targetNum, rois, processedStats):
        self.fractionBound = mpo[dataset.likelihood.diff_coefs < 0.1].sum() / mpo.sum()
        self.cumulativeMedian = mpo[self.cum_median(mpo)]
        try:
            self.rois = rois[int(targetNum)-1]
        except:
            self.rois = None
        self.processedStats = processedStats
        if self.rois is not None:
            self.meanIntensity, self.totalIntensity = self.getIntensity(plt.imread(targetFolder + "/snaps2/" + targetNum + ".tif"))
            self.altMeanIntensity, self.altTotalIntensity = self.getIntensity(plt.imread(targetFolder + "/snaps4/" + targetNum + ".tif"))

        
        
    def cum_median(self, arr: np.ndarray) -> int:
        """Given a 1D numpy array, find the index of the cumulative median."""
        return np.where(np.nancumsum(arr) >=  (np.nansum(arr)/2))[0][0]
    
    #Determines mean and total pixel intensity for a given image array
    def getIntensity(self, arr: np.ndarray):
        count = 0
        intensity = 0
        for i in range(self.rois[0], self.rois[2]):
            for j in range(self.rois[1], self.rois[3]):
                intensity += arr[i][j]
                count += 1
        return intensity / count, intensity
        
    

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()
        
    window = MainWindow()
    window.show()  
    
    sys.exit(app.exec())
