'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy

Original paper by J.P. Vigueras-Guillen
Code written by: J.P. Vigueras-Guillen
Date: 28 Feb 2022

If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at J.P.ViguerasGuillen@gmail.com

This file only contains the keras code (Tensorflow) to create the targets of
the different networks (CNN-Edge and CNN-Body), used in the aformentioned 
paper.
'''


import numpy as np   
import cv2     
import scipy.stats as stats             # Labels: To make Gaussian
import scipy.ndimage as ndi             # Data Aug & Labels.

  

def make_label_CellEdges(labels):
    """ This function takes the gold standard images (originally TIF images, 
    gray scale, 256 values, loaded as numpy arrays [ints]), it binarizes them   
    such that only the cell edges are identified, and it computes the   
    probability images that indicates (for each pixel) how probable it is an edge.  
    For that, we apply a Gaussian smoothing and normalization.    
    Images are stored in a Numpy ndarray, in float32 precision.
    NOTE: I have computed the Gaussian smoothing by a 2D convolution and 
          by two steps of 1D convolution. The latter is 6 times faster!! 
    # Args:
        labels:     numpy array (ints), 4D (N,H,W,C)
    # Returns
        Numpy arrays of the labels (float32).    
    """
    # First, threshold the image so there are only 2 types of labels.
    labels = threshold_labels2edges(labels)
    
    # Modify the Gold standard so edges are 1s and the rest is 0
    labels = np.absolute(labels - 1)
    
    # Allocate space
    probImage = np.zeros(labels.shape, dtype=np.float32)
    
    # Create the Gaussian.
    x = np.linspace(-3,3,7)         # Use 7 pixels
    y = stats.norm.pdf(x,0,1)       # Gaussian with zero mean and sigma=1
    y_norm = np.sum(np.power(y, 2)) # Factor to normalize 
    
    # Apply the Gaussian filter
    for i in range(labels.shape[0]):
        img1 = labels[i,:,:,0]                 # Take one image.
        img2 = ndi.convolve1d(img1, y, axis=0) # Apply convolution in axis 0
        img2 = ndi.convolve1d(img2, y, axis=1) # ... and then in axis 1
        img3 = np.divide(img2, y_norm)         # Normalize
        img3[img3 > 1] = 1                     # Upper threshold 
        probImage[i,:,:,0] = img3
    return probImage



def make_label_CellBodies(labels):
    """ This function takes the gold standard images (originally TIF images, 
    gray scale, 256 values, loaded as numpy arrays [ints]), it binarizes them   
    such that only the (full) cell bodies are identified, and it computes the   
    probability images that indicates (for each pixel) how probable it is a cell.  
    For that, we apply a Gaussian smoothing and normalization.    
    Images are stored in a Numpy ndarray, in float32 precision.
    NOTE: I have computed the Gaussian smoothing by a 2D convolution and 
          by two steps of 1D convolution. The latter is 6 times faster!! 
    # Args:
        labels:     numpy array (ints), 4D (N,H,W,C)
    # Returns
        Numpy arrays of the labels (float32).  
    """
    # First, threshold the image so there are only 2 types of labels.
    labelsTemp = np.zeros(labels.shape, dtype=np.float32)
    for i in range(labels.shape[0]):
      labelsTemp[i,:,:,0] = threshold_labels2cells(labels[i,:,:,0])
    
    # Set (temporally) the cell bodies to zero for computational purposes.
    labels = np.absolute(labelsTemp - 1)
    
    # Allocate space
    probImage = np.zeros(labels.shape, dtype=np.float32)
    
    # Create the Gaussian.
    x = np.linspace(-3,3,7)         # Use 7 pixels
    y = stats.norm.pdf(x,0,1)       # Gaussian with zero mean and sigma=1
    y_norm = np.sum(np.power(y, 2)) # Factor to normalize 
    
    # Apply the Gaussian filter
    for i in range(labels.shape[0]):
        img1 = labels[i,:,:,0]                 # Take one image.
        img2 = ndi.convolve1d(img1, y, axis=0) # Apply convolution in axis 0
        img2 = ndi.convolve1d(img2, y, axis=1) # ... and then in axis 1
        img3 = np.divide(img2, y_norm)         # Normalize
        img3[img3 > 1] = 1                     # Upper threshold 
        probImage[i,:,:,0] = img3

    # Transform back the cell bodies to one.
    probImage = np.absolute(probImage - 1)
    return probImage
    
    
    
def threshold_labels2edges(labels, thresh=25):
    """ Function to transform the original gold standard images (manual 
    annotations) into a binary image where 0 are cell edges.
    The original gold standard images (the input 'labels') are TIF images, 
    gray scale, 256 values. Several labels might appear. Currently, there are:
        - 000: Cell ddges.
        - 128: Areas that should be excluded in the cell evaluation.
        - 255: Cell bodies.
    Thus, the function does:
    1) Edges are 0 and all the rest is 1.
    2) The image is transformed to float32.
    """
    labelX = np.copy(labels)
    labelX[labelX < thresh] = 0  
    labelX[labelX > thresh] = 1  
    labelX = labelX.astype(dtype=np.float32) 
    return labelX
    


def threshold_labels2cells(labels, thresh=25):
    """ Function to transform the original gold standard images (manual 
    annotations) into a binary image where 1 are cell bodies.
    The original gold standard images (the input 'labels') are TIF images, 
    gray scale, 256 values. Several labels might appear. Currently, there are:
        - 000: Cell ddges.
        - 128: Areas that should be excluded in the cell evaluation.
        - 255: Cell bodies.
    Thus, the function does:
    1) Threshold the labels, setting to 1 all the full-cells (not the partial 
       cells in the image border) and setting to 0 the edges, outer area, and 
       inner areas to discard (label 128).
    2) The image is transformed to float32.
    """
    labelX = np.copy(labels)
    labelX[labelX < thresh] = 0   
    labelX[labelX == 128] = 0    # Set to 0s the inner areas to discard
    _, labelX = cv2.connectedComponentsWithAlgorithm(labelX, connectivity=4, 
                                                     ltype = cv2.CV_32S, 
                                                     ccltype=cv2.CCL_DEFAULT)
    
    # Take all the connected components in the image border and set to 0
    lab2rem = np.append(np.unique(labelX[:,0]), np.unique(labelX[0,:]))
    lab2rem = np.append(lab2rem, np.unique(labelX[:,-1]))
    lab2rem = np.append(lab2rem, np.unique(labelX[-1,:]))
    lab2rem = np.unique(lab2rem)
    for ii in range(len(lab2rem)):
        labelX[labelX==lab2rem[ii]] = 0
    
    labelX[labelX > 1] = 1
    labelX = labelX.astype(dtype=np.float32) # Make it float32
    return labelX