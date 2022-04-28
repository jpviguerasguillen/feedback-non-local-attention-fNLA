'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy

Original paper by J.P. Vigueras-Guillen
Code written by: J.P. Vigueras-Guillen
Date: 28 Feb 2022

If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at J.P.ViguerasGuillen@gmail.com

This file only contains the keras code (Tensorflow) for the data augmentation
used in the aformentioned paper.
'''


import numpy as np   
import cv2                              # Data Aug: Elastic deformations.
from scipy.interpolate import interp2d  # Data Aug: Elastic deformations.



def flip_images(img, lbl=None, flip_type='UD', data_format='channels_last'):
    """ Flips all images vertically (Up-Down) or horizontally (Left-Right). 
    - Labels are also flipped, if introduced.
    - By default, flip is Up-Down (UD). 
        * For Left-Right, introduce 'LR'.
        * For both flips, introduce '2A'.
    - See that we flip the indicated axis, all images simultaneously. 
    # Args:
        img:        input tensor, 4D (N,H,W,C) by default
        lbl:        input tensor (the labels), 4D (N,H,W,C) 
        flip_type:  string, the type of fliping: 'UD', 'LR', '2A'
    # Returns
        Numpy arrays of the img and lbl, transformed.
    """ 
    assert(len(img.shape) == 4)     
    # Select the type
    if flip_type == 'LR':
        axis_flip = 2                    # Left-Right 
    elif flip_type == '2A':
        axis_flip = (1, 2)               # Both
    else:
        axis_flip = 1                    # Up-Down    
        
    # Check channel place
    if data_format == 'channels_first':
        img = np.transpose(img, (0, 2, 3, 1))
        lbl = np.transpose(lbl, (0, 2, 3, 1)) if lbl is not None else None
        
    # Flip     
    img = np.flip(img, axis=axis_flip)
    lbl = np.flip(lbl, axis=axis_flip) if lbl is not None else None
    
    # Reshape if necessary
    if data_format == 'channels_first':
        img = np.transpose(img, (0, 3, 1, 2))   
        lbl = np.transpose(lbl, (0, 3, 1, 2)) if lbl is not None else None   
    return img, lbl



# ----------------------------------------------------------------------------
# Elastic deformations.
# ----------------------------------------------------------------------------

random_generator = np.random.RandomState(seed=11) 

  
def apply_random_deformations(random_generator, img, lbl=None, stddev=10, 
                              flag_constant=True, border_value=0):
    """ Based on a deformed grid, it creates the new image and labels.    
    The borders are either treated as a constant or with reflection. 
    Types of border reflection:
        - With BORDER_REFLECT_101, the outer-most border is not reflected. 
        - With BORDER_REFLECT, the outer-most border is reflected (copied). 
    Labels and images need to be clipped at the end, as values smaller 
    than 0 (or larger than 1) can happen.    
    In U-net, the stddev is 10.
    # Args:
        random_generator: a numpy random generator.
        img:            input tensor, 4D (N,H,W,C) by default
        lbl:            input tensor (the labels), 4D (N,H,W,C) 
        stddev:         float, the standard deviation used to create the grid 
                        (as it uses a Gaussian)
        flag_constant:  boolean, to select the type of border mode (so far, 
                        either a constant -the default- or the reflection).
        border_value:   int, the value used in the border mode (constant option).                
    # Returns
        Numpy arrays of the img and lbl, transformed.
    """    
    assert(len(img.shape)==4)  
    border_mode=cv2.BORDER_CONSTANT if flag_constant else cv2.BORDER_REFLECT_101
    
    # Initialize output
    Ni,hi,wi,ci = img.shape  
    deformed_img = np.zeros_like(img)
    deformed_lbl = np.zeros_like(lbl) if lbl is not None else None   
    
    for i in range(Ni):
        # Randomly create an elastic deformed grid
        x_new, y_new = create_elastic_deformed_grid(random_generator, row=hi, 
                                                    col=wi, stddev=stddev)
        # Apply the deformations
        for j in range(ci):
            deformed_img[i,:,:,j] = cv2.remap(img[i,:,:,j], 
                                              map1=x_new, map2=y_new, 
                                              interpolation=cv2.INTER_CUBIC, 
                                              borderMode=border_mode, 
                                              borderValue=border_value)   
        if lbl is not None:
            for j in range(lbl.shape[3]):
                deformed_lbl[i,:,:,j] = cv2.remap(lbl[i,:,:,j], 
                                                  map1=x_new, map2=y_new, 
                                                  interpolation=cv2.INTER_CUBIC,  
                                                  borderMode=border_mode,
                                                  borderValue=border_value)  
    # Clip images and labels
    deformed_img[deformed_img < 0] = 0
    deformed_img[deformed_img > 1] = 1
    if lbl is not None:
        deformed_lbl[deformed_lbl < 0] = 0
        deformed_lbl[deformed_lbl > 1] = 1

    return deformed_img, deformed_lbl



def create_elastic_deformed_grid(random_generator, row, col, stddev=10):
    """ It creates a elastic deformed grid, which will be used to deform the 
    images and labels. It creates the matrices x_new and y_new, which are 
    mesh-grid matrices, this is, for each pixel indicates the original 
    coordinates (x_new for the x-coordinate, y_new for the y-coordinate).
    In U-net, the stddev is 10.
    """
    # Draw random dx,dy displacements for a 4x4 grid covering the whole image
    grid_x = np.linspace(-(col-1),(2*col-1), 4)
    grid_y = np.linspace(-(row-1),(2*row-1), 4)
    aspect_ratio = float(row) / float(col) #To correct for rectangular grid
    grid_dx = random_generator.normal(size=(4,4), 
                                      scale=stddev).astype(np.float32)
    grid_dy = random_generator.normal(size=(4,4), 
                                   scale=aspect_ratio*stddev).astype(np.float32)    
    
    # Create bicubic interp. functions based on the displacements of the grid
    eval_dx = interp2d(x=grid_x, y=grid_y, z=grid_dx, kind='cubic', 
                       copy=True, bounds_error=True, fill_value=np.nan)
    eval_dy = interp2d(x=grid_x, y=grid_y, z=grid_dy, kind='cubic', 
                       copy=True, bounds_error=True, fill_value=np.nan)   
    
    # Evaluate the bicubic interpolation functions over the whole image
    x = np.linspace(0, col-1, col)
    y = np.linspace(0, row-1, row)
    dx = eval_dx(x, y).astype(np.float32)
    dy = eval_dy(x, y).astype(np.float32)       
    
    # Create normal grid
    x_new, y_new = np.meshgrid(x, y)
    x_new = x_new.astype(np.float32)
    y_new = y_new.astype(np.float32)     
    
    # Add the random displacements
    x_new += dx
    y_new += dy    
    return x_new, y_new
