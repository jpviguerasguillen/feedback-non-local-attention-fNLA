'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy

Original paper by J.P. Vigueras-Guillen
Code written by: J.P. Vigueras-Guillen
Date: 28 Feb 2022

If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at J.P.ViguerasGuillen@gmail.com

This file only contains the keras code (Tensorflow) for the network definition  
of the default UNet tested in the aformentioned paper.
'''


import tensorflow as tf 
import tensorflow.keras.layers as layers 
import tensorflow.keras.losses as losses
import tensorflow.keras.backend as K 
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import activations, initializers, regularizers, constraints



#-----------------------------------------------------------------------------
# Unet modules
#-----------------------------------------------------------------------------


def conv_block(x, filters, stage, block, filter_size=3, strides=1, drop_rate=None):
    """A convolutional block.
    # Arguments
        x:            input tensor    
        filters:      integer, the number of feature maps in the conv layer    
        stage:        integer, current stage label (to generate layer names)
        block:        string, current block label (to generate layer names)        
        filter_size:  integer, the size of the filters (default = 3)
        strides:      Strides for the conv layer in the block.
        drop_rate:    float, the dropout rate (0-1), or None if not used.
    # Returns
        Output tensor for the block.
    """ 
    x = layers.Conv2D(filters, filter_size, strides=strides, padding='same',
                      kernel_initializer='he_normal', 
                      name='conv' + str(stage) + block)(x)
    x = layers.BatchNormalization(name='bn' + str(stage) + block)(x)
    x = layers.Activation('relu', name='act' + str(stage) + block)(x)
    if drop_rate is not None:
        x = layers.Dropout(rate=drop_rate, name='drop' + str(stage) + block)(x)
    return x



def upconv_block(x, filters, stage, block, filter_size=2, strides=2, drop_rate=None):
    """A upsampling-convolutional block.
    # Arguments
        x:            input tensor    
        filters:      integer, the number of feature maps in the conv layer    
        stage:        integer, current stage label (to generate layer names)
        block:        string, current block label (to generate layer names)        
        filter_size:  integer, the size of the filters (default = 3)
        strides:      Strides for the conv layer in the block.
        drop_rate:    float, the dropout rate (0-1), or None if not used.
    # Returns
        Output tensor for the block.
    """ 
    x = layers.Conv2DTranspose(filters, filter_size, strides=strides, 
                               padding='same', kernel_initializer='he_normal',
                               name='upconv' + str(stage) + block)(x)
    x = layers.BatchNormalization(name='bn' + str(stage) + block)(x)
    x = layers.Activation('relu', name='act' + str(stage) + block)(x)
    if drop_rate is not None:
        x = layers.Dropout(rate=drop_rate, name='drop' + str(stage) + block)(x)
    return x





#-----------------------------------------------------------------------------
# THE BACKBONE OF THE NETWORK 
#-----------------------------------------------------------------------------
 

def UNet(input_shape, classes, drop_rate=None, weights=None, lr=0.001):
    """ Instantiates the Unet architecture. Optionally loads weights.
    # Arguments
        input_shape:  tuple, (H,W,C)           
        classes:      int, number of classes to classify images into, only to  
                      be specified if no `weights` argument is specified.  
        drop_rate:    float, the dropout rate (0-1), or None if not used.                      
        weights:      one of `None` (random initialization), or the path to the 
                      weights file to be loaded.
        lr:           float, learning rate of the optimizer.
    # Returns
        A TensorFlow-Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
                    or invalid input shape.
    """

    if not (weights in {None} or os.path.exists(weights)):
      raise ValueError('The `weights` argument should be either '
                       '`None` (random initialization), '
                       'or the path to the weights file to be loaded.')
    
      
    inputs = layers.Input(shape=input_shape)
    
    x1 = conv_block(inputs, 64, stage=1, block='a', strides=2, filter_size=7)

    x2 = conv_block(x1, 64, stage=2, block='a', strides=1, drop_rate=drop_rate)
    x2 = conv_block(x2, 64, stage=2, block='b', strides=1, drop_rate=drop_rate)
    x2 = conv_block(x2, 64, stage=2, block='c', strides=1, drop_rate=drop_rate)
    
    x3 = conv_block(x2, 128, stage=3, block='a', strides=2)
    x3 = conv_block(x3, 128, stage=3, block='b', strides=1, drop_rate=drop_rate)
    x3 = conv_block(x3, 128, stage=3, block='c', strides=1, drop_rate=drop_rate)
    x3 = conv_block(x3, 128, stage=3, block='d', strides=1, drop_rate=drop_rate)
    
    x4 = conv_block(x3, 256, stage=4, block='a', strides=2)
    x4 = conv_block(x4, 256, stage=4, block='b', strides=1, drop_rate=drop_rate)
    x4 = conv_block(x4, 256, stage=4, block='c', strides=1, drop_rate=drop_rate)
    x4 = conv_block(x4, 256, stage=4, block='d', strides=1, drop_rate=drop_rate)
    x4 = conv_block(x4, 256, stage=4, block='e', strides=1, drop_rate=drop_rate)
    
    x5 = conv_block(x4, 512, stage=5, block='a', strides=2)
    x5 = conv_block(x5, 512, stage=5, block='b', strides=1, drop_rate=drop_rate)
    x5 = conv_block(x5, 512, stage=5, block='c', strides=1, drop_rate=drop_rate)    

    x6 = upconv_block(x5, 256, stage=6, block='up', strides=2)   
    x6 = layers.Concatenate(axis=-1, name='concat_6')([x6, x4])
    x6 = conv_block(x6, 512, stage=6, block='a', strides=1, drop_rate=drop_rate)
    x6 = conv_block(x6, 512, stage=6, block='b', strides=1, drop_rate=drop_rate)
    x6 = conv_block(x6, 512, stage=6, block='c', strides=1, drop_rate=drop_rate)
    x6 = conv_block(x6, 512, stage=6, block='d', strides=1, drop_rate=drop_rate)
    
    x7 = upconv_block(x6, 128, stage=7, block='up', strides=2)   
    x7 = layers.Concatenate(axis=-1, name='concat_7')([x7, x3])
    x7 = conv_block(x7, 256, stage=7, block='a', strides=1, drop_rate=drop_rate)
    x7 = conv_block(x7, 256, stage=7, block='b', strides=1, drop_rate=drop_rate)
    x7 = conv_block(x7, 256, stage=7, block='c', strides=1, drop_rate=drop_rate)

    x8 = upconv_block(x7, 64, stage=8, block='up', strides=2)   
    x8 = layers.Concatenate(axis=-1, name='concat_8')([x8, x2])
    x8 = conv_block(x8, 128, stage=8, block='a', strides=1, drop_rate=drop_rate)
    x8 = conv_block(x8, 128, stage=8, block='b', strides=1, drop_rate=drop_rate)
    
    x9 = upconv_block(x8, 32, stage=9, block='up', strides=2)      

    x10 = layers.Conv2D(classes, 1, activation='relu', name='conv10_out')(x9)
    x10 = layers.Reshape((int(input_shape[0]*input_shape[1]),classes))(x10)
    x10 = layers.Activation('softmax', name="edge_output")(x10)

    model = Model(inputs=inputs, outputs=x10, name='Unet')
    if weights is not None:
        model.load_weights(weights)
    nadam = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam, 
                  loss= 'categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model


# This model is for CNN-Edge because it has only 1 feature map in the input.
# For the CNN-Body, it should be 2 feature maps.
model = ResUNeXt(input_shape = (528, 240, 1), 
                 classes = 2, 
                 drop_rate = 0.2,
                 lr = 0.001)  
model.summary()