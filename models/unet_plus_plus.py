'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy

Original paper by J.P. Vigueras-Guillen
Code written by: J.P. Vigueras-Guillen
Date: 28 Feb 2022

If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at J.P.ViguerasGuillen@gmail.com

This file only contains the keras code (Tensorflow) for the network definition  
of the default UNet++ tested in the aformentioned paper.
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
    
    x01 = conv_block(inputs, 64, stage=101, block='a', strides=2, filter_size=7)
    
    #---------------------------------------------------------------------------
    # Diagonal #2 
    x02 = conv_block(x01, 64, stage=102, block='a', strides=1, drop_rate=drop_rate)
    x02 = conv_block(x02, 64, stage=102, block='b', strides=1, drop_rate=drop_rate)
    
    x03 = conv_block(x02, 128, stage=103, block='a', strides=2)
    x03 = conv_block(x03, 128, stage=103, block='b', strides=1, drop_rate=drop_rate)
    x03 = conv_block(x03, 128, stage=103, block='c', strides=1, drop_rate=drop_rate)
    x03u = upconv_block(x03, 64, stage=103, block='up', strides=2) 
    
    x04 = conv_block(x03, 256, stage=104, block='a', strides=2)
    x04 = conv_block(x04, 256, stage=104, block='b', strides=1, drop_rate=drop_rate)
    x04 = conv_block(x04, 256, stage=104, block='c', strides=1, drop_rate=drop_rate)
    x04 = conv_block(x04, 256, stage=104, block='d', strides=1, drop_rate=drop_rate)
    x04u = upconv_block(x04, 128, stage=104, block='up', strides=2) 
    
    x05 = conv_block(x04, 512, stage=105, block='a', strides=2)
    x05 = conv_block(x05, 512, stage=105, block='b', strides=1, drop_rate=drop_rate)
    x05 = conv_block(x05, 512, stage=105, block='c', strides=1, drop_rate=drop_rate)  
    x05u = upconv_block(x05, 256, stage=105, block='up', strides=2)     

    #---------------------------------------------------------------------------
    # Diagonal #2 
    x12 = layers.Concatenate(axis=-1, name='concat_112')([x02, x03u])
    x12 = conv_block(x12, 128, stage=112, block='a', strides=1, drop_rate=drop_rate)
    x12 = conv_block(x12, 128, stage=112, block='b', strides=1, drop_rate=drop_rate)
    x12t = conv_block(x12, 64, stage=112, block='tr', strides=1)
    x12u = upconv_block(x12, 32, stage=112, block='up', strides=2)   

    x13 = layers.Concatenate(axis=-1, name='concat_113')([x03, x04u])
    x13 = conv_block(x13, 256, stage=113, block='a', strides=1, drop_rate=drop_rate)
    x13 = conv_block(x13, 256, stage=113, block='b', strides=1, drop_rate=drop_rate)
    x13 = conv_block(x13, 256, stage=113, block='c', strides=1, drop_rate=drop_rate)
    x13t = conv_block(x13, 128, stage=113, block='tr', strides=1)
    x13u = upconv_block(x13, 64, stage=113, block='up', strides=2) 

    x14 = layers.Concatenate(axis=-1, name='concat_114')([x04, x05u])
    x14 = conv_block(x14, 512, stage=114, block='a', drop_rate=drop_rate)
    x14 = conv_block(x14, 512, stage=114, block='b', drop_rate=drop_rate)
    x14 = conv_block(x14, 512, stage=114, block='c', drop_rate=drop_rate)
    x14 = conv_block(x14, 512, stage=114, block='d', drop_rate=drop_rate)
    x14u = upconv_block(x14, 256, stage=114, block='up', strides=2)

    #---------------------------------------------------------------------------
    # Diagonal #3 
    x12t = layers.Add()([x12t, x02])
    x22 = layers.Concatenate(axis=-1, name='concat_122')([x12t, x13u])
    x22 = conv_block(x22, 128, stage=122, block='a', strides=1, drop_rate=drop_rate)
    x22 = conv_block(x22, 128, stage=122, block='b', strides=1, drop_rate=drop_rate)
    x22t = conv_block(x22, 64, stage=122, block='tr', strides=1)
    x22u = upconv_block(x22, 32, stage=122, block='up', strides=2)   

    x13t = layers.Add()([x13t, x03])
    x23 = layers.Concatenate(axis=-1, name='concat_123')([x13t, x14u])
    x23 = conv_block(x23, 256, stage=123, block='a', strides=1, drop_rate=drop_rate)
    x23 = conv_block(x23, 256, stage=123, block='b', strides=1, drop_rate=drop_rate)
    x23 = conv_block(x23, 256, stage=123, block='c', strides=1, drop_rate=drop_rate)
    x23u = upconv_block(x23, 64, stage=123, block='up', strides=2) 

    #---------------------------------------------------------------------------
    # Diagonal #4 
    x22t = layers.Add()([x22t, x12t])
    x32 = layers.Concatenate(axis=-1, name='concat_132')([x22t, x23u])
    x32 = conv_block(x32, 128, stage=132, block='a', strides=1, drop_rate=drop_rate)
    x32 = conv_block(x32, 128, stage=132, block='b', strides=1, drop_rate=drop_rate)
    x32u = upconv_block(x32, 32, stage=132, block='up', strides=2)     
    
    
    #---------------------------------------------------------------------------
    # Output   
    x11 = layers.Conv2D(classes, 1, activation='relu', padding='same',
                        name='conv11_out')(x12u)
    x21 = layers.Conv2D(classes, 1, activation='relu', padding='same',
                        name='conv21_out')(x22u)                 
    x31 = layers.Conv2D(classes, 1, activation='relu', padding='same',
                        name='conv31_out')(x32u)   
    x10 = layers.Add()([x11, x21, x31])
    x10 = layers.Reshape((int(input_shape[0]*input_shape[1]),classes))(x10)
    x10 = layers.Activation('softmax', name="edge_output")(x10)    
    

    model = Model(inputs=inputs, outputs=x10, name='Unet_plus_plus')
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