'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy

Original paper by J.P. Vigueras-Guillen
Code written by: J.P. Vigueras-Guillen
Date: 28 Feb 2022

If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at J.P.ViguerasGuillen@gmail.com

This file only contains the keras code (Tensorflow) for the network definition  
of the default DenseUNet++ Plus Plus tested in the aformentioned paper.
'''


import tensorflow as tf 
import tensorflow.keras.layers as layers 
import tensorflow.keras.losses as losses
import tensorflow.keras.backend as K 
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Nadam



#-----------------------------------------------------------------------------
# Basic Blocks 
#-----------------------------------------------------------------------------

def dense_1st_block(x, blocks, growth_rate, name, drop_rate=None, 
                    activation='elu'):
    """The first dense block, without Conv1x1 (feature reduction).
    # Arguments
        x:           input tensor.
        blocks:      integer, the number of building blocks.
        growth_rate: float, growth rate at dense layers (output maps).
        name:        string, block label.
        drop_rate:   float, the dropout rate (0-1), or None if not used.
        activation:  string, the type of activation
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x1 = layers.Conv2D(growth_rate, 3, padding='same', 
                           use_bias=False, activation=None,
                           name=name + '_block' + str(i+1) + '_conv')(x)
        x1 = layers.BatchNormalization(epsilon=1.001e-5, renorm=True, 
                                       name=name + '_block' + str(i+1) + '_bn')(x1)  
        x1 = layers.Activation(activation, 
                               name=name + '_block' + str(i+1) + '_actv')(x1)
        if drop_rate is not None:
            x1 = layers.Dropout(rate=drop_rate, name=name + str(i+1) + '_drop')(x1)
        x = layers.Concatenate(name=name + '_block' + str(i+1) + '_concat')([x, x1])
    return x


def dense_block(x, blocks, growth_rate, name, drop_rate=None, activation='elu'):
    """A dense block. It constitutes several conv_blocks
    # Arguments
        x:           input tensor.
        blocks:      integer, the number of building blocks.
        growth_rate: float, growth rate at dense layers (output maps).
        name:        string, block label.
        drop_rate:   float, the dropout rate (0-1), or None if not used.
        activation:  string, the type of activation
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, growth_rate, name=name + '_block' + str(i+1), 
                       drop_rate=drop_rate, activation=activation)
    return x


def conv_block(x, growth_rate, name, drop_rate=None, activation='elu'):
    """ A building block for a dense block.
    # Arguments
        x:           input tensor.
        growth_rate: float, growth rate at dense layers.
        name:        string, block label.
        drop_rate:   float, the dropout rate (0-1), or None if not used.
        activation:  string, the type of activation
    # Returns
        Output tensor for the block.
    """
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, activation=None,
                       name=name + '_1_conv')(x)
    x1 = layers.BatchNormalization(epsilon=1.001e-5, renorm=True,
                                   name=name + '_1_bn')(x1) #
    x1 = layers.Activation(activation, name=name + '_1_actv')(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding='same', use_bias=False,
                       activation=None, name=name + '_2_conv')(x1)
    x1 = layers.BatchNormalization(epsilon=1.001e-5, renorm=True, 
                                   name=name + '_2_bn')(x1) 
    x1 = layers.Activation(activation, name=name + '_2_actv')(x1)
    if drop_rate is not None:
        x1 = layers.Dropout(rate=drop_rate, name=name + '_drop')(x1)
    x = layers.Concatenate(name=name + '_concat')([x, x1])
    return x
  

def transition_block(x, out_channels, name, activation='elu'):
    """ A transition block, at the end of the dense block, without including 
    the downsampling.
    # Arguments
        x:            input tensor.
        out_channels: int, the number of feature maps in the convolution.
        name:         string, block label.
        activation:   string, the type of activation
    # Returns
        output tensor for the block.
    """
    x = layers.Conv2D(out_channels, 1, activation=None,
                      use_bias=False, name=name + '_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, renorm=True, 
                                  name=name + '_bn')(x) #
    x = layers.Activation(activation, name=name + '_actv')(x)
    return x


def downsampling_block(x, out_channels, name, activation='elu'):
    """ An upsampling block with tranpose convolutions.
    # Arguments
        x:            input tensor.
        growth_rate:  float, growth rate at the first convolution layer.
        out_channels: int, the number of feature maps in the convolution.
        name:         string, block label.
        activation:   string, the type of activation
    # Returns
        output tensor for the block.
    """
    x = layers.Conv2D(out_channels, 2, activation=None, strides=2, 
                      padding='same', use_bias=False, 
                      name=name + '_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, renorm=True, 
                                  name=name + '_bn')(x) 
    x = layers.Activation(activation, name=name + '_actv')(x)
    return x

	  
def upsampling_block(x, out_channels, growth_rate, name, activation='elu'):
    """ An upsampling block with tranpose convolutions.
    # Arguments
        x:            input tensor.
        growth_rate:  float, growth rate at the first convolution layer.
        out_channels: int, the number of feature maps in the convolution.
        name:         string, block label.
        activation:   string, the type of activation
    # Returns
        output tensor for the block.
    """
    x = layers.Conv2DTranspose(out_channels, 2, activation=None, strides=2, 
                               padding='same', use_bias=False, 
                               name=name + '_conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, renorm=True, 
                                  name=name + '_bn')(x) #
    x = layers.Activation(activation, name=name + '_actv')(x)
    return x





#-----------------------------------------------------------------------------
# THE BACKBONE OF THE NETWORK 
#-----------------------------------------------------------------------------
 
def denseUnet(blocks, input_shape, growth_rate, drop_rate, classes, weights=None, 
              lr=0.001):
    """ Instantiates the DenseUNet architecture. Optionally loads weights.
    # Arguments
        blocks:       numbers of building blocks for the four dense layers.
        input_shape:  tuple, (H,W,C)    
        growth_rate:  int, the number of feature maps in each convolution within 
                      the dense blocks.      
        drop_rate:    float, the dropout rate (0-1), or None if not used.                       
        classes:      int, number of classes to classify images into, only to  
                      be specified if no `weights` argument is specified.   
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

    tran_channels = [int(x*growth_rate*0.50) for x in blocks]

    # --- Input    
    xi0 = layers.Input(shape=input_shape)
        
    
    # ======================================================================== 
    # --- FIRST DIAGONAL -----------------------------------------------------
    x00 = dense_1st_block(xi0, blocks[0], growth_rate, name='conv00')
    x00T = transition_block(x00, tran_channels[0], name='tran00')
    x00D = downsampling_block(x00, tran_channels[0], name='down00')
     
    # ---  
    x10 = dense_block(x00D, blocks[1], growth_rate, name='conv10', drop_rate=drop_rate)
    x10T = transition_block(x10, tran_channels[1], name='tran10')
    x10D = downsampling_block(x10, tran_channels[1], name='down10')
    x10U = upsampling_block(x10, tran_channels[1], growth_rate, name='upsa10')

    # ---     
    x20 = dense_block(x10D, blocks[2], growth_rate, name='conv20', drop_rate=drop_rate)
    x20T = transition_block(x20, tran_channels[2], name='tran20')
    x20D = downsampling_block(x20, tran_channels[2], name='down20')
    x20U = upsampling_block(x20, tran_channels[2], growth_rate, name='upsa20')

    # ----  
    x30 = dense_block(x20D, blocks[3], growth_rate, name='conv30', drop_rate=drop_rate)
    x30T = transition_block(x30, tran_channels[3], name='tran30')
    x30D = downsampling_block(x30, tran_channels[3], name='down30')
    x30U = upsampling_block(x30, tran_channels[3], growth_rate, name='upsa30')

    # ---
    x40 = dense_block(x30D, blocks[4], growth_rate, name='conv40', drop_rate=drop_rate)
    x40U = upsampling_block(x40, tran_channels[4], growth_rate, name='upsa40')
    
    
    # ======================================================================== 
    # --- SECOND DIAGONAL ----------------------------------------------------
    x01 = layers.Concatenate(name='conc01')([x00T,x10U]) 
    x01 = dense_block(x01, blocks[0], growth_rate, name='conv01')
    x01T = transition_block(x01, tran_channels[0], name='tran01')

    # ---
    x11 = layers.Concatenate(name='conc11')([x10T,x20U]) 
    x11 = dense_block(x11, blocks[1], growth_rate, name='conv11', drop_rate=drop_rate)
    x11T = transition_block(x11, tran_channels[1], name='tran11')
    x11U = upsampling_block(x11, tran_channels[1], growth_rate, name='upsa11')

    # ---
    x21 = layers.Concatenate(name='conc21')([x20T,x30U]) 
    x21 = dense_block(x21, blocks[2], growth_rate, name='conv21', drop_rate=drop_rate)
    x21T = transition_block(x21, tran_channels[2], name='tran21')
    x21U = upsampling_block(x21, tran_channels[2], growth_rate, name='upsa21')

    # ---
    x31 = layers.Concatenate(name='conc31')([x30T,x40U])
    x31 = dense_block(x31, blocks[3], growth_rate, name='conv31', drop_rate=drop_rate)
    x31U = upsampling_block(x31, tran_channels[3], growth_rate, name='upsa31')
    
    
    # ======================================================================== 
    # --- THIRD DIAGONAL ----------------------------------------------------
    x02P = layers.Add()([x01T, x00T])
    x02 = layers.Concatenate(name='conc02')([x02P,x11U]) 
    x02 = dense_block(x02, blocks[0], growth_rate, name='conv02')
    x02T = transition_block(x02, tran_channels[0], name='tran02')

    # ---
    x12P = layers.Add()([x11T, x10T])
    x12 = layers.Concatenate(name='conc12')([x12P,x21U]) 
    x12 = dense_block(x12, blocks[1], growth_rate, name='conv12', drop_rate=drop_rate)
    x12T = transition_block(x12, tran_channels[1], name='tran12')
    x12U = upsampling_block(x12, tran_channels[1], growth_rate, name='upsa12')

    # ---
    x22P = layers.Add()([x21T, x20T])
    x22 = layers.Concatenate(name='conc22')([x22P,x31U])
    x22 = dense_block(x22, blocks[2], growth_rate, name='conv22', drop_rate=drop_rate)
    x22U = upsampling_block(x22, tran_channels[2], growth_rate, name='upsa22')
    

    # ======================================================================== 
    # --- FORTH DIAGONAL ----------------------------------------------------
    x03P = layers.Add()([x02T, x01T, x00T])
    x03 = layers.Concatenate(name='conc03')([x03P,x12U]) 
    x03 = dense_block(x03, blocks[0], growth_rate, name='conv03')
    x03T = transition_block(x03, tran_channels[0], name='tran03')

    # ---
    x13P = layers.Add()([x12T, x11T, x10T])
    x13 = layers.Concatenate(name='conc13')([x13P,x22U])
    x13 = dense_block(x13, blocks[1], growth_rate, name='conv13', drop_rate=drop_rate)
    x13U = upsampling_block(x13, tran_channels[1], growth_rate, name='upsa13')


    # ======================================================================== 
    # --- FIFTH DIAGONAL ----------------------------------------------------
    x04P = layers.Add()([x03T, x02T, x01T, x00T])
    x04 = layers.Concatenate(name='conc04')([x04P,x13U])
    x04 = dense_block(x04, blocks[0], growth_rate, name='conv04')
    
    
    # ======================================================================== 
    # --- OUTPUT-------------------------------------------------------------
    x05_01 = transition_block(x01, classes, name='conv05_01', activation='elu')
    x05_02 = transition_block(x02, classes, name='conv05_02', activation='elu')
    x05_03 = transition_block(x03, classes, name='conv05_03', activation='elu')
    x05_04 = transition_block(x04, classes, name='conv05_04', activation='elu')
    x05 = layers.Add()([x05_01, x05_02, x05_03, x05_04])
    x05 = layers.Reshape((int(input_shape[0]*input_shape[1]),classes))(x05)
    x05 = layers.Activation('softmax')(x05)

    model = Model(inputs=xi0, outputs=x05, name='denseUnetPlusPlus')
    if weights is not None:
        model.load_weights(weights)
    nadam = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam, 
                  loss= 'categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model


#------------------------------------------------------------------------------
# Load the model
# This model is for CNN-Edge because it has only 1 feature map in the input.
# For the CNN-Body, it should be 2 feature maps.
model = denseUnet(blocks = (4, 8, 12, 16, 20), 
                  input_shape = (528, 240, 1), 
                  growth_rate = 4, 
                  drop_rate = 0.2,
                  classes = 2, 
                  lr = 0.001)  
model.summary() 

