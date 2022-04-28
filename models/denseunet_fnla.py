'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy

Original paper by J.P. Vigueras-Guillen
Code written by: J.P. Vigueras-Guillen
Date: 28 Feb 2022

If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at J.P.ViguerasGuillen@gmail.com

This file only contains the keras code (Tensorflow) for the network definition  
of the DenseUNet with fNLA (attention) tested in the aformentioned paper.
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
# Attention Block
#-----------------------------------------------------------------------------

def attention_NLA(x, ichannels, g=None, type_att='mul', name=None):
    """ An attention block inspired from the paper 'Wang 2018 - Non-local neural 
    networks'. Several options are available:
    - If a tensor from a lower-level is given, this is used to build phi and gee,  
      otherwise the default Wang version is used.
    - The attention maps can be multiplied (reduced to only one map), summed 
      (expanded to the same number of feature maps), or simply concatenated to 
      the input.
    # Arguments
        x:            input tensor, where attention is applied.
        ichannels:    float, number of channels in the first reduction.
        g:            lower-level input tensor, used to determine where to pay 
                      attention. 
        type_att:     how the attention is applied: 'sum', 'mul', or 'conc'.
        name:         string, block label.
    # Returns
        Output tensor for the block.
    """  
    flag_sNLA = True if g is None else False
    
    # Apply conv1x1 to create 3 different tensors
    the_x = layers.Conv2D(ichannels, 1, strides=1, name=name + '_conv_the_x')(x)
    _,h,w,c = the_x.shape  
    if flag_sNLA:
        phi_x = layers.Conv2D(ichannels, 1, strides=1, name=name + '_conv_phi_x')(x)
        gee_x = layers.Conv2D(ichannels, 1, strides=1, name=name + '_conv_gee_x')(x)
    else:
        phi_x = layers.Conv2DTranspose(ichannels, 2, strides=2, name=name + '_conv_phi_x')(g)
        gee_x = layers.Conv2DTranspose(ichannels, 2, strides=2, name=name + '_conv_gee_x')(g)

    # Reshape the 3 tensors: 
    # the_x and gee_x become (Batch, C, HW). phi_x becomes (Batch, HW, C)
    the_x = layers.Reshape((int(h*w),c), name=name + '_reshape_the_x')(the_x)
    phi_x = layers.Reshape((int(h*w),c), name=name + '_reshape_phi_x')(phi_x)
    gee_x = layers.Reshape((int(h*w),c), name=name + '_reshape_gee_x')(gee_x)
    the_x = layers.Permute((2,1), name=name + '_permute_the_x')(the_x)
    gee_x = layers.Permute((2,1), name=name + '_permute_gee_x')(gee_x)

    # Apply matrix multiplication - zet_x becomes (Batch, HW, C)
    eta_x = tf.matmul(the_x, phi_x)
    eta_x = layers.Activation('softmax', name=name + '_actv_eta_x')(eta_x)
    zet_x = tf.matmul(eta_x, gee_x)

    # Reshape to return to original dimensions -> (Batch, H, W, C)
    zet_x = layers.Reshape((h,w,c), name=name + '_reshape_zet_x')(zet_x)
    
    # Select the type of attention
    if type_att == 'sum':
        att_x = layers.Conv2D(x.shape[3], 1, strides=1, name=name + '_conv_att_x')(zet_x)
        att_x = layers.Activation('relu', name=name + '_actv_relu')(att_x)
        psi_x = layers.Add(name=name + '_add_psi_x')([x, att_x])
    elif type_att == 'mul':
        att_x = layers.Conv2D(1, 1, strides=1, name=name + '_conv_att_x')(zet_x)
        att_x = layers.Activation('sigmoid', name=name + '_actv_sigmoid')(att_x)
        psi_x = layers.Multiply(name=name + '_mul_psi_x')([x, att_x])
    elif type_att == 'conc':
        att_x = layers.Activation('elu', name=name + '_actv_elu')(zet_x)
        psi_x = layers.Concatenate(name=name + '_conc_psi_x')([x, att_x])
    return psi_x




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
    # --- DOWNSAMPLING -----------------------------------------------------
    x00 = dense_1st_block(xi0, blocks[0], growth_rate, name='conv00')
    x00D = downsampling_block(x00, tran_channels[0], name='down00')
        
    x10 = dense_block(x00D, blocks[1], growth_rate, name='conv10', drop_rate=drop_rate)
    x10D = downsampling_block(x10, tran_channels[1], name='down10')
   
    x20 = dense_block(x10D, blocks[2], growth_rate, name='conv20', drop_rate=drop_rate)
    x20D = downsampling_block(x20, tran_channels[2], name='down20')
 
    x30 = dense_block(x20D, blocks[3], growth_rate, name='conv30', drop_rate=drop_rate)
    x30D = downsampling_block(x30, tran_channels[3], name='down30')

    x40 = dense_block(x30D, blocks[4], growth_rate, name='conv40', drop_rate=drop_rate)
    
    
    # --- Feedback fNLA
    x40 = attention_NLA(x40, x40.shape[-1]//8, type_att='mul', name='attn40')
    x40U = upsampling_block(x40, tran_channels[4], growth_rate, name='upsa40')
    
    x30 = attention_NLA(x30, x30.shape[-1]//8, x40, type_att='mul', name='attn30')
    x30T = transition_block(x30, tran_channels[3], name='tran30')
    
    x20 = attention_NLA(x20, x20.shape[-1]//8, x30, type_att='mul', name='attn20')
    x20T = transition_block(x20, tran_channels[2], name='tran20')
    
    x10 = attention_NLA(x10, x10.shape[-1]//8, x20, type_att='mul', name='attn10')
    x10T = transition_block(x10, tran_channels[1], name='tran10') 
    
    x00 = attention_NLA(x00, x00.shape[-1]//8, x10, type_att='mul', name='attn00')   
    x00T = transition_block(x00, tran_channels[0], name='tran00')
    
    
    # ======================================================================== 
    # --- UPSAMPLING --------------------------------------------------------
    x31 = layers.Concatenate(name='conc31')([x30T,x40U])
    x31 = dense_block(x31, blocks[3], growth_rate, name='conv31', drop_rate=drop_rate)
    x31 = attention_NLA(x31, x31.shape[-1]//8, type_att='mul', name='attn31')
    x31U = upsampling_block(x31, tran_channels[3], growth_rate, name='upsa31')
    
    x22 = layers.Concatenate(name='conc22')([x20T,x31U])
    x22 = dense_block(x22, blocks[2], growth_rate, name='conv22', drop_rate=drop_rate)
    x22 = attention_NLA(x22, x22.shape[-1]//8, type_att='mul', name='attn22')
    x22U = upsampling_block(x22, tran_channels[2], growth_rate, name='upsa22')
    
    x13 = layers.Concatenate(name='conc13')([x10T,x22U])
    x13 = dense_block(x13, blocks[1], growth_rate, name='conv13', drop_rate=drop_rate)
    x13 = attention_NLA(x13, x13.shape[-1]//8, type_att='mul', name='attn13')
    x13U = upsampling_block(x13, tran_channels[1], growth_rate, name='upsa13')

    x04 = layers.Concatenate(name='conc04')([x00T,x13U])
    x04 = dense_block(x04, blocks[0], growth_rate, name='conv04')
    x04 = attention_NLA(x04, x04.shape[-1]//8, type_att='mul', name='attn04')
    
    
    # ======================================================================== 
    # --- OUTPUT-------------------------------------------------------------
    x05 = transition_block(x04, classes, name='conv05', activation='elu')
    x05 = layers.Reshape((int(input_shape[0]*input_shape[1]),classes))(x05)
    x05 = layers.Activation('softmax', name='conv05_softmax')(x05)

    model = Model(inputs=xi0, outputs=x05, name='denseUnet_fNLA')
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
                  growth_rate = 5, 
                  drop_rate = 0.2,
                  classes = 2, 
                  lr = 0.001)  
model.summary() 


