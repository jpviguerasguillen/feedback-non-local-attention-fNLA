'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy

Original paper by J.P. Vigueras-Guillen
Code written by: J.P. Vigueras-Guillen
Date: 28 Feb 2022

If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at J.P.ViguerasGuillen@gmail.com

This file only contains the keras code (Tensorflow) for the network definition  
of the default ResUNeXt+ tested in the aformentioned paper.
'''


import tensorflow as tf 
import tensorflow.keras.layers as layers 
import tensorflow.keras.losses as losses
import tensorflow.keras.backend as K 
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras import activations, initializers, regularizers, constraints



#-----------------------------------------------------------------------------
# Group convolution modules
#-----------------------------------------------------------------------------
# This part of the code uses a simplified version of the GroupConv2D class. 
# The original github from which it was obtained seems to be no longer available.

class GroupConv2D(tf.keras.layers.Layer):
    def __init__(self, 
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name_group=None,
                 **kwargs):
        super(GroupConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible " + 
                             "by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible " + 
                             "by the value of groups.")

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.name_group=name_group

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(
                tf.keras.layers.Conv2D(
                    filters=self.group_out_num,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activations.get(activation),
                    use_bias=use_bias,
                    kernel_initializer=initializers.get(kernel_initializer),
                    bias_initializer=initializers.get(bias_initializer),
                    kernel_regularizer=regularizers.get(kernel_regularizer),
                    bias_regularizer=regularizers.get(bias_regularizer),
                    activity_regularizer=regularizers.get(activity_regularizer),
                    kernel_constraint=constraints.get(kernel_constraint),
                    bias_constraint=constraints.get(bias_constraint),
                    name=self.name_group + '_' + str(i),
                    **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](
                inputs[:, :, :, i*self.group_in_num: (i+1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out




class GroupUpConv2D(tf.keras.layers.Layer):
    def __init__(self, 
                 input_channels,
                 output_channels,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 groups=1,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name_group=None,
                 **kwargs):
        super(GroupUpConv2D, self).__init__()

        if not input_channels % groups == 0:
            raise ValueError("The value of input_channels must be divisible " + 
                             "by the value of groups.")
        if not output_channels % groups == 0:
            raise ValueError("The value of output_channels must be divisible " + 
                             "by the value of groups.")

        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.groups = groups
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.name_group=name_group

        self.group_in_num = input_channels // groups
        self.group_out_num = output_channels // groups
        self.conv_list = []
        for i in range(self.groups):
            self.conv_list.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=self.group_out_num,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    data_format=data_format,
                    dilation_rate=dilation_rate,
                    activation=activations.get(activation),
                    use_bias=use_bias,
                    kernel_initializer=initializers.get(kernel_initializer),
                    bias_initializer=initializers.get(bias_initializer),
                    kernel_regularizer=regularizers.get(kernel_regularizer),
                    bias_regularizer=regularizers.get(bias_regularizer),
                    activity_regularizer=regularizers.get(activity_regularizer),
                    kernel_constraint=constraints.get(kernel_constraint),
                    bias_constraint=constraints.get(bias_constraint),
                    name=self.name_group + '_' + str(i),
                    **kwargs))

    def call(self, inputs, **kwargs):
        feature_map_list = []
        for i in range(self.groups):
            x_i = self.conv_list[i](
                inputs[:, :, :, i*self.group_in_num: (i+1) * self.group_in_num])
            feature_map_list.append(x_i)
        out = tf.concat(feature_map_list, axis=-1)
        return out




#-----------------------------------------------------------------------------
# ResNeXt modules
#-----------------------------------------------------------------------------


def identity_block(input_tensor, kernel_size, filters, cardinality, stage, block, 
                   drop_rate=None):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: tensor, 4D (N,H,W,C)
        kernel_size:  the kernel size of middle conv layer at main path              
        filters:      list of integers, the filters of 3 conv layer at main path
        cardinality:  int, the cardinality in the group convolution, i.e. how 
                      many feature maps are grouped together for each group 
                      convolution in the middle conv layer.
        stage:        integer, current stage label (to generate layer names)
        block:        string, current block label (to generate layer names)
        drop_rate:    float, the dropout rate (0-1), or None if not used.
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    act_name_base = 'act' + str(stage) + block + '_branch'
    drop_name_base = 'drop' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('elu', name=act_name_base + '2a')(x)

    x = GroupConv2D(input_channels=filters1, output_channels=filters2,
                    kernel_size=kernel_size, strides=(1, 1), padding='same', 
                    groups=cardinality, kernel_initializer='he_normal',
                    name_group=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('elu', name=act_name_base + '2b')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.Add()([x, input_tensor])
    x = layers.Activation('elu', name=act_name_base + '1')(x)
    if drop_rate is not None:
        x = layers.Dropout(rate=drop_rate, name=drop_name_base + '1')(x)
    return x





def conv_block(input_tensor, kernel_size, filters, cardinality, stage, block, 
               strides=(2, 2), drop_rate=None):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: tensor, 4D (N,H,W,C)
        kernel_size:  the kernel size of middle conv layer at main path.              
        filters:      list of integers, the filters of 3 conv layer at main path
        cardinality:  int, the cardinality in the group convolution, i.e. how 
                      many feature maps are grouped together for each group 
                      convolution in the middle conv layer.
        stage:        integer, current stage label (to generate layer names)
        block:        string, current block label (to generate layer names)
        strides:      int or tuple of ints, the strides for the mid conv layer 
                      and shortcut.
        drop_rate:    float, the dropout rate (0-1), or None if not used. 
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    act_name_base = 'act' + str(stage) + block + '_branch'
    drop_name_base = 'drop' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=(1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu', name=act_name_base + '2a')(x)
  
    x = GroupConv2D(input_channels=filters1, output_channels=filters2,
                    kernel_size=kernel_size, strides=strides, padding='same', 
                    groups=cardinality, kernel_initializer='he_normal',
                    name_group=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu', name=act_name_base + '2b')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, 
                                         name=bn_name_base + '1')(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu', name=act_name_base + '1')(x)
    if drop_rate is not None:
        x = layers.Dropout(rate=drop_rate, name=drop_name_base + '1')(x)
    return x




def upconv_block(input_tensor, kernel_size, filters, cardinality, stage, block, 
                 drop_rate=None):
    """A resnext block for upsampling. The transpose convolution is made with 
    group convolutions.
    # Arguments
        input_tensor: tensor, 4D (N,H,W,C)
        kernel_size:  the kernel size of middle conv layer at main path              
        filters:      list of integers, the filters of 3 conv layer at main path
        cardinality:  int, the cardinality in the group convolution, i.e. how 
                      many feature maps are grouped together for each group 
                      convolution in the middle conv layer.
        stage:        integer, current stage label (to generate layer names)
        block:        string, current block label (to generate layer names)
        drop_rate:    float, the dropout rate (0-1), or None if not used. 
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3 
    conv_name_base = 'up' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    act_name_base = 'act' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=(1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu', name=act_name_base + '2a')(x)
  
    x = GroupUpConv2D(input_channels=filters1, output_channels=filters2,
                      kernel_size=kernel_size, strides=(2, 2), padding='same', 
                      groups=cardinality, kernel_initializer='he_normal',
                      name_group=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu', name=act_name_base + '2b')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=(1, 1),
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis, 
                                         name=bn_name_base + '1')(shortcut)
    shortcut = layers.UpSampling2D(size=(2, 2))(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu', name=act_name_base + '1')(x)
    if drop_rate is not None:
        x = layers.Dropout(rate=drop_rate, name=drop_name_base + '1')(x)
    return x





#-----------------------------------------------------------------------------
# THE BACKBONE OF THE NETWORK 
#-----------------------------------------------------------------------------
 

def ResUNeXt(input_shape, classes, drop_rate=None, weights=None, lr=0.001, 
             data_format='channels_last'):
    """ Instantiates the DenseUNet architecture. Optionally loads weights.
    # Arguments
        input_shape:  tuple, (H,W,C)        
        classes:      int, number of classes to classify images into, only to  
                      be specified if no `weights` argument is specified.  
        drop_rate:    float, the dropout rate (0-1)              
        weights:      one of `None` (random initialization), or the path to the 
                      weights file to be loaded.
        lr:           float, learning rate of the optimizer.
        data_format:  string, to specify the place of the channels. 
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
    
    # The code assumes data_format=='channels_last'!!!
    # In this version, the argument 'data_format' is irrelevant (TO UPDATE).
    bn_axis = 3 
    img_row = input_shape[0]
    img_col = input_shape[1]
      
      
    inputs = layers.Input(shape=input_shape)

    x01 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                      kernel_initializer='he_normal', name='conv1')(inputs)
    x01 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x01)
    x01 = layers.Activation('relu')(x01)

    #---------------------------------------------------------------------------
    # Diagonal #1 (left) 
    x02 = conv_block(x01, 3, [32, 32, 128], cardinality=8, stage=102, block='a', strides=(1, 1), drop_rate=drop_rate)
    x02 = identity_block(x02, 3, [32, 32, 128], cardinality=8, stage=102, block='b', drop_rate=drop_rate)

    x03 = conv_block(x02, 3, [64, 64, 256], cardinality=16, stage=103, block='a')
    x03 = identity_block(x03, 3, [64, 64, 256], cardinality=16, stage=103, block='b', drop_rate=drop_rate)
    x03 = identity_block(x03, 3, [64, 64, 256], cardinality=16, stage=103, block='c', drop_rate=drop_rate)
    x03u = upconv_block(x03, 2, [32, 32, 128], cardinality=8, stage=103, block='up')

    x04 = conv_block(x03, 3, [128, 128, 512], cardinality=32, stage=104, block='a')
    x04 = identity_block(x04, 3, [128, 128, 512], cardinality=32, stage=104, block='b', drop_rate=drop_rate)
    x04 = identity_block(x04, 3, [128, 128, 512], cardinality=32, stage=104, block='c', drop_rate=drop_rate)
    x04 = identity_block(x04, 3, [128, 128, 512], cardinality=32, stage=104, block='d', drop_rate=drop_rate)
    x04u = upconv_block(x04, 2, [64, 64, 256], cardinality=16, stage=104, block='up')

    x05 = conv_block(x04, 3, [256, 256, 1024], cardinality=64, stage=105, block='a')
    x05 = identity_block(x05, 3, [256, 256, 1024], cardinality=64, stage=105, block='b', drop_rate=drop_rate)
    x05 = identity_block(x05, 3, [256, 256, 1024], cardinality=64, stage=105, block='c', drop_rate=drop_rate)
    x05u = upconv_block(x05, 2, [128, 128, 512], cardinality=32, stage=105, block='up')


    #---------------------------------------------------------------------------
    # Diagonal #2 
    x12 = layers.Concatenate(axis=bn_axis, name='concat_12')([x02, x03u])
    x12 = identity_block(x12, 3, [64, 64, 256], cardinality=16, stage=112, block='b', drop_rate=drop_rate)
    x12 = identity_block(x12, 3, [64, 64, 256], cardinality=16, stage=112, block='c', drop_rate=drop_rate)
    x12u = upconv_block(x12, 2, [16, 16, 32], cardinality=8, stage=112, block='up')  
    x12t = conv_block(x12, 3, [32, 32, 128], cardinality=8, stage=112, block='tr', strides=(1, 1))

    x13 = layers.Concatenate(axis=bn_axis, name='concat_113')([x03, x04u])
    x13 = identity_block(x13, 3, [128, 128, 512], cardinality=32, stage=113, block='a', drop_rate=drop_rate)
    x13 = identity_block(x13, 3, [128, 128, 512], cardinality=32, stage=113, block='b', drop_rate=drop_rate)
    x13 = identity_block(x13, 3, [128, 128, 512], cardinality=32, stage=113, block='c', drop_rate=drop_rate)
    x13u = upconv_block(x13, 2, [32, 32, 128], cardinality=8, stage=113, block='up')
    x13t = conv_block(x13, 3, [64, 64, 256], cardinality=16, stage=113, block='tr', strides=(1, 1))
    
    x14 = layers.Concatenate(axis=bn_axis, name='concat_114')([x04, x05u])
    x14 = identity_block(x14, 3, [256, 256, 1024], cardinality=64, stage=114, block='a', drop_rate=drop_rate)
    x14 = identity_block(x14, 3, [256, 256, 1024], cardinality=64, stage=114, block='b', drop_rate=drop_rate)
    x14 = identity_block(x14, 3, [256, 256, 1024], cardinality=64, stage=114, block='c', drop_rate=drop_rate)
    x14 = identity_block(x14, 3, [256, 256, 1024], cardinality=64, stage=114, block='d', drop_rate=drop_rate)
    x14u = upconv_block(x14, 2, [64, 64, 256], cardinality=16, stage=114, block='up')


    #---------------------------------------------------------------------------
    # Diagonal #3  
    x22 = layers.Concatenate(axis=bn_axis, name='concat_122')([x12t, x13u])
    x22 = identity_block(x22, 3, [64, 64, 256], cardinality=16, stage=122, block='a', drop_rate=drop_rate)
    x22 = identity_block(x22, 3, [64, 64, 256], cardinality=16, stage=122, block='b', drop_rate=drop_rate)
    x22u = upconv_block(x22, 2, [16, 16, 32], cardinality=8, stage=122, block='up') 
    x22t = conv_block(x22, 3, [32, 32, 128], cardinality=8, stage=122, block='tr', strides=(1, 1))    
    
    x23 = layers.Concatenate(axis=bn_axis, name='concat_123')([x13t, x14u])
    x23 = identity_block(x23, 3, [128, 128, 512], cardinality=32, stage=123, block='a', drop_rate=drop_rate)
    x23 = identity_block(x23, 3, [128, 128, 512], cardinality=32, stage=123, block='b', drop_rate=drop_rate)
    x23 = identity_block(x23, 3, [128, 128, 512], cardinality=32, stage=123, block='c', drop_rate=drop_rate)
    x23u = upconv_block(x23, 2, [32, 32, 128], cardinality=8, stage=123, block='up')


    #---------------------------------------------------------------------------
    # Diagonal #4  
    x32 = layers.Concatenate(axis=bn_axis, name='concat_132')([x22t, x23u])
    x32 = identity_block(x32, 3, [64, 64, 256], cardinality=16, stage=132, block='a', drop_rate=drop_rate)
    x32 = identity_block(x32, 3, [64, 64, 256], cardinality=16, stage=132, block='b', drop_rate=drop_rate)
    x32u = upconv_block(x32, 2, [16, 16, 32], cardinality=8, stage=132, block='up')  
    

    #---------------------------------------------------------------------------
    # Output   
    x11 = layers.Conv2D(classes, 1, activation='relu', padding='same',
                        name='conv11_out', data_format=data_format)(x12u)
    x21 = layers.Conv2D(classes, 1, activation='relu', padding='same',
                        name='conv21_out', data_format=data_format)(x22u)                 
    x31 = layers.Conv2D(classes, 1, activation='relu', padding='same',
                        name='conv31_out', data_format=data_format)(x32u)   
    x10 = layers.Add()([x11, x21, x31])

    x10 = layers.Reshape((int(img_row*img_col), classes))(x10)
    x10 = layers.Activation('softmax', name="edge_output")(x10)

    model = Model(inputs=inputs, outputs=x10, name='ResUneXt_plus')
    if weights is not None:
        model.load_weights(weights)
    nadam = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    model.compile(optimizer=nadam, 
                  loss= 'categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model


# This model is for CNN-Edge because it has only 1 feature map in the input.
# For the CNN-Body, it should be 2 feature maps
model = ResUNeXt(input_shape = (528, 240, 1), 
                 classes = 2, 
                 drop_rate = 0.2,
                 lr = 0.001,
                 data_format = 'channels_last')  
model.summary()