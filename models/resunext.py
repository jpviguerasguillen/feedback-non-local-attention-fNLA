'''
Paper: DenseUNets with feedback non-local attention for the segmentation of 
       specular microscopy images of the corneal endothelium with Fuchs dystrophy

Original paper by J.P. Vigueras-Guillen
Code written by: J.P. Vigueras-Guillen
Date: 28 Feb 2022

If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at J.P.ViguerasGuillen@gmail.com

This file only contains the keras code (Tensorflow) for the network definition  
of the default ResUNeXt tested in the aformentioned paper.
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

    x1 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                      kernel_initializer='he_normal', name='conv1')(inputs)
    x1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x1)
    x1 = layers.Activation('relu')(x1)

    x2 = conv_block(x1, 3, [32, 32, 128], cardinality=8, stage=2, block='a', strides=(1, 1), drop_rate=drop_rate)
    x2 = identity_block(x2, 3, [32, 32, 128], cardinality=8, stage=2, block='b', drop_rate=drop_rate)
    x2 = identity_block(x2, 3, [32, 32, 128], cardinality=8, stage=2, block='c', drop_rate=drop_rate)

    x3 = conv_block(x2, 3, [64, 64, 256], cardinality=16, stage=3, block='a')
    x3 = identity_block(x3, 3, [64, 64, 256], cardinality=16, stage=3, block='b', drop_rate=drop_rate)
    x3 = identity_block(x3, 3, [64, 64, 256], cardinality=16, stage=3, block='c', drop_rate=drop_rate)
    x3 = identity_block(x3, 3, [64, 64, 256], cardinality=16, stage=3, block='d', drop_rate=drop_rate)

    x4 = conv_block(x3, 3, [128, 128, 512], cardinality=32, stage=4, block='a')
    x4 = identity_block(x4, 3, [128, 128, 512], cardinality=32, stage=4, block='b', drop_rate=drop_rate)
    x4 = identity_block(x4, 3, [128, 128, 512], cardinality=32, stage=4, block='c', drop_rate=drop_rate)
    x4 = identity_block(x4, 3, [128, 128, 512], cardinality=32, stage=4, block='d', drop_rate=drop_rate)
    x4 = identity_block(x4, 3, [128, 128, 512], cardinality=32, stage=4, block='e', drop_rate=drop_rate)

    x5 = conv_block(x4, 3, [256, 256, 1024], cardinality=64, stage=5, block='a')
    x5 = identity_block(x5, 3, [256, 256, 1024], cardinality=64, stage=5, block='b', drop_rate=drop_rate)
    x5 = identity_block(x5, 3, [256, 256, 1024], cardinality=64, stage=5, block='c', drop_rate=drop_rate)

    x6 = upconv_block(x5, 2, [128, 128, 512], cardinality=32, stage=6, block='a_up')   
    x6 = layers.Concatenate(axis=bn_axis, name='concat_6')([x6, x4])
    x6 = identity_block(x6, 3, [256, 256, 1024], cardinality=64, stage=6, block='b', drop_rate=drop_rate)
    x6 = identity_block(x6, 3, [256, 256, 1024], cardinality=64, stage=6, block='c', drop_rate=drop_rate)
    x6 = identity_block(x6, 3, [256, 256, 1024], cardinality=64, stage=6, block='d', drop_rate=drop_rate)
    x6 = identity_block(x6, 3, [256, 256, 1024], cardinality=64, stage=6, block='e', drop_rate=drop_rate)

    x7 = upconv_block(x6, 2, [64, 64, 256], cardinality=16, stage=7, block='a_up')   
    x7 = layers.Concatenate(axis=bn_axis, name='concat_7')([x7, x3])
    x7 = identity_block(x7, 3, [128, 128, 512], cardinality=32, stage=7, block='b', drop_rate=drop_rate)
    x7 = identity_block(x7, 3, [128, 128, 512], cardinality=32, stage=7, block='c', drop_rate=drop_rate)
    x7 = identity_block(x7, 3, [128, 128, 512], cardinality=32, stage=7, block='d', drop_rate=drop_rate)

    x8 = upconv_block(x7, 2, [32, 32, 128], cardinality=8, stage=8, block='a_up')   
    x8 = layers.Concatenate(axis=bn_axis, name='concat_8')([x8, x2])
    x8 = identity_block(x8, 3, [64, 64, 256], cardinality=16, stage=8, block='b', drop_rate=drop_rate)
    x8 = identity_block(x8, 3, [64, 64, 256], cardinality=16, stage=8, block='c', drop_rate=drop_rate)

    x9 = upconv_block(x8, 2, [16, 16, 32], cardinality=8, stage=9, block='a_up')  
    x9 = layers.Conv2D(classes, 1, activation='relu', padding='same',
                       name='conv09_out')(x9)

    x10 = layers.Reshape((int(img_row*img_col), classes))(x9)
    x10 = layers.Activation('softmax', name="edge_output")(x10)

    model = Model(inputs=inputs, outputs=x10, name='ResUneXt')
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
                 lr = 0.001,
                 data_format = 'channels_last')  
model.summary()