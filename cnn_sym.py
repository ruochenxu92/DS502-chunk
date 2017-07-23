import mxnet as mx

from mlp_sym import mlp_layer

def conv_layer(
        input_layer,
        kernel_size=(3, 3),
        stride=(1, 1),
        pad=(0, 0),
        num_filter=32,
        act_type="relu",
        pooling=True,
        pool_type='max',
        BN=True,
        ):
    """
    :return: a single convolution layer symbol
    """
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?

    # get a FC layer
    layer = mx.sym.Convolution(data=input_layer,
                               kernel=kernel_size,
                               stride=stride,
                               pad=pad,
                               num_filter=num_filter,
                               )
    if BN:
        layer = mx.sym.BatchNorm(layer)

    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if act_type is not None:
        layer = mx.sym.Activation(data=layer, act_type=act_type)

    if pooling:
        layer = mx.sym.Pooling(data=layer, kernel=kernel_size, pool_type=pool_type, stride=stride)

    return layer


# Optional
def inception_layer():
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    pass


def get_cnn_sym():

    """
    :return: symbol of a convolutional neural network
    """
    l = mx.sym.Variable("data")

    BN = True

    l = conv_layer(input_layer=l, num_filter=64, act_type="relu", BN=BN, pooling=True)

    l = conv_layer(input_layer=l, num_filter=128, act_type="relu", BN=BN, pooling=True)

    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=BN)

    l = mx.sym.FullyConnected(data=l, num_hidden=10)

    return mx.sym.SoftmaxOutput(data=l, name='softmax')



