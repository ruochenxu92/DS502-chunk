import mxnet as mx
import numpy as np
# import cv2
import logging
from data import get_mnist
from mlp_sym import get_mlp_sym
from cnn_sym import get_cnn_sym
logging.getLogger().setLevel(logging.DEBUG)  # logging to stdout

def train_mlp():
    # Get the data iterator
    batch_size = 100
    train_iter, val_iter = get_mnist(batch_size)

    # Get MLP symbol
    mlp_model = get_mlp_sym()
    # Viz the graph and save the plot for debugging
    plot = mx.viz.plot_network(mlp_model, title="mlp", save_format="pdf", hide_weights=True)
    plot.render("MLP")

    model = mx.mod.Module(symbol=mlp_model, context=mx.cpu())
    model.fit(train_iter,  # train data
              eval_data=val_iter,  # validation data
              optimizer='sgd',  # use SGD to train
              optimizer_params={'learning_rate': 0.1},  # use fixed learning rate
              eval_metric='acc',  # report accuracy during training
              batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              # output progress for each 100 data batches
              num_epoch=5)


def train_cnn():
    batch_size = 100
    train_iter, val_iter = get_mnist(batch_size)

    cnn_model = get_cnn_sym()
    plot = mx.viz.plot_network(cnn_model, title="cnn", save_format="pdf", hide_weights=True)
    plot.render("CNN")

    model = mx.mod.Module(symbol=cnn_model, context=mx.cpu())

    model.fit(train_iter,
              eval_data = val_iter,
              optimizer="sgd",
              optimizer_params={ "learning_rate": 0.1 },
              eval_metric="acc",
              batch_end_callback=mx.callback.Speedometer(batch_size, 100),
              # output progress for each 100 data batches
              num_epoch=5)


if __name__=="__main__":
    train_mlp()
    # train_cnn()
