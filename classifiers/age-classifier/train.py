#!/usr/bin/env python

import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import cnn
import random
from dataset import Dataset
from lasagne.layers import NonlinearityLayer
from lasagne.nonlinearities import softmax

num_classes = 101
batch_size = 16
top_k = 5

def build_vgg16(input_var=None):
    network = cnn.build_model(input_var)['fc8']
    network = NonlinearityLayer(network, softmax)

    network = lasagne.layers.DimshuffleLayer(network, (1, 0))
    network = lasagne.layers.FlattenLayer(network)
    network = lasagne.layers.DimshuffleLayer(network, (1, 0))
    return network

def build_targets_formatter(input_var=None):
    network = lasagne.layers.InputLayer(shape=(batch_size, num_classes), input_var=input_var)
    network = lasagne.layers.FlattenLayer(network, outdim=1)

    return network

def freeze_early_weights(network):
    layers = lasagne.layers.get_all_layers(network)
    for i in range(len(layers)-6):
        layer = layers[i]
        params = layer.get_params()
        if len(params) > 0:
            layer.params[layer.W].remove("trainable")
            layer.params[layer.b].remove("trainable")
    return network

def main(num_epochs=500):
    # Loading dataset
    print "Loading dataset"
    random.seed(123) 
    mean = np.array([103.939, 116.779, 123.68])
    dataset = Dataset("data", 224,  batch_size, mean)
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    # Build CNN model
    print("Building model and compiling functions...")
    network = build_vgg16(input_var)

    # # Load the weights obtained earlier
    # base_data = np.load('snapshot.npz')
    # base_data = base_data[base_data.keys()[0]]
    # lasagne.layers.set_all_param_values(network, base_data)


    print("(sanity check) Trainable layers before: {}".format(len(lasagne.layers.get_all_params(network, trainable="True"))) )
    network = freeze_early_weights(network)
    print("(sanity check) Trainable layers after: {}".format(len(lasagne.layers.get_all_params(network, trainable="True"))) )
    target_formatter = build_targets_formatter(target_var)

    # Expressions for training losses
    prediction = lasagne.layers.get_output(network)
    targets = T.cast( lasagne.layers.get_output(target_formatter), 'int32')
    train_loss = lasagne.objectives.categorical_crossentropy(prediction, targets)
    train_loss = lasagne.objectives.aggregate(train_loss, mode='mean')

    # Update expressions for training (Stochastic Gradient Descent with Nesterov momentum)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=1e-5, momentum=0.9)

    # Expressions for validation loss + accuracy (topk)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, targets)
    test_loss = lasagne.objectives.aggregate(test_loss, mode='mean')

    test_acc = T.mean(T.any(T.eq(T.argsort(test_prediction, axis=1)[:, -top_k:], targets.dimshuffle(0, 'x')), axis=1), dtype=theano.config.floatX)

    # Function to perform mini-batch training
    train_fn = theano.function([input_var, target_var], train_loss, updates=updates, allow_input_downcast=True)

    # Function to compute the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)




    best_val = 100.0
    # Launch the training loop:
    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in dataset.iterate_minibatches():
            inputs, targets = batch
            train_err+= train_fn(inputs, targets)

            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        val_acc = 0
        for batch in dataset.iterate_minibatches(True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.4f} %".format(val_acc / val_batches * 100))
        
        if ((val_err / val_batches) < best_val) :
            best_val = val_err / val_batches
            # save network            
            np.savez('model_' + str(epoch + 1) + '.npz', lasagne.layers.get_all_param_values(network))
        np.savez('snapshot.npz', lasagne.layers.get_all_param_values(network))
        if epoch % 10 == 0:
            np.savez('snapshot_' + str(epoch + 1) + '.npz', lasagne.layers.get_all_param_values(network))
    
        print("lowest val %08f"%(best_val))

    
if __name__ == '__main__':
    main()
