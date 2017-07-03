#!/usr/bin/env python

import time
import datetime
import numpy as np
import theano
import theano.tensor as T
import lasagne
import cnn
import random
from dataset import Dataset
import os

num_classes = 101
top_k = 5
MODE = "PRETRAINED_VGG16" # either "PRETRAINED_VGG16" or "WIDERESNETS"

DEFS={ 'PRETRAINED_VGG16' : {
                                'source_size': 256,
                                'crop_size': 224,
                                'mean': np.array([103.939, 116.779, 123.68]) ,
                                'batch_size': 16,
                                'snapshot': 'snapshot.npz'
                            },
        'WIDERESNETS' :     {
                                'source_size': 75,
                                'crop_size': 64,
                                'mean': np.array([83.8461, 95.9349, 117.756]),
                                'batch_size': 16,
                                'snapshot': None
                            }
               }
def build_network(input_var=None):
    if MODE == "PRETRAINED_VGG16":
        network = cnn.build_vgg16(input_var)['prob']
        print("(sanity check) Trainable layers before: {}".format(len(lasagne.layers.get_all_params(network, trainable="True"))) )
        network = freeze_early_weights(network)
        print("(sanity check) Trainable layers after: {}".format(len(lasagne.layers.get_all_params(network, trainable="True"))) )
    else:
        network = cnn.build_WideResNet(input_var, 16, 8)

    network = lasagne.layers.DimshuffleLayer(network, (1, 0))
    network = lasagne.layers.FlattenLayer(network)
    network = lasagne.layers.DimshuffleLayer(network, (1, 0))
    return network

def build_targets_formatter(input_var=None):
    network = lasagne.layers.InputLayer(shape=(DEFS[MODE]['batch_size'], num_classes), input_var=input_var)
    network = lasagne.layers.FlattenLayer(network, outdim=1)

    return network

def freeze_early_weights(network):
    layers = lasagne.layers.get_all_layers(network)
    for i in range(len(layers)-4):
        layer = layers[i]
        params = layer.get_params()
        if len(params) > 0:
            layer.params[layer.W].remove("trainable")
            layer.params[layer.b].remove("trainable")
    return network

def save_weights(network, file_name, updates=None):
    np.savez(file_name, lasagne.layers.get_all_param_values(network))
    if updates is not None:
        f,_ = os.path.splitext(file_name)
        np.savez(f + "_updates.npz", [p.get_value() for p in updates.keys()] )

def load_weights(network, file_name, updates):
    if os.path.exists(file_name):
        print("Loading previously saved weights from {}".format(file_name))
        base_data = np.load(file_name)
        base_data = base_data[base_data.keys()[0]]
        lasagne.layers.set_all_param_values(network, base_data)
    else:
        print("Couldn't find {} on disk. Learning from scratch".format(file_name))
        return
    # Read and load updates file if it exists
    f,_ = os.path.splitext(file_name)
    updates_file = f + "_updates.npz"
    if os.path.exists(updates_file):
        print("Using optimization data from file {}".format(updates_file))
        base_data_optimization = np.load(updates_file)
        base_data_optimization = base_data_optimization[base_data_optimization.keys()[0]]
        for p, value in zip(updates.keys(), base_data_optimization):
            p.set_value(value)
    else:
        print("Not using optimization. No {} file found".format(updates_file))

def save_epoch_losses(losses_per_batch_training, losses_per_batch_testing, training_loss, validation_loss, validation_accuracy, log_file):
    with open(log_file, "a") as myfile:
        for loss in losses_per_batch_training:
            myfile.write("training_batch_loss " + str(loss) + '\n')
        for loss in losses_per_batch_testing:
            myfile.write("testing_batch_loss " + str(loss) + '\n')

        myfile.write("training_loss " + str(training_loss) + '\n')
        myfile.write("validation_loss " + str(validation_loss) + '\n')
        myfile.write("validation_accuracy " + str(validation_accuracy) + '\n')
        myfile.write("-" + '\n')


def get_log_file():
    now = datetime.datetime.now()
    file_name = os.path.join( "logs" ,
                              "{}-{}-{}-{}-{}-{}.txt".format( now.year, now.month, now.day, now.hour, now.minute, now.second))

    if os.path.exists("logs") is False:
        os.makedirs("logs")
    return file_name


def main(num_epochs=500):
    # Loading dataset
    print "Loading dataset"
    log_file = get_log_file()
    print("Saving logs in {}".format( log_file ))
    random.seed(123)

    dataset = Dataset("data",
                      DEFS[MODE]['source_size'],
                      DEFS[MODE]['crop_size'],
                      DEFS[MODE]['batch_size'],
                      DEFS[MODE]['mean'])
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.matrix('targets')

    # Build CNN model
    print("Building model and compiling functions...")
    network = build_network(input_var)

    target_formatter = build_targets_formatter(target_var)

    # Expressions for training losses
    prediction = lasagne.layers.get_output(network)
    targets = T.cast( lasagne.layers.get_output(target_formatter), 'int32')
    train_loss = lasagne.objectives.categorical_crossentropy(prediction, targets)
    train_loss = lasagne.objectives.aggregate(train_loss, mode='mean')

    # Update expressions for training (Stochastic Gradient Descent with Nesterov momentum)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=1e-2, momentum=0.9)

    # Expressions for validation loss + accuracy (topk)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, targets)
    test_loss = lasagne.objectives.aggregate(test_loss, mode='mean')

    test_acc = T.mean(T.any(T.eq(T.argsort(test_prediction, axis=1)[:, -top_k:], targets.dimshuffle(0, 'x')), axis=1), dtype=theano.config.floatX)

    # Function to perform mini-batch training
    train_fn = theano.function([input_var, target_var], train_loss, updates=updates, allow_input_downcast=True)

    # Function to compute the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], allow_input_downcast=True)

    # Load the weights obtained earlier
    if DEFS[MODE]['snapshot'] is not None:
        load_weights(network, DEFS[MODE]['snapshot'], updates)

    best_val = 100.0
    # Launch the training loop:
    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        losses_per_batch_training = []
        for batch in dataset.iterate_minibatches():
            inputs, targets = batch
            train_err+= train_fn(inputs, targets)
            train_batches += 1
            losses_per_batch_training.append(train_err)


        # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        val_acc = 0
        losses_per_batch_testing = []
        for batch in dataset.iterate_minibatches(True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            losses_per_batch_testing.append(err)
            val_err += err
            val_acc += acc
            val_batches += 1

        training_loss = train_err / train_batches
        validation_loss = val_err / val_batches
        validation_accuracy = val_acc / val_batches * 100

        save_epoch_losses(losses_per_batch_training,
                          losses_per_batch_testing,
                          training_loss,
                          validation_loss,
                          validation_accuracy,
                          log_file)
        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(training_loss))
        print("  validation loss:\t\t{:.6f}".format(validation_loss))
        print("  validation accuracy:\t\t{:.4f} %".format(validation_accuracy))
        
        if ((val_err / val_batches) < best_val) :
            best_val = val_err / val_batches
            # save network
            save_weights(network, 'model' + str(epoch + 1) + '.npz')
        save_weights(network, 'snapshot.npz', updates)
        if epoch % 10 == 0:
            save_weights(network, 'snapshot_' + str(epoch + 1) + '.npz')
    
        print("lowest val %08f"%(best_val))

    
if __name__ == '__main__':
    main()
