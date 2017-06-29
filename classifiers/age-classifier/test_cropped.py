import os
import utils
import argparse
import cv2
import numpy as np
import cnn
import theano
import lasagne

MEAN =  np.array([103.939, 116.779, 123.68])   # BGR
SOURCE_SIZE = 256
CROP_SIZE = 224

class TesterOnCroppedFaces:
    def __init__(self, options):
        self.subsets = [options.subsets] if options.subsets != "both" else ["train", "test"]
        self.root_folder = options.data_path
        self.weigths_file = options.weights

        self.forward_prop = self.init_network()


    def init_network(self):
        input_var =theano.tensor.tensor4('input')
        network = cnn.build_model(input_var)['prob']
        base_data = np.load(self.weigths_file)
        base_data = base_data[base_data.keys()[0]]
        lasagne.layers.set_all_param_values(network, base_data)

        x = theano.tensor.tensor4('x')
        y = lasagne.layers.get_output(network, x, deterministic=True)
        return theano.function( [x], y , allow_input_downcast=True )


    def get_prediction_from_path(self, full_path):
        img = cv2.imread(full_path)
        img = cv2.resize(img, (SOURCE_SIZE, SOURCE_SIZE))
        stride_y, stride_x = int( (SOURCE_SIZE - CROP_SIZE)/2 ), int( (SOURCE_SIZE - CROP_SIZE)/2 )
        img = img[ stride_y:stride_y+CROP_SIZE , stride_x:stride_x+CROP_SIZE,:]

        img = np.float32(img)
        if MEAN is not None:
            for channel_idx, value in enumerate(MEAN):
                img[:,:,channel_idx]-=value

        img = img.transpose(2,0,1)
        img = img.reshape( (1,) +img.shape )

        net_result = self.forward_prop(img)[0]
        expectation = np.sum( np.array(range(101))*net_result )

        return expectation


    def evaluate_subset(self, data):
        errors = []
        print("Evaluation started. Going through {} images".format(len(data)))
        for entry in data:
            expected_age = self.get_prediction_from_path(entry['full_path'])
            errors.append( np.abs(expected_age-entry['age']) )
            if (len(errors) % 1000) == 0:
                print("Analised total {} of images. {}% done ".format(len(errors), '%.2f' % (len(errors)/float(len(data))*100) ) )

        errors = np.array(errors)
        return np.mean(errors), np.std(errors)


    def evaluate_dataset(self):
        for subset in self.subsets:
            print("\n---------------------")
            print("Subset: {}".format(subset))
            mean, std = self.evaluate_subset( utils.load_data( os.path.join(self.root_folder,subset) ) )
            print("Test complete for {}.".format(subset))
            print("Average error:{}".format(mean))
            print("Std error:{}".format(std))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", action="store", default="data",
                        help="root data path (default: %(default)s)")
    parser.add_argument("-w", "--weights", action="store", default="snapshot.npz",
                        help="path to weights file (default: %(default)s)")
    parser.add_argument("-s", "--subsets", type=str, choices=["test", "train", "both"], default="test",
                        help="subsets. select either test (uses test folder); train (uses train folder);"
                             " or both (uses both folders) (default: %(default)s)")
    args = parser.parse_args()
    tester = TesterOnCroppedFaces(args)
    tester.evaluate_dataset()
