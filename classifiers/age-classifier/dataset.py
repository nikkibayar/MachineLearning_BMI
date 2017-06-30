import os
import cv2
from multiprocessing import Process, JoinableQueue
import numpy as np
import random
import utils


class BatchCreator:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        if self.train is True:
            if self.equalise is True:
                self.samples_per_age_group = self.get_age_histogram()
                if self.showequalization is True:
                    self.print_age_groups()
        else:
            self.current_sample = 0



    def get_sample(self, id=None):
        if id is not None:
            sample_idx = id
        else:
            if self.train is True:
                if self.equalise is True:
                    age_group = random.choice( self.samples_per_age_group.keys() )
                    sample_idx = random.choice( self.samples_per_age_group[age_group])
                else:
                    sample_idx = random.choice( range(len(self.data)) )
            else:
                sample_idx= self.current_sample
                self.current_sample+=1
                if self.current_sample == len(self.data):
                    self.current_sample =0

        age = self.data[sample_idx]['age']
        image_path = self.data[sample_idx]['full_path']

        #Load image
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.source_size, self.source_size))

        if self.train is True:
            #perform data-augmentation
            stride_y, stride_x = int(random.random() * (self.source_size - self.crop_size)), int(random.random() * (self.source_size - self.crop_size))
            if random.random() > 0.5:
                image = cv2.flip(image,1)

        else:
            #Always use a fixed crop while testing
            stride_y, stride_x = int( (self.source_size - self.crop_size)/2 ), int( (self.source_size - self.crop_size)/2 )


        image = image[ stride_y:stride_y+self.crop_size , stride_x:stride_x+self.crop_size,:]

        image = np.float32(image)
        if self.mean is not None:
            for channel_idx, value in enumerate(self.mean):
                image[:,:,channel_idx]-=value

        return image, age

    def calculate_mean(self):
        all_images = []
        for i in range(len(self.data)):
            image , _ = self.get_sample(i)
            all_images.append(image)
        all_images = np.array(all_images)
        print(np.mean(all_images[:,:,:,0]))
        print(np.mean(all_images[:,:,:,1]))
        print(np.mean(all_images[:,:,:,2]))


    def get_batch(self):
        inputs = np.empty((self.batch_size, 3, self.crop_size, self.crop_size), dtype=np.float32)
        targets = np.empty((self.batch_size, 1), dtype=np.uint8)
        for i in range(self.batch_size):
            image, target = self.get_sample()
            inputs[i] = image.transpose(2,0,1)
            targets[i] = target

        return inputs, targets



    def print_age_groups(self):
        ages = sorted(self.samples_per_age_group)
        for i in range(len(ages)):
            age = ages[i]
            upperbound = int(ages[i +1]) if i != len(ages) -1 else "Inf"
            print("{}-{}: {} samples".format(int(age),
                                             upperbound,
                                             len(self.samples_per_age_group[age])
                                             ))

    def get_age_histogram(self):
        all_ages = [x['age'] for x in self.data ]

        # Calculating the histogram is not really needed, but it's useful for debugging purposes
        histogram,bins = np.histogram(all_ages, bins=15, range=(0, 95), density=True)

        indices = np.digitize(all_ages, bins)

        data_per_bin = {}
        for i in range(len(indices)):
            bin = bins[indices[i] -1]
            if bin not in data_per_bin:
                data_per_bin[bin] = []
            data_per_bin[bin].append(i)
        return data_per_bin

class Dataset:
    def __init__(self, root_path,  source_size, crop_size, batch_size, mean=None, calculate_mean=None):
        self.root_path, self.source_size, self.crop_size, self.batch_size, self.mean = root_path, source_size, crop_size, batch_size, mean

        if calculate_mean is None:
            self.train_data = utils.load_data(os.path.join(root_path,"train"))
            self.test_data = utils.load_data(os.path.join(root_path,"test"))

            self.train_batch_queue = JoinableQueue(10)
            Process(target=self.batch_creator, args=(self.train_batch_queue, True,  True, True)).start()

            self.test_batch_queue = JoinableQueue(10)
            Process(target=self.batch_creator, args=(self.test_batch_queue, False, False, False)).start()
        else:
            sample_for_mean_calculation = utils.load_data(os.path.join(root_path,calculate_mean))
            sampler = BatchCreator(
                                   data=sample_for_mean_calculation,
                                   crop_size=self.crop_size,
                                   source_size=self.source_size,
                                   mean=None,
                                   train=True,
                                   equalise=False)
            sampler.calculate_mean()





    def batch_creator(self, queue, train,  equaliseDistribution, showEqualisation):
        creator = BatchCreator(batch_size=self.batch_size,
                               crop_size=self.crop_size,
                               source_size=self.source_size,
                               data=self.train_data if train is True else self.test_data,
                               mean=self.mean,
                               train=train,
                               equalise=equaliseDistribution,
                               showequalization=showEqualisation)
        while True:
            inputs, targets = creator.get_batch()
            queue.put((inputs, targets))

    def iterate_minibatches(self, val=False):
        if val:
            #Use all the validation data while testing
            batches = int(len(self.test_data)/(self.batch_size))
            queue = self.test_batch_queue
        else:
            batches = 5000
            queue = self.train_batch_queue
        batches = 1

        for _ in range(batches):
            inputs, targets = queue.get()
            queue.task_done()
            yield inputs, targets



if __name__ == "__main__":
    # Small code just to test the dataset loader
    mean = np.array([103.939, 116.779, 123.68])   # BGR
    batch_size = 3

    # Normal dataset instantiation
    dataset = Dataset("data", 75, 64, batch_size, mean)
    while True:
        for batch in dataset.iterate_minibatches(True):
            inputs, targets = batch
            for i in range(batch_size):
                img=inputs[i].transpose(1, 2, 0)
                if mean is not None:
                    for channel_idx, value in enumerate(mean):
                        img[:, :, channel_idx]+=value

                img = np.uint8(img)
                print("showing img sample: {}".format(i))
                cv2.imshow("img", img)
                cv2.waitKey()
                
    #Calling dataset like this will calculate the mean from the distribution, a mean for each channel (BGR)
    dataset = Dataset("data", 75, 64, batch_size, mean, calculate_mean="train")