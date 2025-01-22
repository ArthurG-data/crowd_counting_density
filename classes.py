import numpy as np
import keras
from tensorflow.keras.layers import Layer
from pathlib import Path
import tensorflow as tf
import h5py
import scipy.io
from scipy import ndimage
from tqdm import tqdm
import datetime
from scipy.ndimage import gaussian_filter 
import os
import shutil

class CustomMultiply(Layer):
    def __init__(self, **kwargs):
        super(CustomMultiply, self).__init__(**kwargs)

    def call(self, x_conct, x_fel):
        result = x_conct * (1 + x_fel)
        return result

class PreprocessLayer(Layer):
    def __init__(self, **kwargs):
        super(PreprocessLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return keras.applications.mobilenet.preprocess_input(inputs * 255)

class Density_map_creator():
    '''create the density maps
    The input is the dimension required, it will adjust the value so that it fits the image
    '''
    def __init__(self,dataset, dim=(640, 360), setType='train' ):
        'Initialization'
        #the first element of dim is the height, second the width
        self.dim = dim
        self.dataset = dataset
        #self.path = f'Beijing-BRT-dataset-master/{setType}/'
        self.path = f'data/{dataset}/{setType}/'
        self.setType =setType
        self.list_IDs =  self.find_ids()
        
        
    @property
    def setType(self):
        return self._setType
    @property
    def dataset(self):
        return self._dataset

    @setType.setter
    def  setType(self, value):
        allowed_values = ['train', 'test']
        if value not in allowed_values:
            raise ValueError(f"Value must be one of {allowed_values}")
        self._setType = value
        
    @dataset.setter
    def  dataset(self, value):
        allowed_values = ['mall_dataset', 'Beijing-BRT-dataset-master']
        if value not in allowed_values:
            raise ValueError(f"Value must be one of {allowed_values}")
        self._dataset = value
        
    def find_ids(self):
        IDs = []
        for file_path in Path(self.path+'/ground_truth/').iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.mat':
                file = str(file_path)
                ID = file.split('/')[-1]
                ID = ID.split('.')[0]
                IDs.append(ID)
        return IDs
    
    def load_ground_truth(self, mat_file_path):
        #load ground truth from .mat file
        ground_truth_data = scipy.io.loadmat(mat_file_path)
        ground_truth = ground_truth_data['loc']
        if self.dataset == 'Beijing-BRT-dataset-master':
            original_width = 360
            original_height = 640
        elif self.dataset == 'mall_dataset':
            original_width = 640
            original_height = 480
        scale_x = self.dim[1]/original_width
        scale_y = self.dim[0]/original_height
        # Update the coordinates in the ground truth based on the new image size
        updated_ground_truth = ground_truth.copy()
        try:
            updated_ground_truth[:, 0] *= scale_x  # Update x-coordinates, first column
            updated_ground_truth[:, 1] *= scale_y  # Update y-coordinates, second column
            updated_ground_truth =updated_ground_truth.astype(np.float64)
        except TypeError as e:
            print(f"TypeError: {e}. File: {mat_file_path}")
            
        return updated_ground_truth

    def density_map_creator(self):
                    # Generate data and save
        counter = len(self.list_IDs)
        if os.path.exists(self.path +'density'):
            shutil.rmtree(self.path +'density')
        os.makedirs(self.path+'density')
        with tqdm(total=counter ) as pbar:
            for i, ID in enumerate(self.list_IDs):
                  # Store sample
                try:
                    file_path = self.path + '/ground_truth/'+ ID + '.mat'
                    gt = self.load_ground_truth(file_path)
                    file_path = Path(file_path)
                    density_file_name = file_path.with_suffix('.h5')
                    density_file_name  = str(density_file_name).replace('ground_truth','density')
                    k = self.map_generation( gt)
                    self.save_density_map(density_file_name,k)
                except OverflowError:
                    print(f"Overflow error encountered at file {ID}. Skipping this point.")
                pbar.update(1)


    def map_generation(self, gt):
        #inputs: ground truth
        #outputs: create a density map
        k = np.zeros((self.dim[1], self.dim[0]))  # Corrected dimensions for k
        for i in range(len(gt)):
            # Check if the point is within the image dimensions
            if int(gt[i][1]) < self.dim[1] and int(gt[i][0]) < self.dim[0]:  # Corrected indexing for dimensions
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = self.gaussian_filter_density(k)
        return k
    
    def save_density_map(self, path, k):
        with h5py.File(path, 'w') as hf:
                hf['density'] = k
                
    def gaussian_filter_density(self, gt):
        density=np.zeros([gt.shape[1], gt.shape[0]],dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return density

        pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        
        distances, locations = tree.query(pts, k=4)
  
        for i, pt in enumerate(pts):
            pt2d = np.zeros([gt.shape[1], gt.shape[0]], dtype=np.float32)

            pt=np.round(pt).astype('int')
            pt2d[pt[1],pt[0]]=1000
            if gt_count > 1:
                sigma = np.clip(np.sum(distances[i][1:4]) * 0.1, 1, 8)
            else:
                sigma = np.clip(np.average(np.array(gt.shape)) / 4., 1, 8)
            
            density+= gaussian_filter(pt2d, sigma, mode='constant')
           
        return density

        
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset, setType ,preprocess=True, batch_size=1, dim=(640,360), n_channels=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.datset = dataset
        self.batch_size = batch_size
        self.path = f'data/{dataset}/{setType}/'
        self.setType =setType
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.list_IDs =  self.find_ids()
        self.preprocess = preprocess
        self.on_epoch_end()
        
    '''
    setter and getter to limite the values of setType
    '''
    @property
    def setType(self):
        return self._setType
    @property
    def dataset(self):
        return self._dataset

    @setType.setter
    def  setType(self, value):
        allowed_values = ['train', 'test']
        if value not in allowed_values:
            raise ValueError(f"Value must be one of {allowed_values}")
        self._setType = value
        
    @dataset.setter
    def  dataset(self, value):
        allowed_values = ['mall_dataset', 'Beijing-BRT-dataset-master']
        if value not in allowed_values:
            raise ValueError(f"Value must be one of {allowed_values}")
        self._dataset = value
        
    def find_ids(self):
        IDs = []
        for file_path in Path(self.path+'/frame/').iterdir():
            if file_path.is_file() and file_path.suffix.lower() == '.jpg':
                file = str(file_path)
                ID = file.split('/')[-1]
                ID = ID.split('.')[0]
                IDs.append(ID)
        return IDs 
    
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        'Generate one batch of data'
          # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

          # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

          # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
    
    def load_image(self, image_path):
#load an image file
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (self.dim[0], self.dim[1]))# Adjust channels if needed
        img = tf.cast(img, tf.float32)
        return img
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = tf.TensorArray(tf.float32, size=self.batch_size)
        y = tf.TensorArray(tf.float32, size=self.batch_size)
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = self.load_image(self.path + 'frame/' + ID + '.jpg')
            if self.preprocess:
                img = self.preprocess_data(img)
            X = X.write(i, img)
            # Store maps
            density_map = self.load_density_map(self.path + 'density/' + ID + '.h5')
            y = y.write(i, density_map)
        
        return X.stack(), y.stack()
    
    def load_density_map(self, density_file_name):
        with h5py.File(density_file_name, 'r') as hf:
            k = hf['density'][:]
        k = tf.convert_to_tensor(k, dtype=tf.float32)
       
        return k

    def preprocess_data(self, image):
        image = (image - 127.5) / 127.5
        return image
 