import cv2
import numpy as np
import scipy.io
from scipy import ndimage
import tensorflow as tf
import h5py
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import random
import shutil
import os

def gaussian_filter_density(gt, verbose=False):
    #print(width, height)
    #density = np.zeros(gt.shape, dtype=np.float32)
   # print(density.shape)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
   
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    density=np.zeros([gt.shape[0], gt.shape[1]],dtype=np.float32)
    distances, locations = tree.query(pts, k=4)
    if verbose:
        print ('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros([gt.shape[0], gt.shape[1]], dtype=np.float32)
        pt=np.round(pt).astype('int')
        pt2d[pt[1],pt[0]]=1000
        sigma=min(max(1,int(np.sum(distances[i][1:3])*0.1)),8)
        density+=ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        #if gt_count > 1:
        #    sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        #else:
        #    sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        #density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant')
    if verbose:
        print ('done.')
    return density

def load_image(image_path):
    #load an image file
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)  # Adjust channels if needed
    return img

def load_ground_truth(mat_file_path, scale_x=1, scale_y=1):
    #load ground truth from .mat file
    ground_truth_data = scipy.io.loadmat(mat_file_path)
    ground_truth = ground_truth_data['loc']
    # Update the coordinates in the ground truth based on the new image size
    updated_ground_truth = ground_truth.copy()
    updated_ground_truth[:, 0] *= scale_x  # Update x-coordinates
    updated_ground_truth[:, 1] *= scale_y  # Update y-coordinates
    
    return updated_ground_truth

def resize(new_height, new_width, tensor):
    resized_tensor = tf.image.resize(tensor, [  new_height, new_width])
    return resized_tensor
    
def create_density_map(image, gt, verbose=False):
    #inputs: take the image, ground truth
    #outputs: create a density map
    if verbose:
        print('Computing density')
    k = np.zeros((image.shape[0],image.shape[1]))

    for i in range(len(gt)):
        if int(gt[i][1]) < image.shape[0] and int(gt[i][0]) < image.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    #print(k.shape)
    k = gaussian_filter_density(k)
    return k

def save_density_map(path, k):
    with h5py.File(path, 'w') as hf:
            hf['density'] = k
    
def load_density_map(density_file_name, verbose=False):
    if verbose:
        print(f'Loading existing density map from {density_file_name}')
    with h5py.File(density_file_name, 'r') as hf:
        k = hf['density'][:]
    return k

def preprocess_data(image):
    '''
    apply the preprocessing necessary for pretrained model
    '''
    image = (image - 127.5) / 127.5
    return image
    
def load_data(path, new_height = 640, new_width = 360, verbose=False, use_existing_density=False, number=None, preprocess =True):
    data = []
    counter = 0
    failure = 0
    error_log = []
    for file_path in Path(path).iterdir():
        counter +=1
    print(f'Data importation starting for {counter} images:')
    with tqdm(total=counter) as pbar:
        for file_path in Path(path).iterdir():
            try:
                if verbose:
                    print(f'Ongoing: {file_path}')
                if file_path.is_file() and file_path.suffix.lower() == '.jpg':
                    file = str(file_path)
                    ID = file.split('/')[-1]
                    ID = ID.split('.')[0]
                    if verbose:
                        print('loading image')
                    image = load_image(path+f'/{ID}'+'.jpg') / 255
                   
                    image_height,image_width = image.shape[:2]
                    image = resize( new_height,new_width,  image)
                    if preprocess:
                        image =   preprocess_data(image*255 )
                
                    y_link = path.split('/')[:-2]
                    y_link = '/'.join(y_link)+'/ground_truth' +'/'+ID+'.mat'
                    if verbose:
                        print('Loading .mat file')
                    x_scale = new_width/image_width
                    y_scale = new_height/image_height

                    gt = load_ground_truth(y_link, x_scale, y_scale )
                    #sensity_files
                    density_file_name = file_path.with_suffix('.h5')
                    density_file_name  = str(density_file_name).replace('frame','density')
                    if use_existing_density:
                        if Path(density_file_name).exists():
                            k =load_density_map(density_file_name, verbose=verbose)
                        else:
                            return
                    else:
                        k = create_density_map(image, gt, verbose=verbose)
                        save_density_map(density_file_name,k)
                    count = len(gt)
                    data.append({'ID': ID ,'image' : image, 'truth': gt , 'count' : count, 'density' : k})
                    if verbose:
                        print('Done')
                    pbar.update(1)
            except Exception as e:
                failure += 1
                traceback_str = traceback.format_exc()
                error_log.append({'file_path': file_path, 'exception': e, 'traceback': traceback_str})
                continue
    print(f'Failed density: {failure}/{counter}')
    return data, error_log
   

def plot_density_result(data_dict):
    #input:the density entry (data_train[i]['density']) for example
    fig = plt.figure(figsize=[40, 16])
    for i in range(0,40,2):
        # Plot density map on the left side of the image
        ax = fig.add_subplot(4, 10, i + 1)  # Adjust subplot arrangement
        ax.imshow(data_dict[i]['image'], cmap=plt.get_cmap('gray'))
        ax.imshow(data_dict[i]['density'],  cmap='prism', interpolation='bicubic',alpha=0.25)  # Use 'hot' colormap for density map
        ax.set_axis_off()

        # Plot image and ground truth on the right side of the density map
        ax = fig.add_subplot(4, 10, i + 2)  # Adjust subplot arrangement
        ax.imshow(data_dict[i]['image'], cmap=plt.get_cmap('gray'))
        num_people = np.sum(data_dict[i]['density']/1000)
        ax.text(0.5, 1.05, f'Number of people: {num_people:.2f}', 
            transform=ax.transAxes, ha='center', va='bottom', fontsize=10, color='red')
        loc = data_dict[i]['truth']
        for l in loc:
            plt.scatter(l[0], l[1], color='red', marker='o')
        ax.set_axis_off()
    plt.show()

#def plot_prediction_results(data_dict, y_pred):
import cv2

def load_density(file_path):
    gt_file = h5py.File(file_path, 'r')
    groundtruth = np.asarray(gt_file['density'])
    return groundtruth

def downsample_density_map(density):
    #here the value 4 was selected to have an output 56*56, matching the output off the network
    density2 = density
    density3 = cv2.resize(density2,(int(density2.shape[1]/2), int(density2.shape[0]/2)),interpolation = cv2.INTER_CUBIC)*64
    density4 = np.expand_dims(density3, axis=0)
    density5 = np.expand_dims(density4, axis=3)
    return density5


def get_batches(generator):
    X = []
    Y = []

    # Iterate over the generator to get batches of data
    for batch_X, batch_Y in generator:
        X.append(batch_X)
        Y.append(batch_Y)

    # Concatenate the batches to form the complete dataset
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    return X,Y

def create_mall_mat(ground_path):
    '''
    from the single file, creates multiple file. do not create a folder.
    '''
    location_file = 'data/mall_dataset/'
    if  os.path.exists(location_file +'ground_truth'):
        shutil.rmtree(location_file +'ground_truth')
    os.makedirs(location_file+'ground_truth')
    mat = scipy.io.loadmat(ground_path)['frame'][0]
    with tqdm(total=mat.shape[0]) as pbar:
        for i in range(mat.shape[0]):
            new_file = mat[i][0][0][0]
            ID = f'{i+1:06}'
            mat_filename =f'seq_{ID}.mat'
            mdic = {"loc": new_file, "label": ID}
            scipy.io.savemat(location_file+'ground_truth'+'/' + mat_filename, mdic )
            pbar.update(1)
        
def create_train_test_mall(ratio):
    '''
    from the original folder of mall dataset, create a train and test folder
    '''
    sets = ['train', 'test']
    train_number = int(2000*ratio)
    newpath = r'data/mall_dataset/'
    samples = [z for z in range(1,2001)]
    train = random.sample(samples, train_number)
    counter = 0
    for cat in sets:
        shutil.rmtree(f'{newpath}{cat}')
        if not os.path.exists(newpath+cat):
            os.makedirs(newpath+cat)
            os.makedirs(newpath+cat+'/'+'frame')
            os.makedirs(newpath+cat+'/'+'ground_truth')
    with tqdm(total=2000 ) as pbar:       
        for i  in range(1,2001):
            source = f'{newpath}frame/seq_{i:06}.jpg'
            source_mat = f'{newpath}ground_truth/seq_{i:06}.mat'
            if i in train:
                destination = f'{newpath}train/frame/seq_{i:06}.jpg'
                destination_mat = f'{newpath}train/ground_truth/seq_{i:06}.mat'
            else:
                destination = f'{newpath}test/frame/seq_{i:06}.jpg'
                destination_mat = f'{newpath}test/ground_truth/seq_{i:06}.mat'
            # Copy the file from the source to the destination
            shutil.copy(source, destination)
            shutil.copy(source_mat, destination_mat)
            pbar.update(1)