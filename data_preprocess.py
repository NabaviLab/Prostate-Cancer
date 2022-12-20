import os
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '20000000000000')

# os.add_dll_directory('C:\\openslide-win64\\openslide-win64\\bin')
# os.environ['PATH'] = "C:\\vips-dev-w64\\vips-dev-8.13\\bin" + ";" + os.environ['PATH']
# import openslide
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
import pickle
# from tiatoolbox.tools import patchextraction
from tiatoolbox.utils.misc import imread
# from tiatoolbox.utils.misc import read_locations
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
import cv2 as cv
# import large_image
# import gzip
import tensorflow as tf
from tensorflow.keras.models import load_model
from skimage import exposure
from sklearn.neighbors import NearestNeighbors



def crop_slides(args):
    folders = os.listdir(args.wsi_dir)
    print('Start cropping the slides......\n')
    for idx in range(len(folders)):
        print('Image number: ',idx)
        print('Total Image number: ',len(folders))
        image_dir = args.wsi_dir + folders[idx]
    
    
        ALL_PATCHES = []
        PATCHES = []
       
        input_img = io.imread(image_dir)
        print('Image res : ',input_img.shape)
        size = args.patch_size
        for x in range(0,input_img.shape[0],size):
            for y in range(0,input_img.shape[1],size):
                patch = input_img[x:min(x+size, input_img.shape[0]), y:min(y+size, input_img.shape[1] )]
                loc = (x,y)
                if patch.shape ==(size,size,3):
                    ALL_PATCHES.append([patch, loc])
                    gray = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
                    gray[gray>200] = 1
                    if np.count_nonzero(gray-1)/(size*size) > 0.75:
                        PATCHES.append([patch, loc])
    
    PATCHES_dir = args.patch_dir+'//' + folders[idx][:-4]
    os.makedirs(PATCHES_dir)
    with open(PATCHES_dir+ '//patches.pkl', 'wb') as f:
        pickle.dump(PATCHES, f)
        
    print('Cropping is Done!!!!\n')
    
    
    
    
def create_labels(args):
    folders = os.listdir(args.wsi_dir)
    clinical = pd.read_csv(args.labels) 
    label_list = []
    print('Start creating labels......\n')
    for idx in range(len(folders)):
        #print('Image number: ',idx)
        score1 = int(clinical['primary_gleason_grade'][idx])
        score2 = int(clinical['secondary_gleason_grade'][idx])
        
        summ = score1 + score2
        
        if summ==0:
            label = 0
        elif summ<=6:
            label = 1                
        elif summ == 7 and score1 ==3:
            label = 2
        elif summ == 7 and score1 ==4:
            label = 3
        elif summ == 8:
            label = 4
        elif summ >= 9:
            label = 5
        
        
        label_list.append([score1, score2, label])
        label_list = np.array(label_list)
        
        label_dir = args.label_dir+'//' + clinical['image_id'][idx]
        os.makedirs(label_dir)
        with open(label_dir+ '//label.pkl', 'wb') as f:
            pickle.dump(label_list, f) 
        label_list = []
    print('Labels are created!!!!\n')
    

def feature_extraction(args):
    print('Loading the Encoder....\n')
    encoder = load_model(args.auto_model)
    print('Encoder is loaded!\n')
    patch_names = os.listdir(args.patch_dir)
    print('Start extracting features...\n')
    for i in range(len(patch_names)):
        print('Image number: ',i)
        model_output = []
        Feature_dir = args.feature_dir +'//' + patch_names[i]
        
        patch_dir  = args.patch_dir + '//' + patch_names[i]
        patches = pickle.load(open(patch_dir  +  '//patches.pkl'  , "rb"))
        
        
        for jj in range(len(patches)):
            patch = patches[jj][0]
        
            z = encoder.predict(patch.reshape((1,patch.shape[1],patch.shape[1],3))).reshape((2560,))
            model_output.append(z)
            
        model_output = np.array(model_output)
        
        os.makedirs(Feature_dir)
        with open(Feature_dir+ '//vectors.pkl', 'wb') as f:
            pickle.dump(model_output, f)
        model_output = []
    print('Feature extraction is finished.\n')
    
    
    
class Graph_Conversion():
    def __init__(self, folders,patch_dir,feature_vector_dir, score_dir,save_graph_dir):
        self.patch_dir = patch_dir
        self.graph_dir = save_graph_dir
        self.score_dir = score_dir
        self.feature_vector_dir = feature_vector_dir
        self.folders = folders
    def load_files(self, ind):
        scores = pickle.load(open(self.score_dir + self.folders[ind] +  '//score.pkl'  , "rb"))
        feature_vector = pickle.load(open(self.feature_vector_dir + self.folders[ind] +  '//vectors.pkl'  , "rb"))

        patches = pickle.load(open(self.patch_dir + self.folders[ind] +  '//patches.pkl'  , "rb"))
        return patches, feature_vector, scores
    
    def getBestScore(self, scores, top_ratio):
        flat = scores.flatten()
        indices = np.argpartition(flat, -int(len(scores)*top_ratio))[-int(len(scores)*top_ratio):]
        indices = indices[np.argsort(-flat[indices])]
       
        return np.unravel_index(indices, scores.shape)[0]
    
    def fitKearest(self,node_locations, n_neighbors):
        if len(node_locations)<n_neighbors+1:
            knn = NearestNeighbors(n_neighbors=len(node_locations))
        else:
            knn = NearestNeighbors(n_neighbors=n_neighbors+1)
        knn.fit(node_locations)
        return knn
    def Build_Graph(self, ind, top_ratio, n_neighbors):
    
        Graph = dict()
        patches, feature_vector, scores = self.load_files(ind)
        feature_vector = np.array(feature_vector)
        best_scores = self.getBestScore(scores, top_ratio)
        node_locations = np.zeros((len(best_scores),2))
        good_feature_vectores = np.zeros((len(best_scores), feature_vector.shape[-1]))
        jj = 0
        for idx in best_scores:
            good_patch_loc = patches[idx][1]
            node_locations[jj][0] = good_patch_loc[0]
            node_locations[jj][1] = good_patch_loc[1]
            good_feature_vectores[jj] = feature_vector[idx]
            jj += 1
        
        knn = self.fitKearest(node_locations, n_neighbors)
        edge_list = []
        for ii in range(len(node_locations)):
            edge_list += list(knn.kneighbors(node_locations[ii].reshape(1,2), return_distance=False)[:,1:][0])
            
        Edges = [list(np.repeat(np.arange(len(node_locations)), knn.n_neighbors-1))] + [edge_list]

        Adjacency_matrix = np.zeros((len(node_locations),len(node_locations)))
        Adjacency_matrix[Edges[0], Edges[1]] = 1
        Adjacency_matrix = np.maximum( Adjacency_matrix, Adjacency_matrix.transpose() )
        
        
        Graph['Adjacency_matrix'] = Adjacency_matrix
        Graph['edge_list'] = [list(Adjacency_matrix.nonzero()[0])] + [list(Adjacency_matrix.nonzero()[1])]
        Graph['node_locations'] = node_locations
        Graph['node_features'] = good_feature_vectores
        return Graph
    
    def Save_Graph(self, Graph, ind):
        os.makedirs(self.graph_dir + self.folders[ind])
        with open(self.graph_dir + self.folders[ind] + '//graph.pkl', 'wb') as f:
            pickle.dump(Graph, f) 


    def Build_All_Graphes(self, top_ratio, n_neighbors):
        for ind in range(len(self.folders)):
            Graph = self.Build_Graph(ind, top_ratio, n_neighbors)
            self.Save_Graph(Graph, ind)
            print(ind)
            


def build_graphs(args):        
    patch_dir = args.patch_dir + '//'
    ###image_dir = 'data//'
    graph_dir = args.graph_dir+'//'
    score_dir = args.scores_dir
    feature_vector_dir = args.feature_dir +'//'
    folders = os.listdir(score_dir)
    print('Start creating the graphs....\n')
    g = Graph_Conversion(folders,patch_dir,feature_vector_dir, score_dir,graph_dir)
    g.Build_All_Graphes( args.top/100, args.n_neighbor)
    print('The graphs are saved!\n')
    
    
def histo_equalization(img_array):
    ######################################
    # PERFORM HISTOGRAM EQUALIZATION
    ######################################
    
    """
    STEP 1: Normalized cumulative histogram
    """
    #flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=256)
    
    #normalize
    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array/num_pixels
    
    #normalized cumulative histogram
    chistogram_array = np.cumsum(histogram_array)
    
    
    """
    STEP 2: Pixel mapping lookup table
    """
    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)
    
    
    """
    STEP 3: Transformation
    """
    # flatten image array into 1D list
    img_list = list(img_array.flatten())
    
    # transform pixel values to equalize
    eq_img_list = [transform_map[p] for p in img_list]
    
    # reshape and write back into img_array
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)
    return eq_img_array



def histo_equalization2(patch):
    img_yuv = cv.cvtColor(patch, cv.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    return img_output