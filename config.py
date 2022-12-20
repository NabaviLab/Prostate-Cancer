import argparse
import json
import shutil
import os
#from utils import ensure_dirs


def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)

class Config(object):
    def __init__(self):

        # parse command line arguments
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in args.__dict__.items():
            print("{0:20}".format(k), v)
            self.__setattr__(k, v)

        # experiment paths
        self.patch_dir = 'Preprocessed_data//patches'
        self.label_dir = 'Preprocessed_data//labels'
        self.feature_dir = 'Preprocessed_data//features'
        self.graph_dir = 'Preprocessed_data//graphs'
        
        
        
        self.auto_model = args.auto_dir + 'encoder.h5'
                
        self.model1_dir = 'Models//model1'   
        self.result1_dir = 'Results//model1'
        self.model2_dir = 'Models//model2'   
        self.result2_dir = 'Results//model2'
        ensure_dirs([self.patch_dir, self.label_dir, self.feature_dir, self.graph_dir,self.model1_dir,self.result1_dir,self.model2_dir,self.result2_dir])

    def parse(self):
        """initiaize argument parser. Define default hyperparameters and collect from command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', type=str, default="crop", help="options: [crop, auto, feature, model1, graph, model2]")
        parser.add_argument('--wsi_dir', type=str, default="images/",
                            help="path to the WSIs")
        
        ## for crop
        parser.add_argument('--labels', type=str, default="clinical.csv",
                            help="directory of the clinical file containing the scores")
        
        parser.add_argument('--patch_size', type=int, default=512, help="patch size")
        
        ## for auto encoder
        parser.add_argument('--n_bag', type=int, default=500, help="number of bags to train the autoencoder")
        parser.add_argument('--lr_a', type=float, default=1e-3, help="learning rate for autoencoder")
        parser.add_argument('--batch_a', type=float, default=4, help="batch size for autoencoder")
        parser.add_argument('--epoch_a', type=int, default=50, help="total number of epochs to train autoencoder")
        parser.add_argument('--auto_dir', type=str, default="autoencoder/",
                            help="path to save autoencoder")
        
        
        
        ## for graph
        parser.add_argument('--scores_dir', type=str, default="Preprocessed_data//scores/",
                            help="path to the scores extracted by the first stage of model")
        
        parser.add_argument('--n_neighbor', type=int, default=10, help="number of neighbors in KNN")
        parser.add_argument('--top', type=int, default=40, help="top S% of the scores")
        
        
        ## for model 1 & 2
        parser.add_argument('--state1', type=str, default="traintest", help="options: [traintest, test]")
        parser.add_argument('--epoch', type=int, default=1000, help="number of epochs")
        parser.add_argument('--lr', type=int, default=5E-5, help="learning rate")
        parser.add_argument('--wd', type=int, default=10E-6, help="weight decay")
        parser.add_argument('--gate', type=bool, default=True, help="Gated attention or not")
        parser.add_argument('--hid_dim', type=int, default=512, help="hidden dimension in linear layer")
        parser.add_argument('--out_dim', type=int, default=256, help="output dim in linear layer")
        parser.add_argument('--n_class', type=int, default=5, help="number of output classes")
        parser.add_argument('--n_heads', type=int, default=6, help="number of attention layer")
        parser.add_argument('--fold', type=int, default=10, help="number of cross fold")
        parser.add_argument('--dense_dim', type=int, default=16, help="dim in last linear layer")
        parser.add_argument('--gat_heads', type=int, default=4, help="number of GAT attention layer")
        parser.add_argument('--GAT', type=bool, default=False, help="whether using GAT or GCN")

        #parser.add_argument('--TOloss', type=bool, default=False, help="test of TO loss (res must be 200400)")
        args = parser.parse_args()
        return parser, args