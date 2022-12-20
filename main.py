from config import * 
from trainer import * 
from data_preprocess import *


cfg = Config()

if cfg.mode == 'crop':
    create_labels(cfg)
    crop_slides(cfg)

if cfg.mode =='auto':
    train_auto(cfg)
    
if cfg.mode == 'feature':
    feature_extraction(cfg)
    
if cfg.mode == 'graph':
    build_graphs(args)
    
if cfg.mode == 'model1':
    if cfg.state1 == 'traintest':
        Learn_discriminative_patches(cfg)
        test_model1(args)
    elif cfg.state1 == 'test':
        test_model1(args)
        
if cfg.mode =='model2':
    Train_model2(args)

