from model import *
import os
from data_preprocess import *
import random
import pickle
import pandas as pd
from torch.utils.data.sampler import Sampler
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
import torch
def train_auto(args):
    print('Training Autoencoder....\n')
    
    model = Autoencoder()
    
    folders = os.listdir(args.patch_dir)
    train_data = []
    random.shuffle(folders)
      
    for i in range(args.n_bag):
        patch_dir  = args.patch_dir + '\\' + folders[i]
        patches = pickle.load(open(patch_dir  +  '//patches.pkl'  , "rb"))
        for jj in range(len(patches) - len(patches)%batch_size):
            train_data.append(patches[jj][0])


    train_data = np.array(train_data)
    mse = tf.keras.losses.MeanSquaredError()
    opt = tf.keras.optimizers.Adam(lr = args.lr_a)

    model.compile(optimizer="adam", loss="mse")
    model.fit(train_data, train_data/255, batch_size = args.batch_a, epochs = args.epoch_a)
    print('Training Autoencoder is finished\n')
    
    base_model = tf.keras.models.Model(model.input,model.layers[-20].output)
    input = base_model.input
    output_layer = tf.keras.layers.GlobalAveragePooling2D()(input)
    encoder = tf.keras.models.Model(input, output_layer)
    
    encoder.save(args.auto_dir+'encoder.h5')
    print('Encoder is saved in '+ args.auto_dir[:-1]+'!!\n')
    
    
 
    
def Load_data_model1(args): 
    feature_dir = args.feature_dir +'//'
    label_dir = args.label_dir+'//'
    folders = os.listdir(feature_dir)
    features = []
    labels = []
    sequence_names =[]
    for i in range(len(folders)):
        features.append(pickle.load(open(feature_dir + folders[i] +  '//vectors.pkl'  , "rb"))) 
        labels.append(pickle.load(open(label_dir + folders[i] +  '//label.pkl'  , "rb"))[0][-1])
        sequence_names.append(folders[i])
    WG = dict(zip(sequence_names, features))
    zipped = list(zip(sequence_names, labels))
    ds = pd.DataFrame(zipped, columns=['names', 'stability'])
    return ds

def NodeM(name):
    return WG[name]


  
class Dataset(Dataset):

    def __init__(self, dataframe):
        self.names = dataframe['names'].values.tolist()
        self.labels = dataframe['stability'].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        sequence_name = self.names[index]
        label = self.labels[index]
        
        sequence_feature = NodeM(sequence_name)


        sample = {'sequence_feature': sequence_feature,\
                  'label': label, \
                  'sequence_name': sequence_name, \
                  }
        return sample
def collate_fn(batch):
    sequence_feature = []
    sequence_names = [] 
    labels=[]   
    for i in range(len(batch)):
        sequence_feature.append(batch[i]['sequence_feature'])
        sequence_feature=np.asarray(sequence_feature)
        sequence_names.append(batch[i]['sequence_name'])
        labels.append(batch[i]['label'])
        labels= np.asarray(labels)

    sequence_feature = torch.from_numpy(sequence_feature).float()
    labels= torch.from_numpy(labels)

    return sequence_feature, labels, sequence_names
def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor([item[1] for item in batch])
	return [img, label]


# Model parameters
# NUMBER_EPOCHS = 1000
# LEARNING_RATE = 5E-5
# WEIGHT_DECAY = 10E-6
# BATCH_SIZE = 1

# DENSE_DIM = 16
# ATTENTION_HEADS = 4
SEED = 2333

def train_one_epoch(model, data_loader, epoch):
    epoch_loss_train = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        model.optimizer.zero_grad()
        sequence_feature, labels, sequence_names = data

        sequence_feature = torch.squeeze(sequence_feature)
       
        if torch.cuda.is_available():
            features = Variable(sequence_feature.cuda())
            y_true = Variable(labels.cuda())
        else:
            features = Variable(sequence_feature)
            y_true = Variable(labels)
                    
        logits, y_pred, Y_hat, A, results_dict,M =  model(features)
        y_true = y_true
        y_pred = y_pred.mean(dim=0)
        Y_hat = torch.argmax(y_pred)        
        # calculate loss
        loss = model.criterion(logits, y_true.to(dtype=torch.long,non_blocking=False))
        # backward gradient
        loss.backward()

        # update all parameters
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n_batches += 1
    epoch_loss_train_avg = epoch_loss_train / n_batches
    return epoch_loss_train_avg

def train(model, dk,dn,fold=0, epochs = 1000, Model_Path = 'model1//',Result_Path = 'result1//'):
    train_loader = DataLoader(dataset=Dataset(dk) ,batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn)
    valid_loader = DataLoader(dataset=Dataset(dn) ,batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn)

    train_losses = []
    train_binary_acc = []

    valid_losses = []
    valid_binary_acc = []

    best_val_loss = 1000
    best_epoch = 0

    for epoch in range(epochs):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        print(epoch_loss_train_avg)

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, valid_name, valid_score,valid_m = evaluate(model, valid_loader)
        result_valid, binp,bint = analysis(valid_true, valid_pred)
        print("Valid binary acc: ", result_valid['binary_acc'])
        valid_binary_acc.append(result_valid['binary_acc'])
        if best_val_loss > epoch_loss_valid_avg:
            best_val_loss = epoch_loss_valid_avg
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_score_model.pkl'))
            valid_detail_dataframe = pd.DataFrame({'names': valid_name, 'stability': valid_true, 'prediction': valid_pred})
            valid_detail_dataframe.sort_values(by=['names'], inplace=True)
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_valid_detail.csv", header=True, sep=',')

def analysis(y_true, y_pred):
    #print(len(y_pred))
    binary_pred = y_pred
    #print(binary_pred, y_true)
    binary_true= y_true

   
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    result = {

        'binary_acc': binary_acc,

    }
    return result , binary_pred, binary_true



def Learn_discriminative_patches(args):
    print("split_seed: ", SEED)
    all_dataframe = Load_data_model1(args)
    
    fold = 0    
    model = Model1(gate = args.gate, hidden_dim = args.hid_dim, out_dim = args.out_dim, lr = args.lr,weight_decay = args.wd)
    if torch.cuda.is_available():
        model.cuda()
    train(model, all_dataframe, all_dataframe, fold + 1,epochs = args.epochs, Model_Path = args.model1_dir,Result_Path = args.result1_dir)
        

        
def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []
    valid_score =[]
    valid_m = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_feature, labels, sequence_names = data

            sequence_feature = torch.squeeze(sequence_feature)
                        
            if torch.cuda.is_available():
                features = Variable(sequence_feature.cuda())

                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_feature)
                y_true = Variable(labels)

            logits, Y_prob, Y_hat, A, results_dict,M =  model(features)
            y_true = y_true
            Y_prob = Y_prob.mean(dim=0)
            Y_hat = torch.argmax(Y_prob)

            loss = model.criterion(logits, y_true.to(dtype=torch.long,non_blocking=False))
           
            y_pred = Y_hat.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            valid_pred.append(y_pred)
            valid_true.append(y_true)
            valid_name.extend(sequence_names)
            valid_score.append(A)
            valid_m.append(M)

            epoch_loss += loss.item()
            n_batches += 1
    epoch_loss_avg = epoch_loss / n_batches
    #print(epoch_loss_avg)

    return epoch_loss_avg, valid_true, valid_pred, valid_name, valid_score,valid_m

def test_model1(args):
    test_dataframe = Load_data_model1(args)
    test_loader = DataLoader(dataset=Dataset(test_dataframe) ,batch_size=1, shuffle=False, num_workers=0,collate_fn=collate_fn)
    test_result = {}
    loss = 10000
    name = ''
    for model_name in sorted(os.listdir(args.model1_dir)):
        #print(model_name)
        model_s = Model1(gate = args.gate, hidden_dim = args.hid_dim, out_dim = args.out_dim, lr = args.lr,weight_decay = args.wd)
        if torch.cuda.is_available():
            model_s.cuda()
        model_s.load_state_dict(torch.load(args.model1_dir + model_name),strict=True)
        model_s.eval()
    
        epoch_loss_valid_avg, valid_true, valid_pred, valid_name, valid_score,valid_m = evaluate(model_s, test_loader)
        if epoch_loss_valid_avg < loss:
            loss = epoch_loss_valid_avg
            name = model_name
    
    
    model_s = Model1(gate = args.gate, hidden_dim = args.hid_dim, out_dim = args.out_dim, lr = args.lr,weight_decay = args.wd)
    if torch.cuda.is_available():
        model_s.cuda()
    model_s.load_state_dict(torch.load(args.model1_dir + name),strict=True)
    model_s.eval()

    epoch_loss_valid_avg, valid_true, valid_pred, valid_name, valid_score,valid_m = evaluate(model_s, test_loader)
    
        
    v=[]
    for i in range(len(valid_score)):
        v.append(valid_score[i].detach ().cpu ().numpy ())
    k = []
    for i in range(len(a)):
        k.append(v[i][0])
    
    label_dir = args.label_dir+'//'
    folders = os.listdir(label_dir)
    for i in range(len(folders)):
        os.makedirs(args.scores_dir + folders[i])
        with open(args.scores_dir + folders[i] + '//score.pkl', 'wb') as f:
            pickle.dump(k[i], f) 
    print('The scores were saved in: '+ args.scores_dir)
    #return valid_score



def load_data_model2(args):

    graph_dir = args.graph_dir+'//'
    label_dir = args.label_dir+'//'
    
    folders = os.listdir(graph_dir)
    sequence_features = []
    sequence_graphs = []
    labels = []
    sequence_names=[]
    for i in range(len(folders)):
        sequence_features.append(pickle.load(open(graph_dir + folders[i] +  '//graph.pkl'  , "rb"))['node_features'])
        sequence_graphs.append(pickle.load(open(graph_dir + folders[i] +  '//graph.pkl'  , "rb"))['Adjacency_matrix'])
        labels.append(pickle.load(open(label_dir + folders[i] +  '//label.pkl'  , "rb"))[0][-1])
        sequence_names.append(folders[i])
    
    MG = dict(zip(sequence_names, sequence_graphs))
    WF=dict(zip(sequence_names, sequence_features))
    zipped = list(zip(sequence_names, labels))
    ds = pd.DataFrame(zipped, columns=['names', 'stability'])
    return ds

class Dataset2(Dataset):

    def __init__(self, dataframe):
        self.names = dataframe['names'].values.tolist()
        self.labels = dataframe['stability'].values.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        sequence_name = self.names[index]
        label = self.labels[index]
        
        sequence_feature = NodeM(sequence_name)

        # L * L
        sequence_graph = GraphM(sequence_name)
        
        
        sample = {'sequence_feature': sequence_feature,\
                  'sequence_graph': sequence_graph, \
                  'label': label, \
                  'sequence_name': sequence_name, \
                  }
        return sample


def collate_fn2(batch):
    sequence_feature = []
    sequence_graph = []
    sequence_names = [] 
    labels=[]   
    for i in range(len(batch)):
        sequence_feature.append(batch[i]['sequence_feature'])
        sequence_feature=np.asarray(sequence_feature)
        sequence_graph.append(batch[i]['sequence_graph'])
        sequence_graph=np.asarray(sequence_graph)
        sequence_names.append(batch[i]['sequence_name'])
        labels.append(batch[i]['label'])
        labels= np.asarray(labels)

    sequence_feature = torch.from_numpy(sequence_feature).float()
    sequence_graph = torch.from_numpy(sequence_graph).float()
    labels= torch.from_numpy(labels)

    return sequence_feature,sequence_graph, labels, sequence_names


def train_one_epoch_model2(model, data_loader, epoch):

    epoch_loss_train = 0.0
    n_batches = 0
    for data in tqdm(data_loader):
        model.optimizer.zero_grad()
        sequence_feature,sequence_graph, labels, sequence_names = data

        sequence_feature = torch.squeeze(sequence_feature)
        sequence_graph = torch.squeeze(sequence_graph)
        if torch.cuda.is_available():
            features = Variable(sequence_feature.cuda())
            graphs = Variable(sequence_graph.cuda())
            y_true = Variable(labels.cuda())
        else:
            features = Variable(sequence_feature)
            graphs = Variable(sequence_graph)
            y_true = Variable(labels)

        logits,y_pred,Y_hat= model(features, graphs)

        y_true = y_true.float()
        y_pred = y_pred.squeeze(0)

        # calculate loss
        loss = model.criterion(logits, y_true.to(dtype=torch.long,non_blocking=False))

        model.optimizer.zero_grad()

        loss.backward()

      #update
        model.optimizer.step()

        epoch_loss_train += loss.item()
        n_batches += 1

    epoch_loss_train_avg = epoch_loss_train / n_batches
    return epoch_loss_train_avg


def evaluate_model2(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []
    valid_logit = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_feature,sequence_graph, labels, sequence_names = data

            sequence_feature = torch.squeeze(sequence_feature)
            sequence_graph = torch.squeeze(sequence_graph)
            if torch.cuda.is_available():
                features = Variable(sequence_feature.cuda())
                graphs = Variable(sequence_graph.cuda())
                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_feature)
                graphs = Variable(sequence_graph)
                y_true = Variable(labels)

            logits, y_pred, Y_hat = model(features, graphs)

            y_true = y_true.float()
            y_pred = y_pred.squeeze(0)

            loss = model.criterion(logits,y_true.to(dtype=torch.long,non_blocking=False))               
            y_pred = y_pred.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            valid_pred.append(y_pred)
            valid_true.append(y_true)
            valid_name.extend(sequence_names)
            valid_logit.append(logits)
            epoch_loss += loss.item()
            n_batches += 1
    epoch_loss_avg = epoch_loss / n_batches
    print(epoch_loss_avg)
    return epoch_loss_avg, valid_true, valid_pred, valid_name, valid_logit




def train_model2(model, dataframe,valid_dataframe,fold=0,epochs = 200, Model_Path = 'model2//',Result_Path = 'result2//'):
    train_loader = DataLoader(dataset=Dataset2(dataframe) ,batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn2)
    valid_loader = DataLoader(dataset=Dataset2(valid_dataframe) ,batch_size=1, shuffle=True, num_workers=0,collate_fn=collate_fn2)

    train_losses = []
    train_binary_acc = []

    valid_losses = []
    valid_binary_acc = []

    best_val_loss = 1000
    best_epoch = 0

    for epoch in range(epochs):
        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        epoch_loss_train_avg = train_one_epoch(model, train_loader, epoch + 1)
        print(epoch_loss_train_avg)

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred, valid_name ,valid_logit= evaluate(model, valid_loader)
        result_valid, binp,bint = analysis_model2(valid_true, valid_pred)
        print("Valid binary acc: ", result_valid['binary_acc'])
        valid_binary_acc.append(result_valid['binary_acc'])
        if best_val_loss > epoch_loss_valid_avg:
            best_val_loss = epoch_loss_valid_avg
            best_epoch = epoch + 1
            checkpoint = {'state_dict': model.state_dict()}
            torch.save(checkpoint,  os.path.join(Model_Path, 'Fold' + str(fold) + '_score_best_model.pkl'))    
            #torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + '_best_model.pkl'))
            valid_detail_dataframe = pd.DataFrame({'names': valid_name, 'stability': valid_true, 'prediction': valid_pred})
            valid_detail_dataframe.sort_values(by=['names'], inplace=True)
            valid_detail_dataframe.to_csv(Result_Path + 'Fold' + str(fold) + "_binary_valid_detail.csv", header=True, sep=',')
 

def analysis_model2(y_true, y_pred):
    #print(len(y_pred))
    binary_pred = y_pred
    #print(binary_pred, y_true)
    binary_true= y_true


    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    
    result = {

        'binary_acc': binary_acc,

    }
    return result , binary_pred, binary_true

def Train_model2(args):
    all_dataframe = load_data_model2(args)
    print("split_seed: ", SEED)
    sequence_names = all_dataframe['names'].values
    sequence_labels = all_dataframe['stability'].values
    kfold = KFold(n_splits=args.fold, shuffle=True)
    fold = 0

    for train_index, valid_index in kfold.split(sequence_names, sequence_labels):
        print("\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Training on", str(train_dataframe.shape[0]), "examples, Validation on", str(valid_dataframe.shape[0]),
              "examples")
        if args.GAT:
            model = GAT(gcn_feature_dim = 2560, gcn_hid_dim = args.hid_dim, gcn_out_dim = args.out_dim,dense_dim = args.dense_dim,gat_heads=args.gat_heads, n_heads = args.n_heads, n_class = args.n_class, lr = args.lr,weight_decay = args.wd)
        else:
            model= Model2(gcn_feature_dim = 2560, gcn_hid_dim = args.hid_dim, gcn_out_dim = args.out_dim,dense_dim = args.dense_dim,n_heads = args.n_heads, n_class = args.n_class, lr = args.lr,weight_decay = args.wd)

        train_model2(model, train_dataframe, valid_dataframe, fold + 1,epochs = args.epoch, Model_Path = args.model2_dir,Result_Path = args.result2_dir)
        fold += 1











def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n_batches = 0
    valid_pred = []
    valid_true = []
    valid_name = []
    valid_score =[]
    valid_m = []

    for data in tqdm(data_loader):
        with torch.no_grad():
            sequence_feature, labels, sequence_names = data

            sequence_feature = torch.squeeze(sequence_feature)
                      
             
            if torch.cuda.is_available():
                features = Variable(sequence_feature.cuda())
               

                y_true = Variable(labels.cuda())
            else:
                features = Variable(sequence_feature)
                y_true = Variable(labels)

            logits, Y_prob, Y_hat, A, results_dict,M =  model(features)
            y_true = y_true
            Y_prob = Y_prob.mean(dim=0)
            Y_hat = torch.argmax(Y_prob)

            loss = model.criterion(logits, y_true.to(dtype=torch.long,non_blocking=False))

            y_pred = Y_hat.cpu().detach().numpy().tolist()
            y_true = y_true.cpu().detach().numpy().tolist()
            valid_pred.append(y_pred)
            valid_true.append(y_true)
            valid_name.extend(sequence_names)
            valid_score.append(A)
            valid_m.append(M)

            epoch_loss += loss.item()
            n_batches += 1
    epoch_loss_avg = epoch_loss / n_batches
    print('Average loss value for validation data: ', epoch_loss_avg)

    return epoch_loss_avg, valid_true, valid_pred, valid_name, valid_score,valid_m 