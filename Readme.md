## Tutorial

This is the code for our paper titled ["Weakly-Supervised Deep Learning Model for Prostate Cancer Diagnosis and Gleason Grading of Histopathology Images".](https://arxiv.org/abs/2212.12844)

### Dependencies

- Pandas == 1.3.5
- NumPy == 1.21.6
- skimage == 0.18.3
- tiatoolbox == 1.0.1
- opencv == 4.5.5
- Tensorflow == 2.3
- sklearn == 1.0.2
- tqdm == 4.64.0
- scipy == 1.7.3
- Pytorch == 1.12.1

### Code Usage

Running the code has 6 steps which are as follows:

- Step 1: Cropping the WSIs to small patches.
  - The following command can be used to run the code for this step:
    - `--mode`: for this step model should be `crop`
    - `--wsi_dir`: the directory of the WSIs, default = 'images/'
    - `--labels`: the directory of the clinical file containing the scores. It should be a csv file containing a column named `image_id` for the name of images, a column named `primary_gleason_grade`, and another column named `secondary_gleason_grade` for the primary and secondary Gleason grades. default = 'clinical.csv'
    - `--patch_size`: size of small patches. default = 512
    - Note: this command saves the patches and labels in `Preprocessed_data//patches` and `Preprocessed_data//labels`, respectively.
- Step 2: Train the Autoencoder.
  - The following commands can be used to train the autoencoder:
    - `--mode`: for this step model should be `auto`
    - `--n_bag`: number of bags to train the autoencoder. default = 500
    - `--lr_a`: learning rate for autoencoder. default = 1e-3
    - `--batch_a`: batch size for autoencoder. default = 4
    - `--epoch_a`: number of epochs to train autoencoder. default = 50
    - `--auto_dir`: path to save autoencoder. default = autoencoder//
    - Note: this command reads the patches from `Preprocessed_data//patches`.
- Step 3: Extract the feature vectors from small patches using the encoder.
  - The following commands can be used to extract the features:
    - `--mode`: for this step model should be `feature`
    - `--auto_dir`: path to the autoencoder. default = autoencoder//
    -  Note: this command reads the patches from `Preprocessed_data//patches` and saves the features in `Preprocessed_data//features`.
- Step 4: Train the first stage of the model and extract the scores.
  - The following commands should be used for this step:
    - `--mode`: for this step model should be `model1`
    - `--state1`: it can be either `traintest` to train and save the scores or `test` to load the model and extract the scores. default = traintest
      - Note: the model will be saved/loaded from `Models//model1` 
    - `--score_dir`: the path to save the scores. default = `Preprocessed_data//scores`
    - `--epoch`: number of epochs. default = 1000
    - `--lr`: learning rate. default = 5e-5
    - `--wd`: weight decay. default = 10e-6
    - `--gate`: whether using gated attention or not. default = True
    - `--hid_dim`: hidden dimension in the linear layer. default = 512
    - `--out_dim`: the output dim in the linear layer. default = 256.
    - `--n_class`: number of class in the data. default = 5.
    - Note: this command reads the features from `Preprocessed_data//features` and labels from `Preprocessed_data//labels`. 
- Step 5: Create the graphs from the discriminative patches.
  - The following commands are for this step:
    - `--mode`: for this step model should be `graph`
    - `--scores_dir`: path to the scores extracted by the first stage of the model. default = Preprocessed_data//scores//
    - `--n_neighbor`: number of neighbors in KNN. default = 10
    - `--top`: choose the top% of high score. default = 40
    - Note: the graphs will be saved in `Preprocessed_data//graphs`. 
- Step 6: Train the graph convolutional neural network.
  - The following commands are used for this step:
    - `--mode`: for this step model should be `model2`
    - `--epoch`: number of epochs. default = 1000
    - `--lr`: learning rate. default = 5e-5
    - `--wd`: weight decay. default = 10e-6
    - `--GAT`: whether using GAT or GCN. default = False
    - `--hid_dim`: hidden dimension in the linear layer. default = 512
    - `--out_dim`: the output dims in the linear layer. default = 256.
    - `--n_class`: number of classes in the data. default = 5.
    - `--n_heads`: number of attention layers. default = 6.
    - `--gat_heads`: number of gated attention layers. default = 4.
    - `--fold`: number of folds in the cross-validation. default = 10.
    - `--dense_dim`: dim of the last linear layer. default = 16.
    - Note: this command reads the graph from `Preprocessed_data//graphs` and labels from `Preprocessed_data//labels`. 

### Using Example

For step 1:

```python
python main.py --mode crop --wsi_dir images// --labels clinical.csv --patch_size 512
```

For step 2:

```python
python main.py --mode auto --n_bag 200 --epoch_a 25
```

For step 3:

```python
python main.py --mode feature --auto_dir Autoencoder//
```

For step 4:

```python
python main.py --mode model1 --state1 traintest --epoch 2500
```

For step 5:

```python
python main.py --mode graph --n_neighbor 15 --top 50
```

For step 6:

```python
python main.py --mode model2 --lr 1e-5 --wd 1e-6 --GAT False --fold 5 --n_heads 4
```



## Citation
```
@misc{behzadi2022weaklysupervised,
      title={Weakly-Supervised Deep Learning Model for Prostate Cancer Diagnosis and Gleason Grading of Histopathology Images}, 
      author={Mohammad Mahdi Behzadi and Mohammad Madani and Hanzhang Wang and Jun Bai and Ankit Bhardwaj and Anna Tarakanova and Harold Yamase and Ga Hie Nam and  Sheida Nabavi},
      year={2022},
      eprint={2212.12844},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
