# WSI-Hepatotoxicity-Project
A graph convolutional model for predicting drug-induced hepatotoxicity based on histopathological whole slide images.


# Keywords
Hepatotoxicity, Graph Convolution, Prediction, Liver, Whole Slide Image, Deep Learning


# Dataset
[Open TG-GATEs](http://togodb.biosciencedbc.jp/togodb/view/open_tggates_pathological_image#en), [Hepatotocxicity Label](https://pubs.acs.org/doi/full/10.1021/tx200148a?casa_token=T5X3K3jwh9oAAAAA%3ACupsGmTKQY9LWO1QhMc-V1uH8jO_FhycViClpeur6OoCSC3Nqy8LgO77J5Pi3sX3PhcNTSXnCKuh-Wcj)


# Requires
  Python version 3.7.0  

  torch==1.6.0  
  torchvision==0.7.0  
  openslide==1.1.2  
  Pillow==8.1.0  
  numpy==1.19.5  
  scikit-learn==1.0.1  
  scipy==1.6.3  
  matplotlib==3.4.1  


# Orgainzation
## *Code*
* **extract_patch.py** is for extracting patches from WSIs through data argumentation.
* **cluster_sample.py** is for clustering the patches of each WSI by K-means menthod and sampling a total of 300 patches.
* **graph_represent.py** is for extract features of patches by pre-trained CNN then completing the graph representation process.
* **model.py** is our proposed deep graph convolution prediction model.
* **model.py** is a data dictionary class for easier use of data.
* **train_test.py** is for training the prediction model using graph structure data by 5-fold cross-validation and finally outputting the prediction results of drug hepatotoxicity.
## *File*
* **label.txt** is for storing the drug name as well as the hepatotoxicity label of each WSI file.
* **wsiname.txt** is for storing the file name of each WSI.
* **hepatotoxicity.xlsx** is for storing the drug hepatotoxicity informations we have compiled, which can be used for reference.

 
# How to Use
**Data:** Download the dataset or use your own WSI data & put the label in 'label.txt'.  
**Path:** Create a path to store WSI data and modify the path in 'extract_patch.py'.  
**Run:** extract_patch.py->cluster_sample.py->graph_represent.py->train_test.py.  