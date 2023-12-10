# CLTSBR: Collaborative Long-tail Session-based Recommendations 

This is the implementation of our paper: <span style="color:blue">Collaborative Long-tail Session-based
Recommendations</span> using **pytorch**

Briefly, Session-Based Recommendation Systems (SBRS) often favor popular items, neglecting diverse, 'long-tail' items. This limits user exploration and undervalues long-tail items, a significant part of item catalogs. Existing solutions fail to use collaborative information from sessions with similar diversity profiles, where diversity is the ratio of popular to unpopular items. Our paper introduces a novel neural network architecture that improves long-tail recommendations without losing accuracy. It integrates popularity into item embeddings and uses collaborative information from similar sessions, adjusted for diversity. Tested on Diginetica, YooChoose, and Cosmetics datasets, our model significantly outperforms baseline models in accuracy and coverage, demonstrating its effectiveness in enhancing long-tail recommendations while maintaining accuracy.

Here is the link for the diginetica dataset used in our paper. 
- DIGINETICA: <https://competitions.codalab.org/competitions/11161#learn_the_details-data2>
   
After downloading the datasets, you can extract the files in the folder `datasets/`:



## Usage

You need to run the file  `data_preprocess.ipynb` first to preprocess the data. 

Then you can run the file `main.ipynb` to train the model. The model hyperparameters can be changed from the  `main.ipynb` file itself.

## Requirements

- Python version > 3.8
- PyTorch > 1.10
- numpy > 1.20
- pandas > 1.3
- math

## Citation

Please cite our paper if you use the code.

