# Adaptive k-nearest neighbor

Beta Distribution-based cross entropy (BetaDCE) is a general framework for variable selection in high dimensional data. BetaDCE generalizes well without the requirement of resampling such as cross validation. Here is a python implementation which can take advantage of multiple CPU cores to significantly accelerate the running time.

|Author|Weixing Dai|
|---|---
|E-mail|1155038704@link.cuhk.edu.hk

## Example

### To train:

```Bash
python betadceTrain.py --data 'data/leukemia_train.csv'
```
The result of training on leukemia data set:

![github](https://github.com/mlalgorithm/imageache/blob/master/aknn_train.png)  

### To predict:

```Bash
python betadcePredict.py --data 'data/leukemia_test.csv'
```
The result of prediction on leukemia test set by the trained model:

![github](https://github.com/mlalgorithm/imageache/blob/master/aknn_test.png)  

## Usage
```Bash
betadceTrain.py
```
Function: Training by Adaptive k-nearest neighbor

usage: betadceTrain.py [-h] [--data D] [--ne N] [--out O] [--cpu C]

optional arguments:  
-h, --help　  show this help message and exit  
--data D　    data to be trained, the last column is the labels (required)  
--ne N 　     maximum number of feature subsets evaluated in each epoch. the default is 2000000 (optional)  
--out O 　    filename to keep the output of training. the default is lbsmodel.npz (optional)    
--cpu C 　    the number of CPUs to use. the default is to use all of CPUs available (optional)  

```Bash
aknnPredict.py
```
Function: Prediction on the test set by the trained model 

usage: betadcePredict.py [-h] [--data D] [--model M] [--result R]

optional arguments:  
-h, --help 　 show this help message and exit  
--data D  　  data to be predicted, the last column is the labels (required)  
--model M 　  filename of the model trained. the default is lbsmodel.npz(optional)  
--result R 　 filename to keep the result of prediction. the default is Result.txt (optional)  

