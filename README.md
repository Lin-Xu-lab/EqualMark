# EqualMarker - Python code examples

## Overview
![EqualMarker workflow](/assets/workflow.png)

## Dependencies
- pandas
- numpy
- scikit-learn
- pytorch
- tqdm
- seaborn
- matplotlib
- openTSNE
- umap
- scipy

## Usage

### Traditional methods for comparison
<code>scripts/references-asd.py</code> contains the codes for training and evaluating traditional classification methods for comparison.
Specifically, it will train four different types of models:
- Random forest (RF);
- Decision tree (DT);
- Linear support vector machine (linear_SVM);
- C-support vector classifier (C_SVC).

The script can be directly run with <code>python references-asd.py</code>.
It will load the example data set and one set of biomarkers, and then train and evaluate the models with respect to three available splitting factors in the data set.
The three factors are gender, paternal age, and maternal age.
When splitting by gender, the data samples will be split into two groups: male and female.
When splitting by paternal age or maternal age, the data samples will be split into two groups by age median.
Results will be written to a table file named <code>references-asd.tsv</code>.

### Joint model, one-shot model, and EqualMarker
<code>scripts/mp-type-ASD.py</code> contains the codes for training and evaluating the joint model, one-shot model, and EqualMarker.
It performs a grid search on all available hyper-parameter combinations, and outputs all results in two table files.
The <code>results_single.tsv</code> contains results for the joint model, and the <code>results.tsv</code> contains results for the one-shot model and EqualMarker.
This script has the following necessary arguments:
- gene_list: Name of the biomarker set. One example biomarker set is hard-coded from line 881 to 900.
- data_source: Name of the data set to load.
- factor: Name of the factor used to split the whole data set into different groups.
- target_col: Name of the target column to be predicted.

It also accepts the following optional arguments:
- shared_output_flag: A bool variable to indicate whether task-specific layers should be shared among different groups. The default value is True.
- middle_inflation: A bool variable to indicate whether neural networks' middle layer should be larger than the input/output layer or not. The default value is True.
- gpu_device: The index of the GPU device to use. The default value is 0.
- criteria: The target metric to balance. The default value is "accuracy". Other acceptable values include "precision", "recall", "f1" for F1 score, "bacc" for balanced accuracy, and "roc_auc" for ROC AUC score.
- n_parallel: Number of parallel processes during grid search. For VRAM of 8 GB or less, a value less than 10 is recommended. The default value is 6.

To run this script with the provided data set ([GSE25507](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25507)) and biomarker set ([Oh et al. 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5290715/)), a quick start is to run the following command. Note that it may take about 30 minutes.

`
python mp-type-ASD.py --gene_list "cpn-15-047" --data_source "GSE25507_example" --criteria "roc_auc" --factor "paternal age" --target_col "diagnosis" --n_parallel 10
`

### Visualize results
<code>scripts/draw-images.ipynb</code> contains the example codes to visualize the results.
Specifically, it draws the following charts:
- Data set description: The number of data samples in each group and their survival status.
- Dimension reduction: Dimension reduction results using TSNE and UMAP.
- Differential expression analysis.
- Model performance bar plot.
- Adjusted metric bar plot.
- Disparity bar plot.
