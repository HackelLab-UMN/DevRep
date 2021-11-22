# DevRep

Gp2 Developability Modeling, Lead Author: Alexander Golinski 

Contact Information
golin010@umn.edu

Hackel & Martiniani Labs
University of Minnesota

This project contains Python3 scripts used to predict the yield of Gp2 paratope variants 
utilizing high-throughput developability assays.

There are brief examples of how to use the code for the most predictive models for each aim of the project
in the main directory under main*example.py. To run, unzip datasets located at ./datasets/ and ./datasets/predicted/.
These scripts should run without without any other modifications. Required packages are included in the conda_package_list.txt.

The additional code for aim one of the project, determining the most predictive HT assays, 
can be found in ./main_paper_one/

The additional code for aim two of the project, creating a sequence-based model to predict yield 
via transfer learning of DevRep, can be found in ./main_paper_two/

Files in ./main_paper_*/ folders need to me moved to the main directory to run. 

For non-top performing models, saved hyperparameter trials and model stats can be found within the zipped folder in the repective folders.


To create the environment with the conda package manager run from the command line

`conda create --name <env>  python=3.7.5 tensorflow=2.0.0 numpy=1.17.4 pandas=0.25.3 seaborn=0.10.1 scikit-learn=0.22 matplotlib`

Where `<env>` is your environment name. Then from the commmand line type: 

`conda activate <env>`

`conda install -c conda-forge hyperopt=0.2.2` 

The environment for DevRep is now setup!

File descriptions:
model_module.py - base model class that defines how to cross-validate, test, and evaluate model performances.
submodels_module.py - subclasses that modify the model inputs/outputs and datasets for model evaluation.
model_architectures.py - describes the hyperparameters and construction of the possible model architectures.
plot_model.py - helper class to plot the predicted results from cv and testing.
load_format_data.py - helper functions to format the data from the pickeled DataFrames to useful inputs for model evaluations.

Folder descriptions:
/datasets/ - location of saved sequences' yields and assay scores.
*Due to GitHub size limits, you will have to unzip the datasets and the example predicted datasets*
*Datasets are a pickeled DataFrame, which can be opened via panda.read_pickle(<filename>)*
CSV examples of the smaller datasets are also included.

/trials/ - location of hyperparameter trials during cross-validation saved as pickeled hyperopt files.
/model_stats/ - location of the best cv- and test- performance of the models
/models/ - location of saved models as either pickeled scikit-learn models or tensorflow2 weights
/plotpairs/ - location of saved pairs of (predicted value, true value, strain or assay id)
/figures/ - location of the saved predicted figures for cv and testing


