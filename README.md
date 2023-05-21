# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python nonstandardcode.py

## Steps to run the code
1. Clone the `mle-training` repository.
2. Setup the conda environment by running the following command `conda env create -f env.yml` and run `conda activate mle-dev` to activate the environment.
3. Run the python script `python nonstandardcode.py`.
4. After running for a few minutes, similar results will be printed to terminal:
```
49175.920081996424 {'max_features': 7, 'n_estimators': 180}
50979.31309987477 {'max_features': 5, 'n_estimators': 15}
50615.31290608045 {'max_features': 3, 'n_estimators': 72}
50356.50620069724 {'max_features': 5, 'n_estimators': 21}
49356.66118397995 {'max_features': 7, 'n_estimators': 122}
50590.24171016077 {'max_features': 3, 'n_estimators': 75}
50439.661883382774 {'max_features': 3, 'n_estimators': 88}
49498.226752242146 {'max_features': 5, 'n_estimators': 100}
50262.550516472504 {'max_features': 3, 'n_estimators': 150}
63055.85950859079 {'max_features': 5, 'n_estimators': 2}
64062.72861884957 {'max_features': 2, 'n_estimators': 3}
55479.21774655382 {'max_features': 2, 'n_estimators': 10}
52966.124941638336 {'max_features': 2, 'n_estimators': 30}
58363.665560496614 {'max_features': 4, 'n_estimators': 3}
52396.19523340483 {'max_features': 4, 'n_estimators': 10}
50215.10958041685 {'max_features': 4, 'n_estimators': 30}
59020.00126496427 {'max_features': 6, 'n_estimators': 3}
52006.13620481499 {'max_features': 6, 'n_estimators': 10}
50051.59528363614 {'max_features': 6, 'n_estimators': 30}
58910.833233016325 {'max_features': 8, 'n_estimators': 3}
52362.774093865475 {'max_features': 8, 'n_estimators': 10}
50273.65306916561 {'max_features': 8, 'n_estimators': 30}
63087.194157215 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
54773.96059466593 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
60379.84359138394 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
51887.898627977534 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
57719.276994840606 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
51419.866104752255 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
```
