# importing packages
import pandas as pd
import numpy as np
import os
import shutil
import mlflow
import mlflow.sklearn
import sys
sys.path.append("..")
from src.score import score

# model training
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split

# defining dictionary to map our predictions to, for our custom score function
lithology_keys = {
    30000: 0,
    65030: 1,
    65000: 2,
    80000: 3,
    74000: 4,
    70000: 5,
    70032: 6,
    88000: 7,
    86000: 8,
    99000: 9,
    90000: 10,
    93000: 11
}


# function for handling model training
def train_model(df, target, random_seed=53):
    np.random.seed(random_seed)
    '''Train a Random Forest Lithofacies Classifier'''
    
    # end any active run
    mlflow.end_run()
    
    # starting an mlflow run
    mlflow.start_run()
    
    # columns to drop from passed dataset
    cols_to_drop =["SGR", "DTS", "ROP", "DCAL", "MUDWEIGHT", "RMIC", "ROPA", "RXO", "BS"]
    
    # target variable
    target = target.map(lithology_keys)
    
    # dropping features with >50% null values
    df = df.drop(cols_to_drop, axis=1)
    
    # getting list of numerical and categorical attributes
    num_attribs = ['DEPTH_MD', 'X_LOC', 'Y_LOC', 'Z_LOC', 'CALI', 'RSHA', 'RMED', 'RDEP',
       'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'DRHO']
    
    cat_attribs = ["FORMATION", "GROUP"]
    
    # Transformer class for dataframe selection based on attribute type
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attrib_names):
            self.attrib_names = attrib_names
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return X[self.attrib_names]
        
    # Transformer class to replace/constrain outliers in the dataset
    class ReplaceOutliers(BaseEstimator, TransformerMixin):
        def __init__(self, ):
            pass
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            low = 0.01
            high = 0.99
            X = pd.DataFrame(X, columns=[*range(X.shape[1])])
            for col in X:
                X[col] = np.where((X[col] > X[col].quantile(high)), X[col].quantile(high), X[col])
                X[col] = np.where((X[col] < X[col].quantile(low)), X[col].quantile(low), X[col])
            return X
            
        
    # numerical pipeline to apply Transformers to numerical features
    num_pipeline = Pipeline([
        ("Select dataframe", DataFrameSelector(num_attribs)),
        ("imputer", SimpleImputer(strategy="median")),
        ("outliers", ReplaceOutliers()),
        ("scaler", StandardScaler()),
        #('poly', PolynomialFeatures(degree=2, include_bias=False)) 
    ])
    
    # category pipeline to apply Tranformers to categorical features
    cat_pipeline = Pipeline([
        ("Select dataframe", DataFrameSelector(cat_attribs)),
        ("OneHot encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    # full transformation pipeline
    transformation_pipeline = FeatureUnion(transformer_list=[
        ("numeric pipeline", num_pipeline),
        ("category pipeline", cat_pipeline)
    ])
    
    # applying full transformation pipeline on entire dataset
    train_full = transformation_pipeline.fit_transform(df)
    
    print("step 1: Data Transformation Complete!\n")
    
    # log full transformation pipeline
    mlflow.sklearn.log_model(transformation_pipeline, "transformer")
    
    # spitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(train_full, target, test_size=0.25, random_state=random_seed, stratify=target) 
    
    # converting data sets into xgboosts data matrix format to speed up training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # creating an evaluation list that will be used during training
    evals = [(dtest, 'eval'), (dtrain, 'train')]
    
    # xgboost parameters
    params = {
        'seed': 53,
        'objective': 'multi:softmax',
        'num_class': 12,
        'max_depth': 6,
        'eta': 0.3,
        'eval_metric': 'mlogloss',
        'device': 'cuda',
        'reg_lambda': 1500,
        'subsample': 0.9,
        'colsample_bytree': 0.6,
        'min_child_weight': 10,
        'gamma': 0.1,
        'n_estimators': 200
    }
    
    # logging parameters in mlflow
    mlflow.log_params(params)
    
    early_stopping, num_boost_round = 20, 100
    
    # Train an XGboost Lithology Facies Classification model with the training set
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, 
                    evals=evals, early_stopping_rounds=early_stopping,
                    verbose_eval=False)
    
    
    print("\nstep 2: Model Trained!\n")
    
    # making predictions on test set
    y_pred = bst.predict(dtest)
    
    # Evaluating model performance
    f1score = f1_score(y_test, y_pred, average="weighted")
    score_ = score(y_test.values, y_pred)
    
    # log model performance metrics
    mlflow.log_metrics({
        "f1_score": f1score,
        "custom_score": score_
    })
    
    # log base model using mlflow
    mlflow.xgboost.log_model(bst, "base_model")
    
    print("\nstep 3: Model & Transformer have been logged!\n")
    
    # registering logged model
    run_id = mlflow.active_run().info.run_id
    base_uri = f'runs:/{run_id}/base_model'
    scaler_uri = f'runs:/{run_id}/transformer'    
    
    # Pyfunc class for custom prediction -> handles transformation and prediction together during inference
    class CustomPredict(mlflow.pyfunc.PythonModel):
        def __init__ (self, ):
            self.transformation_pipeline = mlflow.sklearn.load_model(scaler_uri)
            
        def process_inference_data(self, model_input):
            model_input = model_input.drop(cols_to_drop, axis=1)
            model_input = self.transformation_pipeline.transform(model_input)
            model_input = xgb.DMatrix(model_input)
            return model_input
        
        def load_context(self, context=None):
            self.model = mlflow.xgboost.load_model(base_uri)
            return self.model
        
        def predict(self, context, model_input):
            model = self.load_context()
            model_input = self.process_inference_data(model_input)
            predictions = model.predict(model_input)
            return predictions
        
    
    # saving the custom model
    mlflow.pyfunc.log_model("lithology_classifier", python_model=CustomPredict())
    
    print("\nstep 4: Custom model logged!")

    mlflow.end_run()   