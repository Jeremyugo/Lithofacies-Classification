import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import mlflow
import mlflow.sklearn
import sys
sys.path.append("..")

from src.score import score
from src.utility_functions import preprocess_data

# model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split

lithology_keys = {30000: 'Sandstone',
                 65030: 'Sandstone/Shale',
                 65000: 'Shale',
                 80000: 'Marl',
                 74000: 'Dolomite',
                 70000: 'Limestone',
                 70032: 'Chalk',
                 88000: 'Halite',
                 86000: 'Anhydrite',
                 99000: 'Tuff',
                 90000: 'Coal',
                 93000: 'Basement'}


def remove_files(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                # Remove the file or link
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                # Remove the directory and its contents
                shutil.rmtree(item_path)
        except Exception as e:
            print(f'Failed to delete {item_path}. Reason: {e}')


def train_model(df, target):
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
    num_attribs = df.select_dtypes(include=[int, float]).columns
    cat_attribs = df.select_dtypes(include=[object]).columns
    
    # Transformer class for dataframe selection based on attribute type
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attrib_names):
            self.attrib_names = attrib_names
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            return X[self.attrib_names]
    
    # Transformer class for filling missing values in categorical dataframe subset
    class FillCategoricalMissingValues(BaseEstimator, TransformerMixin):
        def __init__(self, ):
            pass
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            for col in X.columns:
                X.loc[:, col] = X[col].fillna('unkwn')
            return X
    
    # Transformer class to embed categorical features using Keras StringLookup & Embedding Layers
    class KerasCategoryTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, ):
            pass
            
        def fit(self, X, y=None):
            return self
        
        def transform(self, X):
            str_lookup = keras.layers.StringLookup(num_oov_indices=2)
            str_lookup.adapt(X)
            lookup_and_embed = keras.Sequential([
                str_lookup,
                keras.layers.Embedding(input_dim=str_lookup.vocabulary_size(), output_dim=8)
            ])
            return lookup_and_embed(X.to_numpy()).numpy().reshape(X.shape[0], -1)
      
        
    # categorical pipeline to apply Transformers to categorical features    
    cat_pipeline = Pipeline([
        ("Select dataframe", DataFrameSelector(cat_attribs)),
        ("Fill Missing Values", FillCategoricalMissingValues()),
        ("Embed categorical values", KerasCategoryTransformer())
    ])
    
    # numerical pipeline to apply Transformers to numerical features
    num_pipeline = Pipeline([
        ("Select dataframe", DataFrameSelector(num_attribs)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # full transformation pipeline 
    full_pipeline = FeatureUnion(transformer_list=[
        ('numerical', num_pipeline),
        ('cat_pipeline', cat_pipeline)
    ])
    
    print("step 1: Data Transformation Complete!")
    
    # applying full transformation pipeline on entire dataset
    train_full = full_pipeline.fit_transform(df)
    
    # log full transformation pipeline
    mlflow.sklearn.log_model(full_pipeline, "transformer")
    
    # spitting dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(train_full, target, test_size=0.2, random_state=42, stratify=target)
    
    # Train a default Random Forest Lithology Facies Classification model with the training set
    rnd_clf = RandomForestClassifier(random_state=42)
    rnd_clf.fit(X_train, y_train)
    
    print("step 2: Model Trained!")
    
    # making predictions on test set
    y_pred = rnd_clf.predict(X_test)
    
    # Evaluating model performance
    f1score = f1_score(y_test, y_pred, average="weighted")
    
    # log model performance metric
    mlflow.log_metric("f1_score", f1score)
    
    # saving transformer and base model using mlflow
    mlflow.sklearn.save_model(full_pipeline, path="../model/transformer")
    mlflow.sklearn.save_model(rnd_clf, path="../model/base_model")
    
    print("step 3: Models saved to base path!")
    
    # log base model using mlflow
    mlflow.sklearn.log_model(rnd_clf, "base_model")
    
    # registering logged model
    run_id = mlflow.active_run().info.run_id
    base_uri = f'runs:/{run_id}/base_model'
    scaler_uri = f'runs:/{run_id}/transformer'
    model_uri = f'runs:/{run_id}/lithology_classifier'
    
    
    class CustomPredict(mlflow.pyfunc.PythonModel):
        def __init__ (self, ):
            self.full_pipeline = mlflow.sklearn.load_model(scaler_uri)
            
        def process_inference_data(self, model_input):
            model_input = model_input.drop(cols_to_drop, axis=1)
            model_input = self.full_pipeline.transform(model_input)
            return model_input
        
        def load_context(self, context=None):
            self.model = mlflow.sklearn.load_model(base_uri)
            return self.model
        
        def predict(self, context, model_input):
            model = self.load_context()
            model_input = self.process_inference_data(model_input)
            predictions = model.predict(model_input)
            return predictions
        
    
    # saving the custom model
    mlflow.pyfunc.log_model("lithology_classifier", python_model=CustomPredict())
    
    print("step 4: Custom model logged!")
    mlflow.end_run()
    remove_files("../model/base_model/")
    remove_files("../model/transformer/")
    