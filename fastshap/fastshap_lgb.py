import sys
import os
import shap
import numpy as np
import pickle
import lightgbm as lgb
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from fastshap_torch.utils import MaskLayer1d
from fastshap_torch import Surrogate, SoftCrossEntropyLoss, FastSHAP

class FastSHAPExperiment:
    datasets = {
        "Churn": ("../03_bank_churn/Churn.csv", "Exited"),
        "diabetes": ("../02_diabetes/diabetes.csv", "Diabetes"),
        # Add more datasets here
    }

    def __init__(self, dataset_name, random_state=2333):
        self.dataset_name = dataset_name
        self.random_state = random_state
        self.data_path, self.target_column = self.datasets[dataset_name]
        self.model_dir = f"./models/"
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def load_and_preprocess_data(self):
        # Load data
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=[self.target_column])
        Y = df[self.target_column]

        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=self.random_state)
        X_train, X_val, Y_train, Y_val = train_test_split(
            X_train, Y_train, test_size=0.2, random_state=self.random_state)
        
        # Preprocess
        self.ss = StandardScaler()
        X_train_std = self.ss.fit_transform(X_train)
        X_val_std = self.ss.transform(X_val)
        X_test_std = self.ss.transform(X_test)
        
        self.enc = OneHotEncoder(sparse=False)
        Y_train_oh = self.enc.fit_transform(Y_train.values.reshape(-1, 1))
        Y_val_oh = self.enc.transform(Y_val.values.reshape(-1, 1))
        Y_test_oh = self.enc.transform(Y_test.values.reshape(-1, 1))
        
        self.num_features = X_train.shape[1]
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, X_train_std, X_val_std, X_test_std, Y_train_oh, Y_val_oh, Y_test_oh

    def train_lightgbm_model(self, X_train, Y_train, X_val, Y_val, params=None):
        model_path = os.path.join(self.model_dir, f"{self.dataset_name}_model.pkl")
        if os.path.exists(model_path):
            print("Loading existing LightGBM model...")
            with open(model_path, 'rb') as f:
                return pickle.load(f)

        print("Training new LightGBM model...")
        if params is None:
            params = {
                "max_bin": 512,
                "learning_rate": 0.05,
                "boosting_type": "gbdt",
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 10,
                "verbose": -1,
                "min_data": 100,
                "boost_from_average": True
            }
        
        d_train = lgb.Dataset(X_train, label=Y_train)
        d_test = lgb.Dataset(X_val, label=Y_val)
        
        model = lgb.train(params, d_train, 10000, valid_sets=[d_test], 
                          callbacks=[lgb.early_stopping(stopping_rounds=50),
                                     lgb.log_evaluation(1000)])
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return model

    def prepare_surrogate_data(self, model, X_train, X_val):
        Y_train_surrogate = model.predict(X_train)
        Y_train_surrogate = np.vstack([1 - Y_train_surrogate, Y_train_surrogate]).T
        Y_val_surrogate = model.predict(X_val)
        Y_val_surrogate = np.vstack([1 - Y_val_surrogate, Y_val_surrogate]).T
        
        p = Y_train_surrogate.mean(axis=0)
        soft_ce = - np.mean(np.sum(np.log(p) * Y_train_surrogate, axis=1))
        print('Loss given no information = {:.4f}'.format(soft_ce))
        
        return Y_train_surrogate, Y_val_surrogate

    def train_surrogate_model(self, X_train_std, Y_train_surrogate, X_val_std, Y_val_surrogate, surrogate=None):
        surrogate_path = os.path.join(self.model_dir, f"{self.dataset_name}_surrogate.pt")
        if os.path.exists(surrogate_path):
            print("Loading existing surrogate model...")
            return torch.load(surrogate_path)

        print("Training new surrogate model...")
        if surrogate is None:
            surrogate = nn.Sequential(
                MaskLayer1d(value=0, append=True),
                nn.Linear(2 * self.num_features, 128),
                nn.ELU(inplace=True),
                nn.Linear(128, 128),
                nn.ELU(inplace=True),
                nn.Linear(128, 2)).to(self.device)
        
        surr = Surrogate(surrogate, self.num_features)
        surr.train((X_train_std, Y_train_surrogate),
                    (X_val_std, Y_val_surrogate),
                    batch_size=32,
                    max_epochs=100,
                    loss_fn=SoftCrossEntropyLoss(),
                    validation_samples=10,
                    validation_batch_size=10000,
                    verbose=False)
        
        print('Best loss = {:.4f}'.format(min(surr.loss_list)))
        surrogate.cpu()
        surrogate.eval()
        
        torch.save(surrogate, surrogate_path)
        
        return surrogate

    def train_fastshap(self, X_train_std, X_val_std):
        explainer_path = os.path.join(self.model_dir, f"{self.dataset_name}_explainer.pt")
        if os.path.exists(explainer_path):
            print("Loading existing FastSHAP explainer...")
            return torch.load(explainer_path)

        print("Training new FastSHAP explainer...")
        surrogate_path = os.path.join(self.model_dir, f"{self.dataset_name}_surrogate.pt")
        surrogate = torch.load(surrogate_path).eval().to(self.device)
        surr = Surrogate(surrogate, self.num_features)
        
        explainer = nn.Sequential(
            nn.Linear(self.num_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * self.num_features)).to(self.device)
        
        fastshap = FastSHAP(explainer, surr, normalization=None, link=nn.Softmax(dim=-1))
        
        fastshap.train(
            X_train_std,
            X_val_std[:100],
            batch_size=32,
            num_samples=32,
            max_epochs=100,
            eff_lambda=0,
            paired_sampling=True,
            validation_samples=128,
            validation_seed=0,
            verbose=False)
        
        print('Best val loss = {:.8f}'.format(min(fastshap.loss_list)))
        
        explainer.cpu()
        torch.save(explainer, explainer_path)
        
        return explainer

    def compute_shapley_values(self, X_test_std):
        
        print("Computing Shapley values...")
        explainer_path = os.path.join(self.model_dir, f"{self.dataset_name}_explainer.pt")
        surrogate_path = os.path.join(self.model_dir, f"{self.dataset_name}_surrogate.pt")
        
        explainer = torch.load(explainer_path).eval().to(self.device)
        
        surrogate = torch.load(surrogate_path).eval().to(self.device)
        
        surr = Surrogate(surrogate, self.num_features)
        
        # Set up FastSHAP object
        fastshap = FastSHAP(explainer, surr, normalization='additive', link=nn.Softmax(dim=-1))
        
        # Generate explanations
        pred = fastshap.shap_values(X_test_std)
        
        
        return pred

def run(dataset_name):
    experiment = FastSHAPExperiment(dataset_name, random_state=42)
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_train_std, X_val_std, X_test_std, Y_train_oh, Y_val_oh, Y_test_oh = experiment.load_and_preprocess_data()
    
    model = experiment.train_lightgbm_model(X_train, Y_train, X_val, Y_val)
    
    Y_train_surrogate, Y_val_surrogate = experiment.prepare_surrogate_data(model, X_train, X_val)
    
    surrogate = experiment.train_surrogate_model(X_train_std, Y_train_surrogate, X_val_std, Y_val_surrogate)
    
    explainer = experiment.train_fastshap(X_train_std, X_val_std)
    
    shapley_values = experiment.compute_shapley_values(X_train_std)
    
    return X_train.columns, shapley_values



if __name__ == "__main__":
    col_names, shapley_values = run(sys.argv[1] if len(sys.argv) > 1 else 'diabetes')
    print(shapley_values.shape)
    idx = 5
    # sv_idx is the shapley values without aggregation (idx)
    sv_idx = shapley_values[idx,:,1]
    print("shapley_values for {}-th sample is \n{}\n".format(idx, sv_idx))
    sv_sum = np.sum(shapley_values[:,:,1], axis=0)
    # sv_sum is the shapley values with aggregation
    # uncomment the following two lines for MinMaxScaling
    # scaler = MinMaxScaler(feature_range=(0,1))
    # sv_sum = scaler.fit_transform(sv_sum.reshape((-1,1))).reshape((-1,))
    print(sv_sum)
