import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from fastshap_torch.utils import MaskLayer1d
from fastshap_torch import Surrogate, SoftCrossEntropyLoss, FastSHAP

class FastSHAPExperimentNN:
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
        
        self.num_features = X_train.shape[1]
        
        return X_train, X_val, X_test, Y_train, Y_val, Y_test, X_train_std, X_val_std, X_test_std

    def train_nn_model(self, X_train, Y_train, X_val, Y_val,model=None):
        model_path = os.path.join(self.model_dir, f"{self.dataset_name}_nn_model.pt")
        if os.path.exists(model_path):
            print("Loading existing NN model...")
            return torch.load(model_path)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        print("Training new NN model...")
        if model == None:
            model = nn.Sequential(
                nn.Linear(self.num_features, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1)
            ).to(self.device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        Y_train_tensor = torch.FloatTensor(Y_train.values).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        Y_val_tensor = torch.FloatTensor(Y_val.values).to(self.device)

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(100):  # max 100 epochs
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, Y_train_tensor.unsqueeze(1))
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, Y_val_tensor.unsqueeze(1))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model, model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return torch.load(model_path)

    def prepare_surrogate_data(self, model, X_train, X_val):
        model.eval()
        with torch.no_grad():
            Y_train_surrogate = torch.sigmoid(model(torch.FloatTensor(X_train).to(self.device))).cpu().numpy()
            Y_val_surrogate = torch.sigmoid(model(torch.FloatTensor(X_val).to(self.device))).cpu().numpy()

        Y_train_surrogate = np.hstack([1 - Y_train_surrogate, Y_train_surrogate])
        Y_val_surrogate = np.hstack([1 - Y_val_surrogate, Y_val_surrogate])
        
        p = Y_train_surrogate.mean(axis=0)
        soft_ce = - np.mean(np.sum(np.log(p) * Y_train_surrogate, axis=1))
        print('Loss given no information = {:.4f}'.format(soft_ce))
        
        return Y_train_surrogate, Y_val_surrogate

    def train_surrogate_model(self, X_train_std, Y_train_surrogate, X_val_std, Y_val_surrogate):
        surrogate_path = os.path.join(self.model_dir, f"{self.dataset_name}_nn_surrogate.pt")
        if os.path.exists(surrogate_path):
            print("Loading existing surrogate model...")
            return torch.load(surrogate_path)

        print("Training new surrogate model...")
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
        explainer_path = os.path.join(self.model_dir, f"{self.dataset_name}_nn_explainer.pt")
        if os.path.exists(explainer_path):
            print("Loading existing FastSHAP explainer...")
            return torch.load(explainer_path)

        print("Training new FastSHAP explainer...")
        surrogate_path = os.path.join(self.model_dir, f"{self.dataset_name}_nn_surrogate.pt")
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
        explainer_path = os.path.join(self.model_dir, f"{self.dataset_name}_nn_explainer.pt")
        surrogate_path = os.path.join(self.model_dir, f"{self.dataset_name}_nn_surrogate.pt")
        
        explainer = torch.load(explainer_path).eval().to(self.device)
        surrogate = torch.load(surrogate_path).eval().to(self.device)
        
        surr = Surrogate(surrogate, self.num_features)
        
        fastshap = FastSHAP(explainer, surr, normalization='additive', link=nn.Softmax(dim=-1))
        
        pred = fastshap.shap_values(X_test_std)
        
        return pred

def run(dataset_name):
    experiment = FastSHAPExperimentNN(dataset_name, random_state=42)
    
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_train_std, X_val_std, X_test_std = experiment.load_and_preprocess_data()
    
    model = experiment.train_nn_model(X_train_std, Y_train, X_val_std, Y_val)
    
    Y_train_surrogate, Y_val_surrogate = experiment.prepare_surrogate_data(model, X_train_std, X_val_std)
    
    surrogate = experiment.train_surrogate_model(X_train_std, Y_train_surrogate, X_val_std, Y_val_surrogate)
    
    explainer = experiment.train_fastshap(X_train_std, X_val_std)
    
    shapley_values = experiment.compute_shapley_values(X_train_std)
    
    return X_train.columns, shapley_values

if __name__ == "__main__":
    col_names, shapley_values = run(sys.argv[1] if len(sys.argv) > 1 else 'diabetes')
    print(shapley_values.shape)
    idx = 5
    sv_idx = shapley_values[idx,:,1]
    print("shapley_values for {}-th sample is \n{}\n".format(idx, sv_idx))
    sv_sum = np.sum(shapley_values[:,:,1], axis=0)
    print(sv_sum)
