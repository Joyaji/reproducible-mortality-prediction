"""
Data Loader for MIMIC-III In-Hospital Mortality Dataset
Loads and preprocesses data for the LSTM + Attention model
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
import pickle

class MIMICDataLoader:
    """
    DataLoader for MIMIC-III In-Hospital Mortality Dataset
    Loads and preprocesses data for the LSTM + Attention model
    """
    
    def __init__(self, data_dir: str = "data/in-hospital-mortality", 
                 root_dir: str = "data/root",
                 max_timesteps: int = 48,
                 max_features: int = 100):
        """
        Args:
            data_dir: Directory with listfiles
            root_dir: Root directory with patient data
            max_timesteps: Maximum number of timesteps (hours)
            max_features: Maximum number of features per timestep
        """
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.max_timesteps = max_timesteps
        self.max_features = max_features
        
        # Load vocabulary
        vocab_path = os.path.join(data_dir, "vocab.json")
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)
        
        self.itemid_to_idx = {int(itemid): idx for idx, itemid in enumerate(self.vocab.keys())}
        self.n_features = len(self.vocab)
        
        # Scaler for normalization
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        print(f"DataLoader initialized:")
        print(f"  • Vocabulary: {self.n_features} features")
        print(f"  • Max timesteps: {max_timesteps}h")
        print(f"  • Max features/timestep: {max_features}")
    
    def load_listfile(self, split: str = "train") -> pd.DataFrame:
        """
        Load listfile (train or test)
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            DataFrame with columns ['stay', 'y_true']
        """
        listfile_path = os.path.join(self.data_dir, split, "listfile.csv")
        
        if not os.path.exists(listfile_path):
            raise FileNotFoundError(f"Listfile not found: {listfile_path}")
        
        df = pd.read_csv(listfile_path)
        print(f"\n{split.upper()} set:")
        print(f"  • Episodes: {len(df)}")
        print(f"  • Mortality: {df['y_true'].mean():.1%} ({df['y_true'].sum()}/{len(df)})")
        
        return df
    
    def load_timeseries(self, stay_path: str) -> pd.DataFrame:
        """
        Load timeseries file for an episode
        
        Args:
            stay_path: Relative path to the episode (ex: '10000/episode1_timeseries.csv')
            
        Returns:
            DataFrame with columns ['Hours', 'Itemid', 'Value']
        """
        full_path = os.path.join(self.root_dir, stay_path)
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Timeseries not found: {full_path}")
        
        df = pd.read_csv(full_path)
        
        # Filter only first 48h
        df = df[df['Hours'] <= self.max_timesteps]
        
        return df
    
    def timeseries_to_tensor(self, timeseries_df: pd.DataFrame) -> np.ndarray:
        """
        Convert timeseries to 3D tensor: (timesteps, features)
        
        Strategy: Divide into 1-hour bins and aggregate values
        
        Args:
            timeseries_df: DataFrame with ['Hours', 'Itemid', 'Value']
            
        Returns:
            Tensor of shape (max_timesteps, n_features)
        """
        # Create empty tensor
        tensor = np.zeros((self.max_timesteps, self.n_features))
        
        # Discretize time into 1-hour bins
        timeseries_df['Hour_Bin'] = timeseries_df['Hours'].astype(int)
        timeseries_df['Hour_Bin'] = timeseries_df['Hour_Bin'].clip(0, self.max_timesteps - 1)
        
        # Group by hour and itemid, taking the mean of values
        for hour_bin in range(self.max_timesteps):
            hour_data = timeseries_df[timeseries_df['Hour_Bin'] == hour_bin]
            
            for _, row in hour_data.iterrows():
                itemid = int(row['Itemid'])
                value = float(row['Value'])
                
                if itemid in self.itemid_to_idx:
                    feature_idx = self.itemid_to_idx[itemid]
                    
                    # If already has value, take average
                    if tensor[hour_bin, feature_idx] != 0:
                        tensor[hour_bin, feature_idx] = (tensor[hour_bin, feature_idx] + value) / 2
                    else:
                        tensor[hour_bin, feature_idx] = value
        
        return tensor
    
    def load_data(self, split: str = "train", 
                  normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load complete dataset
        
        Args:
            split: 'train' or 'test'
            normalize: If True, normalizes features
            
        Returns:
            X: Array of shape (n_samples, max_timesteps, n_features)
            y: Array of shape (n_samples,)
        """
        print(f"\n{'='*60}")
        print(f"Loading {split.upper()} dataset...")
        print(f"{'='*60}")
        
        # Load listfile
        listfile = self.load_listfile(split)
        
        X_list = []
        y_list = []
        
        # Load each episode
        for idx, row in listfile.iterrows():
            try:
                # Load timeseries
                timeseries = self.load_timeseries(row['stay'])
                
                # Convert to tensor
                tensor = self.timeseries_to_tensor(timeseries)
                
                X_list.append(tensor)
                y_list.append(row['y_true'])
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed: {idx + 1}/{len(listfile)}", end='\r')
            
            except Exception as e:
                print(f"\n  Error processing {row['stay']}: {e}")
                continue
        
        print(f"\n  Processed: {len(X_list)}/{len(listfile)}")
        
        # Convert to arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Normalize
        if normalize:
            if split == "train" and not self.scaler_fitted:
                # Fit scaler on train set
                X_reshaped = X.reshape(-1, self.n_features)
                self.scaler.fit(X_reshaped)
                self.scaler_fitted = True
                print("  Scaler fitted on train set")
            
            if self.scaler_fitted:
                # Transform
                original_shape = X.shape
                X_reshaped = X.reshape(-1, self.n_features)
                X_normalized = self.scaler.transform(X_reshaped)
                X = X_normalized.reshape(original_shape)
                print("  ✓ Features normalized")
        
        print(f"\nFinal shape:")
        print(f"  • X: {X.shape}")
        print(f"  • y: {y.shape}")
        print(f"  • Mortality: {y.mean():.1%}")
        
        return X, y
    
    def save_scaler(self, path: str = "models/scaler.pkl"):
        """Save scaler for future use"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to: {path}")
    
    def load_scaler(self, path: str = "models/scaler.pkl"):
        """Load saved scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.scaler_fitted = True
        print(f"Scaler loaded from: {path}")
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names"""
        return [self.vocab[str(itemid)] for itemid in sorted([int(k) for k in self.vocab.keys()])]
    
    def get_statistics(self, split: str = "train") -> Dict:
        """
        Return dataset statistics
        
        Returns:
            Dict with statistics
        """
        listfile = self.load_listfile(split)
        
        stats = {
            'n_samples': len(listfile),
            'n_positive': int(listfile['y_true'].sum()),
            'n_negative': int((1 - listfile['y_true']).sum()),
            'mortality_rate': float(listfile['y_true'].mean()),
            'n_features': self.n_features,
            'max_timesteps': self.max_timesteps
        }
        
        return stats


def test_data_loader():
    """Test function for data loader"""
    print("="*60)
    print("TEST DATA LOADER")
    print("="*60)
    
    # Initialize loader
    loader = MIMICDataLoader()
    
    # Load train
    X_train, y_train = loader.load_data(split="train", normalize=True)
    
    # Load test
    X_test, y_test = loader.load_data(split="test", normalize=True)
    
    # Statistics
    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    
    train_stats = loader.get_statistics("train")
    test_stats = loader.get_statistics("test")
    
    print("\nTrain:")
    for key, value in train_stats.items():
        print(f"  • {key}: {value}")
    
    print("\nTest:")
    for key, value in test_stats.items():
        print(f"  • {key}: {value}")
    
    # Save scaler
    loader.save_scaler()
    
    print("\n✅ Test completed successfully!")
    
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # Run test
    X_train, y_train, X_test, y_test = test_data_loader()
