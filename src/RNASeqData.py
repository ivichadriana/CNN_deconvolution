import torch
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import MeanSquaredError
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import numpy as np

class RNASeqData(LightningDataModule):
    def __init__(self, train_expressions, train_proportions, test_expressions, test_proportions, batch_size, n_splits=5, random_state=42):
        super().__init__()
        self.train_expressions = train_expressions
        self.train_proportions = train_proportions
        self.test_expressions = test_expressions
        self.test_proportions = test_proportions
        self.batch_size = batch_size
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.splits = list(self.kf.split(self.train_expressions))
        self.current_split = 0  # To keep track of the current split

        # Placeholder for train and validation splits
        self.train_expressions_split = None
        self.val_expressions_split = None
        self.train_proportions_split = None
        self.val_proportions_split = None

        # Identify gene and cell columns
        self.gene_cols = list(self.train_expressions.columns.values)
        self.cell_cols = list(self.train_expressions.index.values)

    def prepare_fold_data(self, fold_index):
        """ Prepare the data for a specific fold. """
        self.current_split = fold_index
        train_idx, val_idx = self.splits[fold_index]
        
        self.train_expressions_split = self.train_expressions.iloc[train_idx].copy()
        self.train_proportions_split = self.train_proportions.iloc[train_idx].copy()
        self.val_expressions_split = self.train_expressions.iloc[val_idx].copy()
        self.val_proportions_split = self.train_proportions.iloc[val_idx].copy()

    def sample_specific_normalization(self, df, gene_cols):
        """Apply sample-specific normalization following the Kaggle competition approach."""
        # Gene-specific normalization
        q2_genes = df[gene_cols].quantile(0.25, axis=1)
        q7_genes = df[gene_cols].quantile(0.75, axis=1)
        qmean_genes = (q2_genes + q7_genes) / 2
        df[gene_cols] = (df[gene_cols].T - qmean_genes.values).T

        # Cell-specific normalization
        q2_cells = df.quantile(0.25, axis=1)
        q7_cells = df.quantile(0.72, axis=1)
        qmean_cells = (q2_cells + q7_cells) / 2
        df = (df.T - qmean_cells.values).T

        # Scaling step
        qmean2_cells = df.abs().quantile(0.75, axis=1) + 4
        df = (df.T / qmean2_cells.values).T

        return df

    def normalize_data(self):
        """Apply sample-specific normalization independently for each data split."""
        # Apply normalization independently to each set
        self.train_expressions_split = self.sample_specific_normalization(self.train_expressions_split, self.gene_cols)
        self.val_expressions_split = self.sample_specific_normalization(self.val_expressions_split, self.gene_cols)
        self.test_expressions = self.sample_specific_normalization(self.test_expressions.copy(), self.gene_cols)

    def setup(self, stage=None):
        """ Setup function for different stages of training. """
        self.prepare_fold_data(self.current_split)
        self.normalize_data()
        
        # Convert data to torch tensors
        self.train_expressions_split = torch.tensor(self.train_expressions_split.values, dtype=torch.float32)
        self.train_proportions_split = torch.tensor(self.train_proportions_split.values, dtype=torch.float32)
        self.val_expressions_split = torch.tensor(self.val_expressions_split.values, dtype=torch.float32)
        self.val_proportions_split = torch.tensor(self.val_proportions_split.values, dtype=torch.float32)
        self.test_expressions = torch.tensor(self.test_expressions.values, dtype=torch.float32)
        self.test_proportions = torch.tensor(self.test_proportions.values, dtype=torch.float32)

    def train_dataloader(self):
        train_dataset = TensorDataset(self.train_expressions_split, self.train_proportions_split)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        val_dataset = TensorDataset(self.val_expressions_split, self.val_proportions_split)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.test_expressions, self.test_proportions)
        return DataLoader(test_dataset, batch_size=self.batch_size)

    def next_fold(self):
        """ Move to the next fold in K-Fold cross-validation. """
        if self.current_split < len(self.splits) - 1:
            self.current_split += 1
            self.setup()  # Re-setup with the new fold
        else:
            print("All folds completed.")
