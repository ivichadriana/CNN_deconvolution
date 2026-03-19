'''Utilities for data loading, shuffling, and model definitions
for MLP, CNN, and Transformer experiments. '''
    
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as transforms
import h5py

def load_training_data_fullshuffle(dataset, batch_size, pcam_data_path=None, val_split=0.2, shuffle_order=None):

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
    ])

    if 'MNIST' in dataset and 'Fash' not in dataset:
        full_train_dataset = torchvision.datasets.MNIST(root="~/datasets/mnist", train=True, download=True, transform=transform)
        image_dim = 28
    elif 'MNIST' in dataset and 'Fash' in dataset:
        full_train_dataset = torchvision.datasets.FashionMNIST(root='~/datasets/fashion_mnist', train=True, download=True, transform=transform)
        image_dim = 28
    elif 'CIFAR10' in dataset:
        full_train_dataset = torchvision.datasets.CIFAR10(root="~/datasets/cifar10", train=True, download=True, transform=transform)
        image_dim = 32
    elif 'PCam' in dataset:
        # Load PCam train dataset
        data_path = "~/datasets/pcam/"
        full_train_dataset = PCamDataset(
            data_path=f"{data_path}camelyonpatch_level_2_split_train_x.h5",
            targets_path=f"{data_path}camelyonpatch_level_2_split_train_y.h5"
        )
        image_dim = 96  # PCam images are 96x96x3
    else:
        raise ValueError('Data must be MNIST, FashMNIST, PCam or CIFAR10')

    shuffle = True if 'shuffle' in dataset else False

    if shuffle:
        # Generate shuffle order if not provided
        if shuffle_order is None:
            shuffle_order = generate_shuffle_order(image_dim * image_dim)
        
        if 'MNIST' in dataset or 'Fash' in dataset:
            full_train_dataset.data = torch.tensor(
                np.array([shuffle_image(img.numpy(), shuffle_order) for img in full_train_dataset.data]), dtype=torch.uint8
            )
        elif 'CIFAR10' in dataset or 'PCam' in dataset:
            full_train_dataset.data = np.array([
                shuffle_image_3CH(img, shuffle_order) for img in full_train_dataset.data
            ])

    # Split training data into train and validation sets
    if isinstance(full_train_dataset, PCamDataset):
        # For PCam, validation is pre-defined
        val_dataset = PCamDataset(
            data_path=f"{data_path}camelyonpatch_level_2_split_valid_x.h5",
            targets_path=f"{data_path}camelyonpatch_level_2_split_valid_y.h5"
        )
    else:
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        full_train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    shuffle_order = shuffle_order if shuffle else None

    return train_loader, val_loader, shuffle_order

def load_testing_data_fullshuffle(dataset, batch_size, pcam_data_path, shuffle_order=None):

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
    ])

    # Load the full MNIST dataset
    if 'MNIST' in dataset and 'Fash' not in dataset:
        full_test_dataset = torchvision.datasets.MNIST(root="~/datasets/mnist", train=False, download=True, transform=transform)
        image_dim = 28
    elif 'MNIST' in dataset and 'Fash' in dataset:
        # Load FashionMNIST dataset
        full_test_dataset = torchvision.datasets.FashionMNIST(root='~/datasets/fashion_mnist', train=False, download=True, transform=transform)
        image_dim = 28
    elif 'CIFAR10' in dataset:
        full_test_dataset = torchvision.datasets.CIFAR10(root="~/datasets/cifar10", train=False, download=True, transform=transform)
        image_dim = 32
    
    shuffle = True if 'shuffle' in dataset else False

    if shuffle:
        if shuffle_order is None:
            shuffle_order = generate_shuffle_order(image_dim * image_dim)

        if 'MNIST' in dataset:
            full_test_dataset.data = torch.tensor(
                np.array([shuffle_image(img.numpy(), shuffle_order) for img in full_test_dataset.data]), dtype=torch.uint8
            )
        elif 'CIFAR10' in dataset or 'PCam' in dataset:
            full_test_dataset.data = np.array([
                shuffle_image_3CH(img, shuffle_order) for img in full_test_dataset.data
            ])

    # Create DataLoader
    full_test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    shuffle_order = shuffle_order if shuffle else None
    
    return full_test_loader, shuffle_order

def get_dimensions(dataset):
    ''' Get image dimensions from dataset name'''
    if "MNIST" in dataset:
        image_dim = 28
    elif "PCam" in dataset:
        image_dim = 96
    elif "CIFAR10" in dataset:
        image_dim = 32
    else:
        raise ValueError('Data must be MNIST, FashMNIST, PCam or CIFAR10')
    return image_dim

# Shuffle the rows and columns of an image
def shuffle_image_rows_columns(image, shuffle_order_rows, shuffle_order_columns):
    image = image[shuffle_order_rows, :]  # Shuffle rows
    image = image[:, shuffle_order_columns]  # Shuffle columns
    return image

# Shuffle the rows and columns of an image
def shuffle_image_rows_columns_3CH(image, shuffle_order_rows, shuffle_order_columns):
    image = image[shuffle_order_rows, :, :]  # Shuffle rows
    image = image[:, shuffle_order_columns, :]  # Shuffle columns
    return image

# Fully shuffle all pixels in an image (grayscale)
def shuffle_image(image, shuffle_order):
    flat_image = image.flatten()  # Flatten image into 1D
    shuffled_image = flat_image[shuffle_order]  # Apply the shuffle order
    return shuffled_image.reshape(image.shape)  # Reshape back to original dimensions

# Fully shuffle all pixels in an image (3-channel)
def shuffle_image_3CH(image, shuffle_order):
    flat_image = image.reshape(-1, image.shape[-1])  # Flatten while keeping color channels
    shuffled_image = flat_image[shuffle_order]  # Apply the shuffle order
    return shuffled_image.reshape(image.shape)  # Reshape back to original dimensions

# Generate shuffle orders
def generate_shuffle_orders(size):
    shuffle_order_rows = np.random.permutation(size)  # Shuffle rows
    shuffle_order_columns = np.random.permutation(size)  # Shuffle columns
    return shuffle_order_rows, shuffle_order_columns

# Generate a fully randomized shuffle order
def generate_shuffle_order(size):
    return np.random.permutation(size)  # Fully random shuffle order

class PCamDataset(Dataset):
    """Custom Dataset for PCam data."""
    def __init__(self, data_path, targets_path):
        self.data = self._load_h5(data_path)
        self.targets = self._load_h5(targets_path)
        print(f"Data shape: {self.data.shape}, targets shape: {self.targets.shape}")

    def _load_h5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            data = np.array(f['x'] if 'x' in f else f['y'])
        return data.squeeze()  # Remove all singleton dimensions

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = self.data[idx]  # (96, 96, 3)
        label = self.targets[idx]  # 0 or 1
        img = img.astype(np.float32) / 255.0  # Scale to [0, 1]
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

def load_training_data(dataset, batch_size, pcam_data_path=None, val_split=0.2, shuffle_order_rows=None):

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
    ])

    if 'MNIST' in dataset and 'Fash' not in dataset:
        full_train_dataset = torchvision.datasets.MNIST(root="~/datasets/mnist", train=True, download=True, transform=transform)
        image_dim = 28
    elif 'MNIST' in dataset and 'Fash' in dataset:
        full_train_dataset = torchvision.datasets.FashionMNIST(root='~/datasets/fashion_mnist', train=True, download=True, transform=transform)
        image_dim = 28
    elif 'CIFAR10' in dataset:
        full_train_dataset = torchvision.datasets.CIFAR10(root="~/datasets/cifar10", train=True, download=True, transform=transform)
        image_dim = 32
    elif 'PCam' in dataset:
        # Load PCam train dataset
        data_path = "~/datasets/pcam/"
        full_train_dataset = PCamDataset(
            data_path=f"{data_path}camelyonpatch_level_2_split_train_x.h5",
            targets_path=f"{data_path}camelyonpatch_level_2_split_train_y.h5"
        )
        image_dim = 96  # PCam images are 96x96x3
    else:
        raise ValueError('Data must be MNIST, FashMNIST, PCam or CIFAR10')

    shuffle = True if 'shuffle' in dataset else False

    if shuffle:
        # Generate shuffle orders
        if shuffle_order_rows is None:
            shuffle_order_rows, shuffle_order_columns = generate_shuffle_orders(image_dim)
        if 'MNIST' in dataset or 'Fash' in dataset:
            full_train_dataset.data = torch.tensor(
                np.array([
                    shuffle_image_rows_columns(img.numpy(), shuffle_order_rows, shuffle_order_columns)
                    for img in full_train_dataset.data]), dtype=torch.uint8)
        elif 'CIFAR10' in dataset or 'PCam' in dataset:
            full_train_dataset.data = np.array([
                shuffle_image_rows_columns_3CH(img, shuffle_order_rows, shuffle_order_columns)
                for img in full_train_dataset.data])

    # Split training data into train and validation sets
    if isinstance(full_train_dataset, PCamDataset):
        # For PCam, validation is pre-defined
        val_dataset = PCamDataset(
            data_path=f"{data_path}camelyonpatch_level_2_split_valid_x.h5",
            targets_path=f"{data_path}camelyonpatch_level_2_split_valid_y.h5"
        )
    else:
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        full_train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    shuffle_order_rows = shuffle_order_rows if shuffle else 0
    shuffle_order_columns = shuffle_order_columns if shuffle else 0

    return train_loader, val_loader, shuffle_order_rows, shuffle_order_columns

def load_testing_data(dataset, batch_size, pcam_data_path, shuffle_order_rows=None, shuffle_order_columns=None):

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
    ])

    # Load the full MNIST dataset
    if 'MNIST' in dataset and 'Fash' not in dataset:
        full_test_dataset = torchvision.datasets.MNIST(root="~/datasets/mnist", train=False, download=True, transform=transform)
        image_dim = 28
    elif 'MNIST' in dataset and 'Fash' in dataset:
        # Load FashionMNIST dataset
        full_test_dataset = torchvision.datasets.FashionMNIST(root='~/datasets/fashion_mnist', train=False, download=True, transform=transform)
        image_dim = 28
    elif 'CIFAR10' in dataset:
        full_test_dataset = torchvision.datasets.CIFAR10(root="~/datasets/cifar10", train=False, download=True, transform=transform)
        image_dim = 32
    
    shuffle = True if 'shuffle' in dataset else False

    if shuffle:
        if shuffle_order_columns is None:
            # Generate shuffle orders
            shuffle_order_rows, shuffle_order_columns = generate_shuffle_orders(image_dim)
        # Apply shuffling to the original dataset's data (convert to NumPy, shuffle, convert back to tensor)
        if 'MNIST' in dataset:
            full_test_dataset.data = torch.tensor(
                np.array([
                    shuffle_image_rows_columns(img.numpy(), shuffle_order_rows, shuffle_order_columns)
                    for img in full_test_dataset.data]), dtype=torch.uint8) # Maintain original dtype
        elif 'CIFAR10' in dataset or 'PCam' in dataset:
            full_test_dataset.data = np.array([
                shuffle_image_rows_columns_3CH(img, shuffle_order_rows, shuffle_order_columns)
                for img in full_test_dataset.data])

    # Create DataLoader
    full_test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    shuffle_order_rows = shuffle_order_rows if shuffle else 0
    shuffle_order_columns = shuffle_order_columns if shuffle else 0
    
    return full_test_loader, shuffle_order_rows, shuffle_order_columns

# Define the CNN model for CIFAR10 (3-channel)
class SimpleCNN_3CH(nn.Module):
    def __init__(self, cha_input, cha_hidden, fc_hidden, kernel_size=3, stride=1, padding=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, cha_input, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(cha_input)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(cha_input, cha_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(cha_hidden)
        self.conv3 = nn.Conv2d(cha_hidden, cha_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(cha_hidden)
        self.fc1 = nn.Linear(cha_hidden * 4 * 4, fc_hidden)
        self.bn4 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define the CNN model for MNIST (1-channel)
class SimpleCNN(nn.Module):
    def __init__(self, cha_input, cha_hidden, fc_hidden, kernel_size=3, stride=1, padding=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cha_input, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(cha_input)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(cha_input, cha_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(cha_hidden)
        self.conv3 = nn.Conv2d(cha_hidden, cha_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(cha_hidden)
        self.fc1 = nn.Linear(cha_hidden * 3 * 3, fc_hidden)
        self.bn4 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)  # 10 classes 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))  # Flatten
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Define the MLP model with dropout
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, fc1_hidden, fc2_hidden, fc3_hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, fc1_hidden)
        self.bn1 = nn.BatchNorm1d(fc1_hidden)
        self.fc2 = nn.Linear(fc1_hidden, fc2_hidden)
        self.bn2 = nn.BatchNorm1d(fc2_hidden)
        self.fc3 = nn.Linear(fc2_hidden, fc3_hidden)
        self.bn3 = nn.BatchNorm1d(fc3_hidden)
        self.fc4 = nn.Linear(fc3_hidden, 10)  # 10 classes
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)  # Output layer
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, emb_dim, num_heads, mlp_dim, num_classes=10, dropout=0.1):
        """
        Args:
            image_size (int): Height (and width) of the input image (assumed square).
            patch_size (int): Size of each square patch.
            in_channels (int): Number of channels in the input image (1 for grayscale, 3 for RGB).
            emb_dim (int): Embedding dimension for each patch.
            num_heads (int): Number of attention heads in each transformer layer.
            mlp_dim (int): Dimension of the feedforward (MLP) layer inside each transformer block.
            num_classes (int): Number of output classes.
            dropout (float): Dropout rate.
        """
        super(SimpleTransformer, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        # Compute the number of patches (assumes image dimensions are divisible by patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding: use a conv layer with kernel and stride equal to patch size
        self.patch_embed = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
        # Learnable class token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        # Positional embeddings for patches + class token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Fix the number of transformer layers to 3 (similar in depth to your CNN and MLP)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        num_layers_fixed = 3  # Fixed number of layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_fixed)
        
        # Classification head: layer normalization followed by a linear layer
        self.norm = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embeddings and class token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Initialize patch embedding weights
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        
    def forward(self, x):
        # x: (batch_size, in_channels, image_size, image_size)
        batch_size = x.shape[0]
        # Patch embedding: (B, emb_dim, num_patches_h, num_patches_w)
        x = self.patch_embed(x)
        # Flatten spatial dimensions: (B, emb_dim, num_patches)
        x = x.flatten(2)
        # Transpose to (B, num_patches, emb_dim)
        x = x.transpose(1, 2)
        
        # Prepend the class token for each sample in the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, emb_dim)
        
        # Add positional embeddings and apply dropout
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer encoder expects input shape: (sequence_length, batch_size, emb_dim)
        x = self.transformer_encoder(x)
        x = x[:, 0]
        
        # Use the output corresponding to the class token for classification
        # x = x[0]  # (B, emb_dim)
        x = self.norm(x)
        x = self.fc(x)
        return x
