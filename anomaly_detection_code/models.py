import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from collections import defaultdict
import torch.nn.functional as F


def train_autoencoder(autoencoder,save_path, train_loader, epochs=1, learning_rate=0.001, data_usage_percent=100):
    """
    Trains an autoencoder model.

    Parameters:
    - autoencoder: The autoencoder model to be trained.
    - train_loader: DataLoader for the training data.
    - epochs: Number of epochs to train for.
    - learning_rate: Learning rate for the optimizer.
    - data_usage_percent: Percentage of training data to use for training (default 100%).

    Returns:
    - A tuple containing the trained autoencoder model and the training loss per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder = autoencoder.to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    # Initialize dictionaries to store accuracies
    epoch_category_accuracies = defaultdict(list)  # This should be a defaultdict of lists
    total_accuracy_per_epoch = []

    # Calculate number of batches to use per epoch based on the percentage
    total_batches = len(train_loader)
    batches_to_use = int((data_usage_percent / 100) * total_batches)

    for epoch in range(epochs):
        autoencoder.train()
        running_loss = 0.0
        for batch_idx, (images, _) in enumerate(train_loader):  # Labels not used in training
            if batch_idx >= batches_to_use:
                # Break the loop if reached the specified percentage of batches
                break
            images = images.to(device)
            optimizer.zero_grad()
            outputs = autoencoder(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(batch_idx)
            # Print the progress
            if batch_idx % 100 == 0:
                percentage = (batch_idx / batches_to_use) * 100
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{batches_to_use} ({percentage:.2f}%)")

        # Evaluation phase adjustments
        autoencoder.eval()
        # Save the model state at the end of each epoch
        torch.save(autoencoder.state_dict(), save_path)

        avg_loss = running_loss / batches_to_use
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_loss}")
        total_accuracy_per_epoch.append(avg_loss)

    return autoencoder, total_accuracy_per_epoch




class AutoencoderWithDropout(nn.Module):
    """
    Autoencoder network with dropout for reducing overfitting.

    Includes an encoder and decoder with convolutional and pooling layers, incorporating dropout layers.

    Attributes:
    - dropout_rate (float): Dropout probability.

    Methods:
    - forward(x): Forward pass using input x.
    """

    def __init__(self, dropout_rate=0.5):
        super(AutoencoderWithDropout, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.enc_pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.enc_pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # Decoder
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.dec_conv2 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        # Encoder
        x = F.relu(self.enc_conv1(x))
        x, indices1 = self.enc_pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.enc_conv2(x))
        x, indices2 = self.enc_pool2(x)
        x = self.dropout1(x)
        
        # Decoder
        x = self.dec_conv1(F.max_unpool2d(x, indices2, kernel_size=2, stride=2))
        x = self.dropout2(x)
        x = self.dec_conv2(F.max_unpool2d(x, indices1, kernel_size=2, stride=2))
        x = torch.sigmoid(x)
        
        return x