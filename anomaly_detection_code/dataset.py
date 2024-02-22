from torchvision import transforms, datasets
import os
from preprocess import get_anomaly_type
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class GridDataset(Dataset):
    def __init__(self, base_dir, transform=None, is_train=True):
        """
        Initializes the dataset.

        Args:
            base_dir (str): The base directory containing the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_train (bool): Flag indicating whether this is training data.
        """
        self.base_dir = base_dir
        self.transform = None
        self.samples = []
        self.is_train = is_train
        self.anomaly_types = {'good': 0}  # Initialize with 'good' mapped to 0

        # Load all images
        for file in os.listdir(base_dir):
            if file.lower().endswith('.png'):
                if self.is_train:
                    # For training data, label everything as 'good'
                    label = 0
                else:
                    # For test data, use filename to determine label
                    anomaly_type = get_anomaly_type(file)
                    if anomaly_type not in self.anomaly_types:
                        # Assign a new label to a previously unseen anomaly type
                        self.anomaly_types[anomaly_type] = len(self.anomaly_types)
                    label = self.anomaly_types[anomaly_type]
                self.samples.append((os.path.join(base_dir, file), label))
                
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches the item (image and label) at the given index.
        """
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Directly convert PIL image to PyTorch tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        return image, label
