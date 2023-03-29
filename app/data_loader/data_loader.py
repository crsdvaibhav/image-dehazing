from torchvision import transforms
from torch.utils.data import DataLoader
from . import dataset
class CustomDataLoader(DataLoader):
    """
    Custom data loader for image deblurring
    """

    def __init__(self, data_dir):
        transform = transforms.Compose([
            transforms.ToTensor(),  # convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize
        ])
        self.dataset = dataset.CustomDataset(data_dir, transform=transform)

        super(CustomDataLoader, self).__init__(self.dataset)
