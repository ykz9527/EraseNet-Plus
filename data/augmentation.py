import torch
import torchvision
from torchvision import datasets, transforms
import os
from PIL import Image

# Define data augmentation operations
data_transforms = transforms.Compose([
    transforms.RandomRotation(10),  # Random rotation
    # transforms.RandomHorizontalFlip(0.3),  # Random horizontal flip
])

# Specify the directories for train and test data
dirs = ['/root/autodl-tmp/train2/all_images','/root/autodl-tmp/train2/all_labels', '/root/autodl-tmp/train2/mask']
file_names = os.listdir(dirs[0])  # Get the list of file names

for i, file_name in enumerate(file_names):
    # Set a random seed to ensure the same augmentation for corresponding files
    random_seed = torch.randint(0, 10000, (1,)).item()
    torch.cuda.manual_seed_all(1)
    for dir in dirs:
        path = os.path.join(dir, file_name)
        image = Image.open(path)  # Load the image using PIL.Image
        input = data_transforms(image)  # Apply the transformations
        input_tensor = torchvision.transforms.functional.to_tensor(input)  # Convert the transformed image to tensor

        # Convert tensor to uint8 data type
        input_tensor = input_tensor.mul(255).byte()

        # Save the augmented image without overwriting the original image
        save_path = os.path.join(dir, f'{file_name[:-4]}_a.jpg')
        torchvision.io.write_jpeg(input_tensor, save_path)
