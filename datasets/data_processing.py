import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.img_dir = img_dir
        self.transform = transform
        self.images = data['images']
        self.annotations = data['annotations']
        
        # Create a dictionary to map image_id to annotations
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
        
        print(f"Succesfully Loaded {len(self.images)} images from {img_dir} data set")
        # for item in self.images:
        #     print(item['id'], len(self.img_to_anns[item['id']]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        # Open image file
        image = Image.open(img_path).convert("RGB")
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        # Get annotations for this image
        img_id = img_info['id']
        annotations = self.img_to_anns.get(img_id, [])
        
        # Convert annotations to tensor
        boxes = [ann['bbox'] for ann in annotations]
        # boxes = torch.tensor(boxes, dtype=torch.float32)
        
        labels = [ann['category_id'] for ann in annotations]
        # labels = torch.tensor(labels, dtype=torch.int64)
        
        # target = labels
        label_name = []
        for label in labels:
            if label == 0:
                label_name.append("Ship")
            elif label == 1:
                label_name.append("aircraft")
            elif label == 2:
                label_name.append("car")
            elif label == 3:
                label_name.append("tank")
            elif label == 4:
                label_name.append("bridge")
            elif label == 5:
                label_name.append("harbor")
            else :
                label_name.append(f"new object with label = {label}")
        target = {"image_id" : img_id, "annotations":annotations}
        
        return image, target

# Usage example
def get_dataloader(data_type, batch_size = 4, input_img_size = None, num_workers = 0):
    transform = transforms.Compose([
        transforms.Resize((input_img_size, input_img_size)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    img_dir = f'datasets/SARDet100K/images/{data_type}'
    json_file = f'datasets/SARDet100K/Annotations/{data_type}.json'

    dataset = CustomImageDataset(json_file, img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    
    return dataloader

def collate_fn(batch):
    return tuple(zip(*batch))

# Example usage
if __name__ == "__main__":
    # json_file = 'datasets/SARDet100K/Annotations/train.json'
    # img_dir = 'datasets/SARDet100K/images/train'
    dataloader = get_dataloader(data_type='test', batch_size=5)

    for images, targets in dataloader:
        print(f"Batch size: {len(images)}")
        print(f"Image shape: {images[0].shape}")
        print(f"Target keys: {targets[0].keys()}")
        break