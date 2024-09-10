from datasets.data_processing import CustomImageDataset, DataLoader
from utils import get_backbone, get_detector
import torch
from tqdm.auto import tqdm
from PIL import Image, ImageDraw, ImageFont

def convert_box_format(box):
    """
    Convert [x_center, y_center, width, height] to [x_min, y_min, x_max, y_max]
    """
    x_center, y_center, width, height = box
    x_min = x_center - width / 2
    y_min = y_center - height / 2
    x_max = x_center + width / 2
    y_max = y_center + height / 2
    return [x_min, y_min, x_max, y_max]

def convert_coco_to_xyxy(box):
    """
    Convert COCO format [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
    """
    x_min, y_min, width, height = box
    x_max = x_min + width
    y_max = y_min + height
    return [x_min, y_min, x_max, y_max]
def collate_fn(batch):
    return tuple(zip(*batch))

def annotate_image(image, ground_truth, predictions, model):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # Draw ground truth boxes in green
    for label, box in zip(ground_truth['labels'], ground_truth['boxes']):
        box = convert_coco_to_xyxy(box)
        draw.rectangle(box, outline=(0, 255, 0), width=2)
        draw.text((box[0], box[1]), f"GT: {label}", fill=(0, 255, 0), font=font)

    # Draw prediction boxes in red
    for score, label_id, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
        box = [round(i, 2) for i in box.tolist()]
        draw.rectangle(box, outline=(255, 0, 0), width=2)
        label = model.model.config.id2label[label_id.item()]
        draw.text((box[0], box[3]), f"Pred: {label} ({score:.2f})", fill=(255, 0, 0), font=font)

    return image

data_type = "train"
img_dir = f'datasets/SARDet100K/images/{data_type}'
json_file = f'datasets/SARDet100K/Annotations/{data_type}.json'
dataset = CustomImageDataset(json_file, img_dir, transform=None)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {device} to Detect")

# Model
backbone = get_backbone(backbone='no_backbone')
model = get_detector(detector='DeformableDetr', device=device)
print("Detection Started   ⚡⚡⚡⚡⚡⚡⚡ 🔥🔥🔥🔥🔥🔥🔥")

with torch.no_grad():
    progress_bar = tqdm(total=len(dataloader))
    progress_bar.set_description(f"Detection")
    for step, batch in enumerate(dataloader):
            
            images, targets, info = batch
            info = info[0]
            # images = torch.stack(images).to(device)
            # labels = torch.cat([t.float().mean().unsqueeze(0) for t in targets]).unsqueeze(1).to(device)
            # images = images[0].to(device)
            #####################
            features = backbone.extract_features(images)
            predictions = model.get_predictions(features)

            #####################
            
            print(f"Ground Truth for Image {step+1} : {info['labels']} and {info['boxes']}")
            print(f"predictions for Image {step+1}")
            for result in predictions:
                for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                    score, label = score.item(), label_id.item()
                    box = [round(i, 2) for i in box.tolist()]
                    print(f"{model.model.config.id2label[label]}: {score:.2f} {box}")
            annotated_image = annotate_image(images[0].copy(), info, result, model)
        
            # Save the annotated image
            annotated_image.save(f"DeformableDetr_predictions/annotated_image_{step+1}.png")
            if step==200:
                exit()

    progress_bar.close()

    print("Detection Completed   ✅✅✅ 💯💯💯")