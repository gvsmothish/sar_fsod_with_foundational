from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, AutoImageProcessor
from transformers import DeformableDetrConfig, DeformableDetrForObjectDetection
from transformers import DeformableDetrImageProcessor
import torch

class RTDetr():
    def __init__(self,device=None):
        self.image_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        self.model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd")
        self.device = device

    # def get_predictions(self,image):
        
    #     image = image.repeat(1, 3, 1, 1)
    #     batch_images = image
    #     self.model.to(self.device)

    #     # Process the batch of images
    #     # Note: RTDetrImageProcessor expects a list of images or a single image
    #     # So we need to process each image in the batch separately
    #     inputs = self.image_processor(images=[img.permute(1, 2, 0).cpu().numpy() for img in batch_images], return_tensors="pt", do_rescale=False)
        
    #     # Move inputs to the same device as the model
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}

    #     # Perform inference
    #     with torch.no_grad():
    #         outputs = self.model(**inputs)

    #     # Post-process the outputs
    #     target_sizes = torch.tensor([batch_images.shape[2:4] for _ in range(batch_images.shape[0])]).to(self.device)
    #     results = self.image_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)

    #     return results
    #     # inputs = self.image_processor(images=image, return_tensors="pt")
    #     # outputs = self.model(**inputs)
    #     # results = self.image_processor.post_process_object_detection(outputs, target_sizes=image.shape, threshold=0.3)

    #     # return results
    def get_predictions(self,image):
        
        image = image[0].convert("RGB")
        # self.model.to(self.device)

        inputs = self.image_processor(images=image, return_tensors="pt")
    
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.3)

        return results
    






class DeformableDetr():
    def __init__(self,device=None):
        # super(self).__init__()
        # self.num_classes = num_classes
        config = DeformableDetrConfig(num_labels=6)
        self.model = DeformableDetrForObjectDetection(config)
        self.image_processor = DeformableDetrImageProcessor.from_pretrained("SenseTime/deformable-detr")
        # self.model = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
        self.device = device

    
    def get_predictions(self,image):
        
        image = image[0].convert("RGB")
        # self.model.to(self.device)

        inputs = self.image_processor(images=image, return_tensors="pt")
    
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.image_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=0.1)

        return results
    
    def forward(self, pixel_values, labels=None):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return outputs
    
    def preprocess(self, images, targets=None):
        # Preprocess images and targets
        encodings = self.image_processor(images=images, annotations=targets, return_tensors="pt", do_rescale=False)
        
        # if targets is not None:
        #     # Convert targets to the format expected by the model
        #     labels = [{
        #         'boxes': torch.tensor(target['boxes']),
        #         'labels': torch.tensor(target['labels'])
        #     } for target in targets]
        #     encodings['labels'] = labels

        return encodings["pixel_values"], encodings["pixel_mask"] , encodings["labels"]
    def save(self, path):
        self.model.save_pretrained(path)

    @classmethod
    def load(cls, path, num_classes):
        instance = cls(num_classes, pretrained=False)
        instance.model = DeformableDetrForObjectDetection.from_pretrained(path)
        instance.image_processor = DeformableDetrImageProcessor.from_pretrained(path)
        return instance