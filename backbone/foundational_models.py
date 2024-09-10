from transformers import AutoImageProcessor, AutoModel

class Identity_backbone():
    def __init__(self,device=None):
        pass

    def extract_features(self,image):

        # Do Nothing, as there is no requirement of backbone

        return image

class Dinov2_base():
    def __init__(self,device=None):
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')

    def extract_features(self,image):

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states


