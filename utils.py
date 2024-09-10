from backbone.foundational_models import *
from detector.DETR_s import *




def get_detector(detector = None, device=None):

    if detector == "RTDetr":

        detector_model = RTDetr(device=device)
    elif detector == "DeformableDetr":
        return DeformableDetr(device=device)
    else:
        print("Kindly Specify the Available Detector 🤦‍♀️🤦‍♀️🤦‍♀️")
        exit()


    return detector_model

def get_backbone(backbone = None):

    if backbone == "no_backbone":
        backbone_model = Identity_backbone()
    elif backbone == "Dinov2_base":
        backbone_model = Dinov2_base()

    else:
        print("Kindly Specify the Available Backbone 🤦‍♀️🤦‍♀️🤦‍♀️")
        exit()

    return backbone_model