import torch
from utils import get_backbone, get_detector

import torch
import argparse
from datasets.data_processing import get_dataloader
from tqdm.auto import tqdm
from torch.nn import functional as F
from torch import nn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="exp_default")
    parser.add_argument("--dataset", type=str, default="SARDet100k", choices=["SARDet100k"])
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--input_img_size", type=int, default=800)
    parser.add_argument("--detector", type=str, default="RTDetr", choices=["RTDetr"])
    parser.add_argument("--backbone", type=str, default="no_backbone", choices=["no_backbone","Dinov2_base"])
    #####################################################################################
    # parser.add_argument("--obs_embedding_size", type=int, default=48)
    # parser.add_argument("--hidden_size", type=int, default=256)
    # parser.add_argument("--hidden_layers", type=int, default=6)
    # parser.add_argument("--time_embedding", type=str, default="sinusoidal", choices=["sinusoidal", "learnable", "linear", "zero"])
    # parser.add_argument("--obs_embedding", type=str, default="learnable_multi", choices=["sinusoidal", "learnable","learnable_multi", "linear", "identity"])
    # parser.add_argument("--save_images_step", type=int, default=1)
    # parser.add_argument("--obs_size", type=int, default=30*5)
    # parser.add_argument("--action_size", type=int, default=6)
    config = parser.parse_args()

    test_dataloader = get_dataloader(data_type='test', batch_size=config.test_batch_size, input_img_size=config.input_img_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to test")

    # Model
    backbone = get_backbone(config.backbone)
    model = get_detector(config.detector, device=device)

    final_test_loss = 0
    print("Testing Started   ⚡⚡⚡⚡⚡⚡⚡ 🔥🔥🔥🔥🔥🔥🔥")

    num_tests = len(test_dataloader)/config.test_batch_size

    with torch.no_grad():
        test_loss = 0.0
        progress_bar = tqdm(total=len(test_dataloader))
        progress_bar.set_description(f"Test")
        for step, batch in enumerate(test_dataloader):
            
            images, targets, _ = batch

            images = torch.stack(images).to(device)
            labels = torch.cat([t.float().mean().unsqueeze(0) for t in targets]).unsqueeze(1).to(device)

            #####################
            features = backbone.extract_features(images)
            predictions = model.get_predictions(features)

            #####################

            # test_loss = F.mse_loss(predictions, labels)
            for result in predictions:
                for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
                    score, label = score.item(), label_id.item()
                    box = [round(i, 2) for i in box.tolist()]
                    print(f"{model.model.config.id2label[label]}: {score:.2f} {box}")
            final_test_loss += test_loss
            if step>0:
                exit()

    progress_bar.close()

    print("Testing Completed   ✅✅✅ 💯💯💯")
    print(f"Testing Loss is {final_test_loss/num_tests}   ✅✅✅ 💯💯💯")

