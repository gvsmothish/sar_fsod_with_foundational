import torch
from utils import get_backbone, get_detector

import torch
import argparse
from datasets.data_processing import get_dataloader
from tqdm.auto import tqdm
from torch.nn import functional as F
from torch import nn
from detector.DETR_s import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="exp_default")
    parser.add_argument("--dataset", type=str, default="SARDet100k", choices=["SARDet100k"])
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--val_batch_size", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=40000)
    parser.add_argument("--learning_rate", type=float, default=1*1e-4)
    parser.add_argument("--input_img_size", type=int, default=800)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--resume_training", type=bool, default=False)
    parser.add_argument("--resume_training_path", type=str, default="No path")
    parser.add_argument("--detector", type=str, default="DeformableDetr", choices=["RTDetr","DeformableDetr"])
    parser.add_argument("--backbone", type=str, default="no_backbone", choices=["Dinov2_base"])
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

    train_dataloader = get_dataloader(data_type='test', batch_size=config.train_batch_size, input_img_size=config.input_img_size)
    val_dataloader = get_dataloader(data_type='val', batch_size=config.val_batch_size, input_img_size=config.input_img_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device} to train")

    # Model
    backbone = get_backbone(config.backbone)
    model = get_detector(config.detector)
    model.model.to(device)

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.3,
        patience=1,
        mode="min",
    )
    losses = []
    print("Training Started   ⚡⚡⚡⚡⚡⚡⚡ 🔥🔥🔥🔥🔥🔥🔥")
    best_val_loss = float('inf')
    # if config.resume_training:
    #     model.load_state_dict(torch.load(config.resume_training_path))

    for epoch in range(config.num_epochs):
        model.model.train()

        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):

            images, targets = batch

            pixel_values, pixel_mask, labels = model.preprocess(list(images), targets)

            # Move data to the correct device
            pixel_values = pixel_values.to(device)
            pixel_mask = pixel_mask.to(device)
            # labels = labels.to(device)
            # labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
            # labels = torch.cat([t.float().mean().unsqueeze(0) for t in targets]).unsqueeze(1).to(device)

            #############################################################
            
            predictions = model.model(pixel_values = pixel_values,pixel_mask= pixel_mask, labels = labels)
            # train_loss = F.mse_loss(predictions, labels)
            train_loss = predictions.loss
            train_loss.backward()

            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"train_loss": train_loss.detach().item()}
            losses.append(train_loss.detach().item())
            progress_bar.set_postfix(**logs)
        model.eval()
        with torch.no_grad():
            print("Calculating validation loss  🔍🔍🔍🔍🔍🔍")
            test_loss = 0.0
            test_samples = 0
            progress_bar = tqdm(total=len(val_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in enumerate(val_dataloader):
                
                images, targets = batch

                pixel_values, pixel_mask, labels = model.preprocess(list(images), targets)

                # Move data to the correct device
                pixel_values = pixel_values.to(device)
                pixel_mask = pixel_mask.to(device)
                predictions = model.model(pixel_values = pixel_values,pixel_mask= pixel_mask, labels = labels)

                test_loss = predictions.loss

                # test_loss = F.mse_loss(predictions, labels)

        scheduler.step(test_loss)


        print(f'Epoch [{epoch+1}/{config.num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if test_loss < best_val_loss :
            best_val_loss = test_loss
            # torch.save(model.state_dict(), "saved_models/resnet_new")
            model.save("./deformable_detr_model")
        progress_bar.close()

    print("Training Completed   ✅✅✅ 💯💯💯")

        # if (epoch+1) % 1000 ==0:
        #     print("Saving model...")
        #     outdir = f"exps/{config.experiment_name}"
        #     os.makedirs(outdir, exist_ok=True)
        #     torch.save(model.state_dict(), f"{outdir}/model_{epoch+1}.pth")
