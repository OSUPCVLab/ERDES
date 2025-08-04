import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

# Import your own model class and dataset class
from your_model_module import Model
from your_dataset_module import VideoDataset      # <- Your dataset that handles video processing

# Paths to your pre-trained Lightning checkpoints
RD_CKPT_PATH = "checkpoints/rd_classifier.ckpt"
MD_CKPT_PATH = "checkpoints/md_classifier.ckpt"

# Threshold for binary classification
THRESHOLD = 0.5

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models from your custom model class
rd_classifier = Model.load_from_checkpoint(RD_CKPT_PATH).to(device)
md_classifier = Model.load_from_checkpoint(MD_CKPT_PATH).to(device)

# Set both models to evaluation mode
rd_classifier.eval()
md_classifier.eval()

# Load your video dataset (assumes each sample is one video)
video_dataset = VideoDataset(video_path="path/to/input/video.mp4")  # Adjust as needed
dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False)

# Inference loop
with torch.no_grad():
    for i, batch in enumerate(dataloader):
        # Move input to device
        inputs = batch.to(device)

        # Step 1: RD Classifier
        rd_logits = rd_classifier(inputs)
        rd_probs = torch.sigmoid(rd_logits)
        rd_pred = (rd_probs > THRESHOLD).int()

        if rd_pred.item() == 1:
            # Step 2: MD Classifier
            md_logits = md_classifier(inputs)
            md_probs = torch.sigmoid(md_logits)
            md_pred = (md_probs > THRESHOLD).int()

            if md_pred.item() == 1:
                print(f"Video {i}: ✅ Macula Intact")
            else:
                print(f"Video {i}: ❌ Macula Detached")
        else:
            print(f"Video {i}: Skipped (No Retinal Detachment)")
