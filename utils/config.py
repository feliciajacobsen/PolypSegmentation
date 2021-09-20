from dataloader import PolypDataset(Dataset)
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#local imports
from utils import get_mean_std
from dataloader import PolypDataset


def run_model(root_dir, use_GPU=True):
    config = dict()
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["lr"] = 1e-4
    config["batch_size"] = 64
    config["num_epochs"] = 3
    config["num_workers"] = 2
    config["num_classes"] = 1
    config["image_height"] = 256 # 1280 originally
    config["image_width"] = 240 # 1918 originally
    config["pin_memory"] = True
    config["load_model"] = False
    config["img_dir"] = "data/Kvasir-SEG/images/"
    config["mask_dir"] = "data/Kvasir-SEG/masks/"

    data_set = 

    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=True)

    mean, std = get_mean_std(train_loader)



    train_transforms = A.Compose([
        A.Resize(width=config["image_width"], height=config["image_height"]),
        A.Rotate(limit=40, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    val_transforms = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(
            mean=[0, 0, 0],
            std=[1, 1, 1],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


if __name__ == "__main__":
    run_model()