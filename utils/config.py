



def run_model(root_dir, use_GPU=True):
    config = dict()
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["lr"] = 1e-4
    config["batch_size"] = 64
    config["num_epochs"] = 3
    config["num_workers"] = 2
    config["num_classes"] = 1
    #config["image_height"] = 160 # 1280 originally
    #config["image_width"] = 240 # 1918 originally
    config["pin_memory"] = True
    config["load_model"] = False
    config["img_dir"] = "data/Kvasir-SEG/images/"
    config["mask_dir"] = "data/Kvasir-SEG/masks/"




if __name__ == "__main__":
    run_model()