from world_modality.train import main as train_main

if __name__ == "__main__":
    import sys

    sys.argv.append("--model_type")
    sys.argv.append("C")
    train_main()

