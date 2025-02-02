from datasets import load_dataset

# Load the dataset and save it to the folder "shapes_dataset"
dataset = load_dataset("0-ma/geometric-shapes")
dataset.save_to_disk("./shapes_dataset")
print("Dataset saved to ./shapes_dataset")
