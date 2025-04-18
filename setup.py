import kagglehub

# Download latest version
path = kagglehub.dataset_download("balraj98/modelnet40-princeton-3d-object-dataset")


print("Path to dataset files:", path)