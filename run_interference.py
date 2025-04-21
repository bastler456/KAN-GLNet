import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Replace with your actual model class and loading method
from models.pointnet2_sem_seg import get_model

model = get_model(40)
checkopoint = torch.load("log/sem_seg/pointnet2_sem_seg/checkpoints/pointnet2_sem_seg.pth")
model.load_state_dict(checkopoint["model_state_dict"])
model.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

model.to(device)
pcd = o3d.io.read_point_cloud("data/random_test_set/CSite1_orig-utm.pcd")
pcd = pcd.voxel_down_sample(voxel_size=0.05)
points = np.asarray(pcd.points)
points_tensor = torch.tensor(points, dtype=torch.float32)
points_tensor = points_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    pred, _ = model(points_tensor.transpose(2, 1))  # assuming model expects [B, 3, N]
    pred_label = pred.argmax(dim=1)  # shape: (1, N)
    pred_label = pred_label.squeeze().cpu().numpy()


# Define a colormap (assign unique RGB colors to each label)
def get_colormap(n_classes):
    cmap = plt.get_cmap("tab20", n_classes)
    return np.array([cmap(i)[:3] for i in range(n_classes)])  # RGB

# Get color for each point
num_classes = 40  # or whatever your model was trained on
colormap = get_colormap(num_classes)
colors = colormap[pred_label]  # shape: (N, 3)

# Assign colors to point cloud
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
o3d.visualization.draw_geometries([pcd])