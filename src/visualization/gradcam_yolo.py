from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2, torch, matplotlib.pyplot as plt

model = YOLO("../weights/best.pt").model
target_layers = [model.model[-2]]

img = cv2.imread("sample.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_norm = img_rgb / 255.0
tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).float()

cam = GradCAM(model=model, target_layers=target_layers)
heatmap = cam(input_tensor=tensor)[0]

overlay = show_cam_on_image(img_norm, heatmap, use_rgb=True)
plt.imshow(overlay)
plt.axis("off")
plt.show()
