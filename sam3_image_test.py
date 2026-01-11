import torch
from PIL import Image
import matplotlib.pyplot as plt

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results

device = "cuda" if torch.cuda.is_available() else "cpu"

# 离线/没网时：传本地权重，并关掉 HF 下载
# model = build_sam3_image_model(device=device, load_from_HF=False, checkpoint_path="path/to/sam3.pt")
model = build_sam3_image_model(device=device)

processor = Sam3Processor(model, device=device, confidence_threshold=0.5)

image = Image.open("rgb.jpg").convert("RGB")
state = processor.set_image(image)

output = processor.set_text_prompt(state=state, prompt="computer")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# 可视化
plot_results(image, output)
plt.axis("off")
plt.show()

# 导出给 Diffuser 的 label 图（0/1；不要存 0/255 直接喂给 Diffuser）
if scores.numel() > 0:
    best = int(scores.argmax().item())
    label01 = masks[best].squeeze(0).to(torch.uint8).cpu().numpy()
    Image.fromarray(label01).save("labels.png")
