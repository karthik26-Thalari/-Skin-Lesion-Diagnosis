import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

class MedGradECLIP_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU())
        self.up1  = nn.ConvTranspose2d(128,64,2,stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2  = nn.ConvTranspose2d(64,32,2,stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(64,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32,1,1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b  = self.bottleneck(self.pool(e2))
        d1 = self.dec1(torch.cat([self.up1(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return torch.sigmoid(self.final(d2))

device = torch.device('cpu')
model  = MedGradECLIP_Small().to(device)
model.load_state_dict(torch.load('skin_lesion_best.pth', map_location=device))
model.eval()

tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def segment(image):
    tensor = tf(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        pred = model(tensor).squeeze().numpy()
    mask      = (pred > 0.35).astype(np.uint8) * 255
    mask_img  = Image.fromarray(mask).resize(image.size)
    overlay   = image.copy().convert('RGBA')
    mask_rgba = Image.new('RGBA', image.size, (255, 0, 0, 120))
    mask_bin  = mask_img.point(lambda p: 255 if p > 0 else 0)
    overlay.paste(mask_rgba, mask=mask_bin)
    return mask_img, overlay.convert('RGB')

demo = gr.Interface(
    fn=segment,
    inputs=gr.Image(type='pil', label='Upload Dermoscopy Image'),
    outputs=[
        gr.Image(type='pil', label='Segmentation Mask'),
        gr.Image(type='pil', label='Overlay'),
    ],
    title='Skin Lesion Segmentation',
    description='Trained on ISIC 2018 | Accuracy: 87.51% | F1: 78.37% | For research use only.',
    theme=gr.themes.Soft(),
)

if __name__ == '__main__':
    demo.launch()
