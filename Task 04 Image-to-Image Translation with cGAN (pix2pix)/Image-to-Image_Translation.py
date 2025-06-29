import torch
import torchvision.transforms as T
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained Pix2Pix model
model = torch.hub.load('phillipi/pix2pix', 'facades_label2photo', pretrained=True).eval()

# Load input image (segmentation map)
url = "https://raw.githubusercontent.com/phillipi/pix2pix/master/datasets/facades/test/1_label.png"
response = requests.get(url)
input_image = Image.open(BytesIO(response.content)).convert('RGB')

# Preprocess image
transform = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

input_tensor = transform(input_image).unsqueeze(0)

# Generate translated image
with torch.no_grad():
    output = model(input_tensor)[0].detach().cpu()

# Post-process and save image
output_image = T.ToPILImage()(output * 0.5 + 0.5)
output_image.save('translated_image.png')
