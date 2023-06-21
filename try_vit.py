from torchvision import transforms
import torch
import urllib
from PIL import Image
from torchvision.models.vision_transformer import vit_b_16

transform_test = transforms.Compose([
    transforms.Resize((300, 300), Image.BILINEAR),
    transforms.CenterCrop((224, 224)),
    # transforms.RandomHorizontalFlip(),  # only if train
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device':'cpu', 'num_classes': 200})
# model.eval()
model = vit_b_16(pretrained=True)
model.eval()
url = 'https://raw.githubusercontent.com/nicolalandro/ntsnet-cub200/master/images/nts-net.png'
img = Image.open(urllib.request.urlopen(url))
scaled_img = transform_test(img)
torch_images = scaled_img.unsqueeze(0)
logits = model(torch_images)
_, predict = torch.max(logits, 1)
pred_id = predict.item()
print(pred_id)
