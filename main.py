import os
import sys
import ast
import json
import torch
from PIL import Image
from torchvision import models, transforms

# parse command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('image_name', help='the input image name')
parser.add_argument('labels', help='the data labels filepath')
parser.add_argument('--output', default='results.json', help='results filepath')
args = parser.parse_args()

# define data directories
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT, 'data')
WEIGHTS_PATH = os.path.join(ROOT, 'weights/resnet101_weights.pth')

# define device variable to set model to the best available resource
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main():
    # load pretrained model (for this tutorial we are going to use ResnNet101 for image classification)
    weights = torch.load(WEIGHTS_PATH)
    resnet_model = models.resnet101()
    resnet_model.load_state_dict(weights)
    
    # set model to device
    resnet_model.to(device)

    # load data and do any required preprocessing
    # data
    image = Image.open(os.path.join(DATA_PATH, args.image_name))
    # labels
    with open(os.path.join(args.labels), 'r') as f:
        labels = ast.literal_eval(f.read())

    # define data transform
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # apply data transformation
    img_t = transform(image)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    # run inference
    resnet_model.eval()
    output = resnet_model(batch_t)
    # convert ouptuts to softmax probabilities
    percentage = torch.nn.functional.softmax(output, dim=1)[0]

    _, indices = torch.sort(output, descending=True)
    top5_preds = [(labels[idx.item()], percentage[idx].item()) for idx in indices[0][:5]]

    # save output
    results = {'results': top5_preds}
    print(results)

    with open(args.output, 'w') as output_path:
        json.dump(results, output_path)



if __name__ == "__main__":
    main()