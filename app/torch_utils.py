import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

#load model
#image to tensor
#predict using model

input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10

#load model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
model = NeuralNet(input_size, hidden_size, num_classes)
PATH = "mnist_ffn.pth"
model.load_state_dict(torch.load(PATH))
model.eval()


#image to tensor
def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((28, 28)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
    image = Image.open(io.BytesIO(image_bytes))#reads the binary data and creates a pillow image object
    #this allows the transforms to be applied to the image
    #greyscale and resiye work on PIL images. toTenser converts to tensor
    #we need to do unsqueeze to add a batch dimension
    #our model is designed for batches of images
    #unsqueeze does NOT add fake samples to make a batch, instead ot
    #adds a dimension of size 1 to the tensor at the specified position(the batch dimension)
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    #we are using a fully connected network, so we need to flatten the image
    #-1 means the size of that dimension is inferred from other dimensions at runtime
    #28*28 is the size of each image, so it sees 784 elements, so the first num 
    #can only be 1
    #this, of course, is called after transform_image
    labels = model(images)
    _, predicted = torch.max(labels.data, 1)
    return predicted.item() 