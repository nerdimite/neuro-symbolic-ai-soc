import numpy as np
import pandas as pd
import os
import dlib
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms as tfms
from PIL import Image
from tqdm.notebook import tqdm


def train_detector(train_data, filename='detector.svm'):
    '''Trains an object detector (HOG + SVM) and saves the model'''
    
    # Seperate the images and bounding boxes in different lists.
    images = [val[0] for val in train_data.values()]
    bounding_boxes = [val[1] for val in train_data.values()]
    
    # Initialize object detector Options
    options = dlib.simple_object_detector_training_options()
    options.add_left_right_image_flips = False
    options.C = 5
    
    # Train the model
    detector = dlib.train_simple_object_detector(images, bounding_boxes, options)
    
    # Check results
    results = dlib.test_simple_object_detector(images, bounding_boxes, detector)
    print(f'Training Results: {results}')
    
    # Save model
    detector.save(filename)
    print(f'Saved the model to {filename}')

    
class ShapeClassifier(nn.Module):
    '''Simple CNN based Image Classifier for Shapes (circle | rectangle)'''
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, 28, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(28),
                                  nn.Conv2d(28, 28, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(28),
                                  nn.Conv2d(28, 28, 3, stride=2, padding=1),
                                  nn.BatchNorm2d(28))
        self.fc = nn.Linear(700, 1)
    
    def forward(self, x):
        '''Forward Pass'''
        # batch_size (N)
        N = x.size()[0]
        # Extract features with CNN
        x = self.conv(x)
        # Classifier head
        x = self.fc(x.reshape(N, -1))
        
        return x
    
    def train_classifier(self, train_loader, lr=0.0001, epochs=10, filename='classifier.pth', device=None):
        '''Train the shape classifier'''
        # Automatically set device if not provided
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Mount to device
        self.to(device)
        
        # Create optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        self.train()
        # Start Training
        for epoch in range(epochs):
            pbar = tqdm(total=len(train_loader), desc='Epoch {}'.format(epoch+1))
            losses = []
            
            for i, (image, label) in enumerate(train_loader):
                # Mount to device
                image, label = image.to(device).float(), label.to(device)
                
                # Forward prop
                out = self(image)
                
                # Loss
                loss = criterion(out.squeeze(1), label.float())
                
                # Backprop and Optimization
                loss.backward()
                optimizer.step()
                
                # Verbose
                losses.append(loss.item())
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
            
            print(f'Epoch {epoch+1}: Mean Loss = {sum(losses)/len(losses)}')
            pbar.close()
            
        # Save model
        torch.save(self.state_dict(), filename)

class Binarize(object):
    def __init__(self):
        '''Converts Grayscale to Binary (except white every other color is zeroed)'''
        pass
    
    def __call__(self, img_tensor):
        '''
        Args:
            img_tensor (tensor): 0-1 scaled tensor with 1 channel
        Returns:
            tensor
        '''
        return (img_tensor > 0.95).float()

class PerceptionPipe():
    '''
    Full Perception Pipeline i.e.
    detector -> attribute extraction -> structural scene representation
    '''
    def __init__(self, detector_file, classifer_file, device='cpu'):
        # Object detector
        self.detector = dlib.simple_object_detector(detector_file)
        
        # Shape Classifier
        self.classifier = ShapeClassifier().to(device)
        self.classifier.load_state_dict(torch.load(classifer_file))
        self.device = device
        
        self.colors = np.array([[0,0,255], [0,255,0], [255,0,0], 
                               [0,156,255], [128,128,128], [0,255,255]])
        
        self.idx2color = {0: 'red', 1: 'green', 2: 'blue', 3: 'orange', 4: 'gray', 5: 'yellow'}
        self.preproc = tfms.Compose([tfms.Grayscale(),
                                     tfms.Resize((40, 40)),
                                     tfms.ToTensor(),
                                     Binarize()])
    
    
    def detect(self, img):
        '''Detects and Returns Objects and its centers'''
        # Detect
        detections = self.detector(img)
        objects = []
        
        for detection in detections:
            # Get the bbox coords
            x1, y1 = int(detection.left()), int(detection.top())
            x2, y2 = int(detection.right()), int(detection.bottom())
            
            # Clip negative values to zero
            x1, y1, x2, y2 = np.array([x1, y1, x2, y2]).clip(min=0).tolist()

            # Find the center
            center = (int((x1+x2)/2), int((y1+y2)/2))

            # Crop the individual object
            obj = img[y1:y2, x1:x2]

            objects.append((obj, center))
            
        return objects
    
    
    def extract_attributes(self, x_img, prob=0.5, debug=False):
        '''Returns the shape and color of a given object'''
        # Load image as PIL instance (color image)
        image = Image.fromarray(cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB))
        # Preprocess (binarized image)
        img = self.preproc(image).unsqueeze(0).to(self.device)
        
        # Predict Shape
        with torch.no_grad():
            out = torch.sigmoid(self.classifier(img)).squeeze()
            if debug:
                print(out)
        if out < prob:
            shape = 'circle'
        else:
            shape = 'rectangle'
            
        # Extract Color
        center_pixel = (x_img[20, 20, :]).astype('int')
        
        color_id = cosine_similarity(center_pixel.reshape(1, -1), self.colors).argmax()
        color = self.idx2color[color_id]
        
#         print(center_pixel)
#         print(color_id)
        
        return shape, color
    
    def scene_repr(self, img, prob=0.5, debug=False):
        '''Returns a structured scene representation as a dataframe'''
        # Perform object detection and get the objects
        objects = self.detect(img)
        
        # Init Scene representation
        scene_df = pd.DataFrame(columns=['shape', 'color', 'position'])
        
        for obj, center in objects:
            shape, color = self.extract_attributes(obj, prob, debug)
            scene_df = scene_df.append({'shape': shape, 
                                        'color': color, 
                                        'position': center}, ignore_index=True)
        
        return scene_df