# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiven

## Neural Network Model
<img width="1043" height="508" alt="Screenshot 2026-02-13 155140" src="https://github.com/user-attachments/assets/0ba99647-5c2d-41f8-b244-5a7fb5985897" />








## DESIGN STEPS

### STEP 1: Problem Statement
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:Dataset Collection
Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.
### STEP 3: Data Preprocessing
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.
### STEP 4:Model Architecture
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.
### STEP 5:Model Training
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.
### STEP 6:Model Evaluation
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.
### STEP 7: Model Deployment & Visualization
Save the trained model, visualize predictions, and integrate it into an application if needed.


## PROGRAM

### Name: SOMALARAJU ROHINI
### Register Number: 212224240156
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
      x=self.pool(torch.relu(self.conv1(x)))
      x=self.pool(torch.relu(self.conv2(x)))
      x=self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0), -1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x  


```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: SOMALARAJU ROHINI')
        print('Register Number: 212224240156')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch




<img width="515" height="153" alt="Screenshot 2026-02-13 154858" src="https://github.com/user-attachments/assets/7c2eda9e-8613-4581-82bd-c558c82b20e2" />



### Confusion Matrix




<img width="749" height="668" alt="Screenshot 2026-02-13 154924" src="https://github.com/user-attachments/assets/4d6bf1a8-25ff-4fef-97b4-253b5a0da6b6" />



### Classification Report




<img width="1506" height="440" alt="Screenshot 2026-02-13 154935" src="https://github.com/user-attachments/assets/2c0a4888-a988-4918-9aa1-0638b511a93c" />




### New Sample Data Prediction




<img width="650" height="214" alt="Screenshot 2026-02-13 155609" src="https://github.com/user-attachments/assets/ec6d7b56-588d-4c86-8ec6-39854a46e5a6" />





<img width="486" height="396" alt="Screenshot 2026-02-13 154940" src="https://github.com/user-attachments/assets/6811daad-eb4e-42be-b49b-fccf78a1cf5f" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
