# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset
Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiven



<img width="767" height="789" alt="download-(8)" src="https://github.com/user-attachments/assets/0671b457-8ed9-4ef8-aa75-64b51de11a01" />


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

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1,28,28) → (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                         # → (16,14,14)

            nn.Conv2d(16, 32, kernel_size=3, padding=1), # → (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                          # → (32,7,7)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 classes
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x 


```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # ✅ This print MUST be inside the epoch loop
        print('Name: Somalaraju Rohini')
        print('Register Number: 212224240156')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch



<img width="848" height="312" alt="Screenshot 2026-03-17 155429" src="https://github.com/user-attachments/assets/82791f68-fa91-499a-a97b-01365c77a447" />




### Confusion Matrix


<img width="890" height="703" alt="Screenshot 2026-03-17 155626" src="https://github.com/user-attachments/assets/5e01e2ea-7742-425b-af00-2ac8a020e5bc" />




### Classification Report



<img width="588" height="380" alt="Screenshot 2026-03-17 155649" src="https://github.com/user-attachments/assets/9a85c665-0f61-4b20-96cb-dd77c66d514e" />




### New Sample Data Prediction


<img width="930" height="577" alt="Screenshot 2026-03-17 155718" src="https://github.com/user-attachments/assets/39355d10-db25-484b-9daa-282c959d975c" />




## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
