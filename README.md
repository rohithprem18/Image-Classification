# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a **Convolutional Neural Network (CNN)** for image classification using the **Fashion-MNIST** dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve **high accuracy** while maintaining **efficiency**.

## Neural Network Model
![425277678-8b69f643-0ff5-4643-b9ce-f9edd4bdfbdb](https://github.com/user-attachments/assets/6acab57a-cf5e-4963-a584-024b1d03e3e9)


## DESIGN STEPS

#### STEP 1: Problem Statement  
Define the objective of classifying fashion items (T-shirts, trousers, dresses, shoes, etc.) using a **Convolutional Neural Network (CNN)**.  

#### STEP 2: Dataset Collection  
Use the **Fashion-MNIST dataset**, which contains **60,000** training images and **10,000** test images of various clothing items.  

#### STEP 3: Data Preprocessing  
Convert images to tensors, normalize pixel values, and create **DataLoaders** for batch processing.  

#### STEP 4: Model Architecture  
Design a CNN with **convolutional layers**, **activation functions**, **pooling layers**, and **fully connected layers** to extract features and classify clothing items.  

#### STEP 5: Model Training  
Train the model using a suitable **loss function** (**CrossEntropyLoss**) and **optimizer** (**Adam**) for multiple epochs.  

#### STEP 6: Model Evaluation  
Test the model on unseen data, compute **accuracy**, and analyze results using a **confusion matrix** and **classification report**.  

#### STEP 7: Model Deployment & Visualization  
Save the trained model, visualize predictions, and integrate it into an application if needed.  

## PROGRAM

### Name: ROHITH PREM S
### Register Number: 212223040172
```python
class CNNClassifier (nn.Module):
  def __init__(self):
    super (CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128)
    self.fc2 = nn.Linear (128, 64)
    self.fc3 = nn.Linear (64, 10)
  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu (self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0), -1) # Flatten the image
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
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
    # Move model to GPU if available, inside the function
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Move images and labels to the same device as the model
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Name: ROHITH PREM S')
        print('Register Number: 212223040172')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
```

## OUTPUT

### Training Loss per Epoch
![Screenshot 2025-03-23 204655](https://github.com/user-attachments/assets/3c38dd7a-c070-43c8-9b70-bbf5b5366a88)

### Confusion Matrix
![Screenshot 2025-03-23 205844](https://github.com/user-attachments/assets/e98c535c-1b1d-4dec-8de8-759124b59883)

### Classification Report
![Screenshot 2025-03-23 204725](https://github.com/user-attachments/assets/4a1d4b5a-4ceb-40ad-af50-9d280c4c6954)

### New Sample Data Prediction
![Screenshot 2025-03-23 204746](https://github.com/user-attachments/assets/d08404dd-51f1-4fd9-8098-90abde851bbc)


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
