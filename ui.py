import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms
import db
import atexit


# Define the model architecture (match exactly with app.py)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the pre-trained PyTorch model more efficiently
@st.cache_resource
def load_model():
    model = SimpleCNN()
    model.load_state_dict(torch.load('mnist_model.pth'))
    model.eval()
    return model

model = load_model()

# Initialize database
try:
    db.init_db()
    # Register cleanup function to close database connections
    atexit.register(db.close_db)
    st.sidebar.success("Connected to database!")
except Exception as e:
    st.sidebar.error(f"Database connection failed: {e}")

# Title of the app
st.title('Draw a Digit')

# Canvas for drawing
canvas_result = st_canvas(
    fill_color="white",  # White background for the canvas
    stroke_width=5,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

def preprocess_image(image):
    # Debug preprocessing steps
    st.write("Processing image...")
    
    # Convert to grayscale
    gray_img = image.convert('L')
    # st.image(gray_img, caption="Grayscale", width=150)
    
    # Resize to 28x28
    resized_img = gray_img.resize((28, 28))
    
    # Invert colors (MNIST has white digits on black background)
    inverted_img = Image.eval(resized_img, lambda x: 255 - x)
    # st.image(inverted_img, caption="Resized & Inverted (28x28)", width=150)
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Use exact same normalization as training
    ])
    tensor_img = transform(inverted_img).unsqueeze(0)  # Add batch dimension
    
    # Show tensor stats
    # st.write(f"Tensor shape: {tensor_img.shape}, Min: {tensor_img.min().item():.2f}, Max: {tensor_img.max().item():.2f}")
    
    return tensor_img

if canvas_result.image_data is not None:
    # Convert canvas to image
    img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('RGB')
    
    # Check if there's actually a drawing (more than just white background)
    # Convert to grayscale and check if there are any non-white pixels
    img_array = np.array(img.convert('L'))
    if np.any(img_array < 250):  # Check if there are any dark pixels (value < 250)
        img_tensor = preprocess_image(img)  # Preprocess image for PyTorch
        
        # Make prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            values, predicted_digit = torch.max(probs, 1)
        
        # Convert to Python native types
        pred_digit = predicted_digit.item()
        confidence = values.item()
        
        # Show the prediction and confidence
        st.write(f"Predicted Digit: {pred_digit} (Confidence: {confidence:.2f})")
        
        # Allow user to provide the true label
        true_label = st.number_input(
            "Enter the true digit (optional):", 
            min_value=0, 
            max_value=9, 
            value=None,
            step=1,
            key="true_label")
        
        # Log button
        if st.button("Log this prediction"):
            try:
                success = db.log_prediction(
                    predicted_digit=pred_digit,
                    true_label=true_label if true_label is not None else None,
                    confidence=confidence
                )
                if success:
                    st.success("Prediction logged successfully!")
                else:
                    st.error("Failed to log prediction")
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # # Show bar chart of all predictions
        # fig = plt.figure(figsize=(10, 4))
        # plt.bar(range(10), probs[0].cpu().numpy())
        # plt.xticks(range(10))
        # plt.xlabel('Digit')
        # plt.ylabel('Confidence')
        # plt.title('Prediction Confidence for Each Digit')
        # st.pyplot(fig)
    else:
        st.write("Draw a digit to see the prediction")
