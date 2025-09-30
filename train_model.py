from model import unet
import numpy as np

# Load data
def load_data():
    # Replace with your dataset loading logic
    X = np.random.rand(100, 256, 256, 3)
    y = np.random.rand(100, 256, 256, 3)
    return X, y

model = unet()
model.compile(optimizer='adam', loss='binary_crossentropy')
X_train, y_train = load_data()
model.fit(X_train, y_train, epochs=10, batch_size=8)
