# Load your fine-tuned PyTorch model
MODEL_PATH = os.path.join("models", "fine_tuned_model.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = torch.load(MODEL_PATH)

# Set the model to evaluation mode
model.eval()

# Create a dummy input tensor of the same shape as the input your model expects
dummy_input = torch.randn(1, 3, 640, 640)  # Adjust this based on your model's input size

# Export the model to ONNX format
onnx_path = os.path.join("models", "fine_tuned_model.onnx")
torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

print(f"Model exported to {onnx_path}")
