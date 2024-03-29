import torch
from ultralytics import YOLO
# Load your TorchScript model from file
model_script = torch.jit.load("models/v8n_relu_DETRAC2.torchscript")

# Set model to evaluation mode
model_script.eval()

# Example input (shape should match your model's input shape)
example_input = torch.randn(1, 3, 640, 640)

# Export model to ONNX format
torch.onnx.export(model_script,                        # TorchScript model
                  example_input,                      # Example input tensor
                  "models/v8n_relu_DETRAC2_quantized_fb.onnx",    # Path to save the ONNX model
                  opset_version=13)                  # ONNX opset version

# model = YOLO('models/v8n_relu_DETRAC2.pt')
# model.export(format='onnx')