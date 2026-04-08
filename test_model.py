from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Load processor from base model
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

# Load your trained model
model = VisionEncoderDecoderModel.from_pretrained("prescription_model")

image = Image.open("377.png").convert("RGB")

pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)

prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Prediction:", prediction)