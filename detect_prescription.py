import easyocr
import cv2
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# -----------------------------
# Load models ONCE
# -----------------------------
reader = easyocr.Reader(['en'])

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("prescription_model")


def detect_medicines(image_path):

    image = cv2.imread(image_path)

    results = reader.readtext(image_path)

    detected_medicines = {}

    # -----------------------------
    # Process detected regions
    # -----------------------------
    for bbox, text, prob in results:

        x1, y1 = map(int, bbox[0])
        x2, y2 = map(int, bbox[2])

        width = x2 - x1
        height = y2 - y1

        # Ignore very small text boxes
        # if width < 150 or height < 40:
        #     continue

        crop = image[y1:y2, x1:x2]
        crop_pil = Image.fromarray(crop).convert("RGB")

        # -----------------------------
        # Run TrOCR
        # -----------------------------
        pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values

        generated_ids = model.generate(pixel_values)

        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        print("Raw prediction:", prediction)

        # -----------------------------
        # Extract medicine and generic
        # -----------------------------
        prediction = prediction.lower()

        medicine = None
        generic = None

        if "medicine:" in prediction:

            medicine = prediction.split("medicine:")[1]

            if "generic:" in medicine:
                medicine = medicine.split("generic:")[0]

            medicine = medicine.strip().title()

        if "generic:" in prediction:
            generic = prediction.split("generic:")[1].strip().title()

        # -----------------------------
        # Store unique medicines
        # -----------------------------
        if medicine:
            detected_medicines[medicine] = generic if generic else "Unknown"

    # -----------------------------
    # Print result
    # -----------------------------
    print("\nDetected Medicines:\n")

    for med, gen in detected_medicines.items():
        print("Medicine:", med, "| Generic:", gen)

    return detected_medicines





# import easyocr
# import cv2
# from PIL import Image
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# # -----------------------------
# # Load models ONCE
# # -----------------------------
# reader = easyocr.Reader(['en'])

# processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
# model = VisionEncoderDecoderModel.from_pretrained("prescription_model")


# def detect_medicines(image_path):

#     image = cv2.imread(image_path)

#     results = reader.readtext(image_path)

#     detected_medicines = {}

#     # -----------------------------
#     # Process detected regions
#     # -----------------------------
#     for bbox, text, prob in results:

#         x1, y1 = map(int, bbox[0])
#         x2, y2 = map(int, bbox[2])

#         width = x2 - x1
#         height = y2 - y1

#         # Ignore very small text boxes
#         if width < 150 or height < 40:
#             continue

#         crop = image[y1:y2, x1:x2]
#         crop_pil = Image.fromarray(crop).convert("RGB")

#         # -----------------------------
#         # Run TrOCR
#         # -----------------------------
#         pixel_values = processor(images=crop_pil, return_tensors="pt").pixel_values

#         generated_ids = model.generate(pixel_values)

#         prediction = processor.batch_decode(
#             generated_ids,
#             skip_special_tokens=True
#         )[0]

#         print("Raw prediction:", prediction)

#         # -----------------------------
#         # Extract medicine and generic
#         # -----------------------------
#         prediction = prediction.lower()

#         medicine = None
#         generic = None

#         if "medicine:" in prediction:

#             medicine = prediction.split("medicine:")[1]

#             if "generic:" in medicine:
#                 medicine = medicine.split("generic:")[0]

#             medicine = medicine.strip().title()

#         if "generic:" in prediction:
#             generic = prediction.split("generic:")[1].strip().title()

#         # -----------------------------
#         # Store unique medicines
#         # -----------------------------
#         if medicine and generic:
#             detected_medicines[medicine] = generic

#     # -----------------------------
#     # Print result
#     # -----------------------------
#     print("\nDetected Medicines:\n")

#     for med, gen in detected_medicines.items():
#         print("Medicine:", med, "| Generic:", gen)

#     return detected_medicines




