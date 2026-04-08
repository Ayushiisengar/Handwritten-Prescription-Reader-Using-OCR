from detect_prescription import detect_medicines
from difflib import SequenceMatcher

# -----------------------------
# Similarity Function (for fuzzy matching)
# -----------------------------
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


# -----------------------------
# Test Dataset (ADD YOUR IMAGES HERE)
# -----------------------------
test_images = [
    "23.png",
    "72.png",
    "191.png",
    "261.png",
    "310.png",
    "359.png",
    "377.png",
    "394.png",
    "415.png",
    "471.png"
]

# Ground Truth (correct answers)
actual_data = [
    {"Alatrol": "Cetirizine Hydrochloride"},
    {"Azyth": "Azithromycin Dihydrate"},
    {"Esonix": "Esomeprazole"},
    {"Esoral":"Esomeprazole"},
    {"Fexofast":"Fexofenadine Hydrochloride"},
    {"Flexibac":"Baclofen"},
    {"Flugal":"Fluconazole"},
    {"Ketoral":"Ketoconazole (Tablet)"},
    {"Ketozol":"Ketoconazole (Shampoo)"},
    {"Maxpro":"Esomeprazole"}
]


# -----------------------------
# Accuracy Calculation
# -----------------------------
correct = 0
total = 0

for img, actual in zip(test_images, actual_data):

    print("\n==============================")
    print("Testing Image:", img)

    predicted = detect_medicines(img)

    print("Predicted:", predicted)
    print("Actual:", actual)

    for actual_med, actual_gen in actual.items():
        total += 1
        match_found = False

        for pred_med, pred_gen in predicted.items():

            med_sim = similar(pred_med.lower(), actual_med.lower())
            gen_sim = similar(pred_gen.lower(), actual_gen.lower())

            print(f"Comparing -> {pred_med} vs {actual_med} | Similarity: {med_sim}")
            print(f"Comparing -> {pred_gen} vs {actual_gen} | Similarity: {gen_sim}")

            # Threshold (you can adjust 0.7 → 0.6 if needed)
            if med_sim > 0.7 and gen_sim > 0.7:
                correct += 1
                match_found = True
                break

        if not match_found:
            print("❌ No match found")

# -----------------------------
# Final Accuracy
# -----------------------------
accuracy = (correct / total) * 100 if total > 0 else 0

print("\n==============================")
print("FINAL RESULTS")
print("Correct:", correct)
print("Total:", total)
print("Accuracy:", accuracy, "%")