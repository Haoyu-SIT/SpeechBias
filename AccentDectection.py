from speechbrain.inference import EncoderClassifier
import os

# Create the model
model = EncoderClassifier.from_hparams(
  "Jzuluaga/accent-id-commonaccent_ecapa"
)

# File path
file_path = "/home/asian/Projects/AudioSynthesisBias/generated_speech_expanded_v2/accent_nurse_african/nurse_african_content1.wav"

# Run classification and extract just the accent label
result = model.classify_file(file_path)
accent_label = result[-1][0]  # Get the first element of the last item in the tuple

# Get just the filename without the path
file_name = os.path.basename(file_path)


# Print file name and accent label
print(f"File: {file_name}, Accent: {accent_label}")