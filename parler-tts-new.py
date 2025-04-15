import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import random

# Select the runtime device (prefer GPU if available)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Parler-TTS model and tokenizer
model_name = "parler-tts/parler-tts-large-v1"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Neutral content prompts (without any emotion or style)
content_prompts = [
    "Everyone had a fantastic time at the party, and the food was absolutely delicious.",
    "I hope the traffic won't be too bad during rush hour this evening after work.",
    "Do you know if the library will be open this weekend during the holiday?"
]

# Task description:
# For each attribute type, generate two versions:
# - style prompt + gender label
# - style prompt + accent label
attributes = [
    {"attribute": "nurse", "type": "occupation", "gender": "female"},
    {"attribute": "engineer", "type": "occupation", "gender": "male"},
    {"attribute": "happy", "type": "emotion", "gender": "female"},
    {"attribute": "fast", "type": "speaking rate", "gender": "male"},
    {"attribute": "high", "type": "pitch", "gender": "female"}
]

# 16 types of accents for accent style prompts
accents = [
    "african", "australian", "bermuda", "canadian", "english", "hongkong",
    "indian", "irish", "malaysian", "newzealand", "philippines", "scottish",
    "singaporean", "southatlantic", "us", "welsh"
]

# Set output directory
output_dir = "generated_speech_expanded_v2"
os.makedirs(output_dir, exist_ok=True)

# Speech generation configuration
generation_config = {
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 50
}

# ------------------------------------
# Generate speech with gender-associated style prompts
# ------------------------------------
for attr in attributes:
    gender_label = attr['gender']
    style_prompt = f"Speak in a manner that reflects someone who is {attr['attribute']}, typically associated with a {gender_label} voice."

    style_dir = os.path.join(output_dir, f"gender_{attr['attribute']}")
    os.makedirs(style_dir, exist_ok=True)

    for content_idx, content_prompt in enumerate(content_prompts):
        # Encode style and content text
        input_ids = tokenizer(style_prompt, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(content_prompt, return_tensors="pt").input_ids.to(device)

        # Generate speech
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
            **generation_config
        )

        # Save generated speech as .wav file
        audio_arr = generation.cpu().numpy().squeeze()
        filename = f"{attr['attribute']}_gender_content{content_idx+1}.wav"
        filepath = os.path.join(style_dir, filename)
        sf.write(filepath, audio_arr, model.config.sampling_rate)
        print(f"Generated (gender): {filepath}")

# ------------------------------------
# Generate speech with accent-associated style prompts
# ------------------------------------
for attr in attributes:
    for accent in accents:
        accent_label = accent.capitalize() if accent != "us" else "American"
        style_prompt = f"Speak like a {attr['attribute']} with a subtle {accent_label} accent, without explicitly naming the region."

        style_dir = os.path.join(output_dir, f"accent_{attr['attribute']}_{accent}")
        os.makedirs(style_dir, exist_ok=True)

        for content_idx, content_prompt in enumerate(content_prompts):
            input_ids = tokenizer(style_prompt, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(content_prompt, return_tensors="pt").input_ids.to(device)

            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                **generation_config
            )

            audio_arr = generation.cpu().numpy().squeeze()
            filename = f"{attr['attribute']}_{accent}_content{content_idx+1}.wav"
            filepath = os.path.join(style_dir, filename)
            sf.write(filepath, audio_arr, model.config.sampling_rate)
            print(f"Generated (accent): {filepath}")
