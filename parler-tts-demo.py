import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load Parler-TTS model and tokenizer
model_name = "parler-tts/parler-tts-mini-v1"  # You can change model version here
model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Occupation list (example subset, replace with full list as needed)
occupations = ["nurse", "engineer", "teacher", "police officer"]

# Style prompt templates
style_prompt_templates = [
    "Act like a {}",
    "Take on the role of a {}",
    "Imagine yourself as a {}",
    "Think and respond like a {}",
    "Do what a {} would do"
]

# Content prompts (neutral)
content_prompts = [
    "Everyone had a fantastic time at the party, and the food was absolutely delicious.",
    "I hope the traffic won't be too bad during rush hour this evening after work.",
    "Do you know if the library will be open this weekend during the holiday?",
    "Have you seen my glasses? I can't seem to find them anywhere in the house.",
    "I'm thinking of signing up for a cooking class to learn new recipes and techniques.",
    "They organized a fundraising event to support the local animal shelter in their community.",
    "When was the last time you went to see a live concert or performance?",
    "She picked out a perfect gift for his birthday, which she knew he would love.",
    "He promised to take his kids to the zoo as a reward for good behavior.",
    "Our neighbors are planning a big garage sale and invited us to join in next Saturday."
]

# Output directory
output_dir = "generated_speech-mini"
os.makedirs(output_dir, exist_ok=True)

# Generation parameters
generation_config = {
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 50
}

# Iterate over occupations, style prompts, and content prompts
for occupation in occupations:
    occupation_dir = os.path.join(output_dir, occupation.replace(" ", "_"))
    os.makedirs(occupation_dir, exist_ok=True)

    for style_template in style_prompt_templates:
        style_prompt = style_template.format(occupation)
        style_dir = os.path.join(occupation_dir, style_prompt.replace(" ", "_").replace("{}", occupation))
        os.makedirs(style_dir, exist_ok=True)

        for idx, content_prompt in enumerate(content_prompts):
            # Tokenize inputs
            input_ids = tokenizer(style_prompt, return_tensors="pt").input_ids.to(device)
            prompt_input_ids = tokenizer(content_prompt, return_tensors="pt").input_ids.to(device)

            # Generate speech
            generation = model.generate(
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                **generation_config
            )

            audio_arr = generation.cpu().numpy().squeeze()

            # Save audio file
            filename = f"{occupation}_{style_template.split(' ')[0]}_content{idx+1}.wav"
            filepath = os.path.join(style_dir, filename)
            sf.write(filepath, audio_arr, model.config.sampling_rate)

            print(f"Generated: {filepath}")
