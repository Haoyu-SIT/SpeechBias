import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import random

# 选择运行设备（GPU 优先）
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载 Parler-TTS 模型和 tokenizer
model_name = "parler-tts/parler-tts-large-v1"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 生成用的内容文本（neutral prompt，不带任何情绪或风格）
content_prompts = [
    "Everyone had a fantastic time at the party, and the food was absolutely delicious.",
    "I hope the traffic won't be too bad during rush hour this evening after work.",
    "Do you know if the library will be open this weekend during the holiday?"
]

# 任务说明：
# 对每个属性类型，分别生成 style prompt + gender label 与 style prompt + accent label 的版本
attributes = [
    {"attribute": "nurse", "type": "occupation", "gender": "female"},
    {"attribute": "engineer", "type": "occupation", "gender": "male"},
    {"attribute": "happy", "type": "emotion", "gender": "female"},
    {"attribute": "fast", "type": "speaking rate", "gender": "male"},
    {"attribute": "high", "type": "pitch", "gender": "female"}
]

# 16 种口音，用于 accent 风格 prompt
accents = [
    "african", "australian", "bermuda", "canadian", "english", "hongkong",
    "indian", "irish", "malaysian", "newzealand", "philippines", "scottish",
    "singaporean", "southatlantic", "us", "welsh"
]

# 输出路径设置
output_dir = "generated_speech_expanded_v2"
os.makedirs(output_dir, exist_ok=True)

# 设置语音生成参数
generation_config = {
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 50
}

# ------------------------------------
# 遍历所有属性，生成带性别风格的语音
# ------------------------------------
for attr in attributes:
    gender_label = attr['gender']
    style_prompt = f"Speak in a manner that reflects someone who is {attr['attribute']}, typically associated with a {gender_label} voice."

    style_dir = os.path.join(output_dir, f"gender_{attr['attribute']}")
    os.makedirs(style_dir, exist_ok=True)

    for content_idx, content_prompt in enumerate(content_prompts):
        # 编码输入文本
        input_ids = tokenizer(style_prompt, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(content_prompt, return_tensors="pt").input_ids.to(device)

        # 生成语音
        generation = model.generate(
            input_ids=input_ids,
            prompt_input_ids=prompt_input_ids,
            **generation_config
        )

        # 保存语音为 wav 文件
        audio_arr = generation.cpu().numpy().squeeze()
        filename = f"{attr['attribute']}_gender_content{content_idx+1}.wav"
        filepath = os.path.join(style_dir, filename)
        sf.write(filepath, audio_arr, model.config.sampling_rate)
        print(f"Generated (gender): {filepath}")

# ------------------------------------
# 遍历所有属性，结合16种口音生成带口音风格的语音
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
