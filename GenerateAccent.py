import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os

# 设置设备
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载模型和分词器
model_name = "parler-tts/parler-tts-mini-v1"
model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置要测试的口音名称（可以修改为其他任意你想测试的）
accent_name = "Singaporean"  # 可替换为：American, Irish, Singaporean 等

# 构造风格 prompt（显式写出口音名称）
style_prompt = f"Speak with a clear and noticeable {accent_name} accent, like a native speaker from {accent_name}."

# 内容 prompt
content_prompt = "I just finished reading the book you recommended. It was absolutely brilliant."

# Tokenize prompts
input_ids = tokenizer(style_prompt, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(content_prompt, return_tensors="pt").input_ids.to(device)

# 生成语音
generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_input_ids,
    temperature=1.0,
    top_p=0.9,
    top_k=50
)

# 输出语音文件
output_dir = "test_accent_output"
os.makedirs(output_dir, exist_ok=True)
filename = f"test_{accent_name.lower()}.wav"
filepath = os.path.join(output_dir, filename)

audio_arr = generation.cpu().numpy().squeeze()
sf.write(filepath, audio_arr, model.config.sampling_rate)

print(f"Generated speech with {accent_name} accent saved to: {filepath}")
