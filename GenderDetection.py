import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch.nn.functional as F

# 模型加载 (论文中明确提到的wav2vec2模型)
model_name = "audeering/wav2vec2-large-robust-24-ft-age-gender"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 加载模型和特征提取器
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(device)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)


# 性别识别函数
def predict_gender(audio_file):
    # 加载音频文件
    waveform, sample_rate = torchaudio.load(audio_file)

    # 转换为模型所需的输入格式 (单声道、16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    waveform = waveform.mean(dim=0, keepdim=True)  # 转单声道

    # 提取特征
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").to(device)

    # 模型推理
    with torch.no_grad():
        logits = model(**inputs).logits

    # 计算概率并确定性别
    probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze()

    labels = model.config.id2label
    predicted_label = labels[probs.argmax()]
    confidence = probs.max()

    return predicted_label, confidence


# 示例用法
audio_path = "/home/asian/Projects/AudioSynthesisBias/parler_tts_out.wav"
gender, confidence = predict_gender(audio_path)

print(f"Predicted Gender: {gender} with confidence {confidence:.2f}")
