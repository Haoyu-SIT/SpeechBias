# SpeechBias

This project investigates bias in speech synthesis and recognition systems.

We have **completed the initial implementation** of both **speech synthesis** and **bias detection** pipelines.

### Components

- `parler-tts-demo/`: Reproduction of the original [Parler-TTS](https://huggingface.co/parler-tts/parler_tts) model as described in the bias paper.
- `parler-tts-new/`: Extended version that supports style control based on single attributes (e.g., occupation, emotion, pitch, speaking rate), including **accent-specific** synthesis prompts.
- `GenderDection/`: Codebase for detecting **gender bias** in generated speech.
- `AccentDectection/`: Codebase for detecting **accent bias** based on classification outputs.

More detailed documentation and usage examples will be added in future updates.

CosyVoice2 need to download a third party tts to rush with link below
https://github.com/shivammehta25/Matcha-TTS/tree/dd9105b34bf2be2230f4aa1e4769fb586a3c824e

# CosyVoice2 Style Prompt-based Speech Synthesis

The file CosyVoice2 demonstrates how to generate stylized speech using [CosyVoice2](https://github.com/FunAudioLLM/CosyVoice) by combining:

- **Text-based style prompts** (e.g., "A female happy voice with American accent")
- **Reference audio** (prompt_speech_16k) to mimic prosody and tone

- 
## 🚀 How It Works

This script performs **three types of generation** for each target style:

1. **Instruct2 Mode**: Text prompt controls style
2. **Cross-lingual Mode**: Allows expressions like `[laughter]`
3. **Zero-shot Mode**: Multi-sentence input via a generator

Each voice style is defined by a `style_prompt` and a reference speech clip.

## ✍️ Customize

Edit the `attributes` list in the script to add your own combinations of:
- `type`: "emotion", "occupation", etc.
- `value`: e.g., "angry", "doctor"
- `gender`: "male", "female"
- `accent`: "american", "singaporean", etc.

You can also modify `content_prompts` for different speech generation targets.

## 📝 Output

All generated WAV files will be saved in the `cosyvoice_outputs_prompt_only/` directory, categorized by style.

## 📣 Note

- This demo does **not** rely on `add_zero_shot_spk()` or `spk` arguments that may not exist in all versions.
- CosyVoice2 must be locally installed and accessible via `cosyvoice.cli.cosyvoice.CosyVoice2`.


