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
