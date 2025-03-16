# discord-image-gen-bot
Discord bot
## Prerequisites
- Install Nvidia CUDA
- Create a Discord bot through https://discord.com/developers
- Set env vars for Discord API in your .env:
```bash
DISCORD_APP_ID=''
DISCORD_PUBLIC_KEY=''
DISCORD_BOT_TOKEN=''
```
- Retrieve a huggingface token through https://huggingface.co/
- For FLUX.1, you will need to request access to the model's repository - takes about 30 seconds if you have an account.
```bash
HF_TOKEN=''
```
- If you want to generate images through the OpenAI API, you will need an API key:
```bash
OPENAI_API_KEY=''
```
## Recommended setup
### Create a virtual Environment:
```bash
python -m venv venv
```
### Install required python modules:
```bash
pip install -r requirements.txt
```
## Start Bot
### Activate venv:
```bash
venv\Scripts\activate
```
### Run the bot:
```bash
python app.py
```
### Freeze pip packages:
```bash
pip freeze > requirements.txt
```
### Deactivate venv:
```bash
deactivate
```
## Git Commands:
```bash
git status
git add .
git commit -m "message"
git push
```
## Developer Notes:

GUIDANCE SCALE: In Stable Diffusion, "guidance scale" refers to a parameter that controls how closely the generated image adheres to the text prompt provided, essentially dictating how strictly the AI model should follow the instructions given; a higher guidance scale means the image will more closely resemble the prompt, while a lower scale allows for more creative interpretation and variation in the output. - Google Search AI

INFERENCE STEPS: num_inference_steps ( int , optional, defaults to 50) — The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference. guidance_scale ( float , optional, defaults to 7.5) — Guidance scale as defined in Classifier-Free Diffusion Guidance. -HuggingFace

SEQUENCE LENGTH: In Stable Diffusion, "MAX_sequence_length" refers to the maximum number of tokens (words or parts of words) that a text prompt can be before it gets truncated when feeding it into the model; essentially, it sets a limit on how long a textual description can be for generating an image, with any text exceeding this length being cut off. - Google Search AI