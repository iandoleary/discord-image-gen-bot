import discord
import random
import os
import io
import aiohttp
import openai
import yt_dlp
import asyncio

# If you want to declare where to store model data:
os.environ["HF_HOME"] = "E:\\huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "E:\\huggingface_cache"
os.environ["DIFFUSERS_CACHE"] = "E:\\huggingface_cache"

import torch
from huggingface_hub import login
from discord.ext import commands
from dotenv import load_dotenv
from pathlib import Path
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline
from transformers import T5EncoderModel

env_path = Path(".").parent / ".env"
load_dotenv(env_path)

FFMPEG_OPTIONS = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}
FFMPEG_PATH = "C:/ffmpeg/ffmpeg.exe"
os.environ["FFMPEG_EXECUTABLE"] = FFMPEG_PATH

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

login(token=HF_TOKEN)

openai.api_key = OPENAI_API_KEY

# Discord bot intents
intents = discord.Intents.default()
intents.members = True
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"
# This is the other model I tried, switching would require rework at 'pipe = StableDiff...'
# model_id = "black-forest-labs/FLUX.1-dev"


# ML model device
device = "cuda" if torch.cuda.is_available() else "cpu"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.float16
)

t5_nf4 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.float16)

# Init PyTorch diffusion pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    transformer=model_nf4,
    text_encoder_3=t5_nf4,
    torch_dtype=torch.float16 #bfloat16 preferred
    ).to(device)
pipe.enable_model_cpu_offload()

# OPTIONAL - Helps run large models when you're limited on VRAM (<16gb)
# pipe.vae.enable_tiling()
# pipe.vae.enable_slicing()
# pipe.enable_sequential_cpu_offload()


queue = {}


@bot.event
async def on_ready():
    print(f"We have logged in as {bot.user}")
    guild_count = 0
    for guild in bot.guilds:
        print(f"- {guild.id} (name: {guild.name})")
        guild_count = guild_count + 1

    print(f"Bot is in " + str(guild_count) + " guild(s).")


@bot.command()
async def img_openai(ctx, *, arg):
    await ctx.send("Generating image with OpenAI dall-e-3 for prompt: " + arg)
    
    response = openai.images.generate(
        model="dall-e-3", prompt=arg, n=1, size="1024x1024"
    )

    img_url = response.data[0].url

    async with aiohttp.ClientSession() as session:
        async with session.get(img_url) as resp:
            if resp.status == 200:
                img_data = await resp.read()
                file = discord.File(fp=io.BytesIO(img_data), filename="image.png")
                await ctx.send(file=file)


@bot.command()
async def img_sd(ctx, *, arg):
    await ctx.send("Generating image with Stable Diffusion for prompt: " + arg)
    # Generate a random seed for the generator
    seed = random.randint(0, 2**32 - 1)

    img = await asyncio.to_thread(
        pipe,
        arg,
        height=1024,
        width=1024,
        guidance_scale=3.5, # 3.5 default (1-20)
        num_inference_steps=30, # 50 default - lowering can provide cool results
        max_sequence_length=512, # 512 default
        generator=torch.Generator(device).manual_seed(seed)
    )

    image_bytes = io.BytesIO()
    img.images[0].save(image_bytes, format="PNG")
    image_bytes.seek(0)

    file = discord.File(fp=image_bytes, filename="image.png")
    await ctx.send(file=file)

# Plays music from a YouTube search
@bot.command()
async def play(ctx, *, query: str):
    if ctx.guild.id not in queue:
        queue[ctx.guild.id] = []

    if not ctx.voice_client:
        if ctx.author.voice:
            await ctx.author.voice.channel.connect()
        else:
            await ctx.send("You need to be in a voice channel to play music.")
            return

    if not query.startswith("http"):
        with yt_dlp.YoutubeDL(
            {"format": "bestaudio", "quiet": True, "default_search": "ytsearch"}
        ) as ydl:
            info = ydl.extract_info(query, download=False)
            if "entries" in info:
                info = info["entries"][0]

    with yt_dlp.YoutubeDL({"format": "bestaudio"}) as ydl:
        info = ydl.extract_info(info["url"], download=False)
        audio_url = info["url"]

    if not ctx.voice_client.is_playing():
        ctx.voice_client.play(
            discord.FFmpegPCMAudio(audio_url, **FFMPEG_OPTIONS),
            after=lambda e: play_next(ctx),
        )
        await ctx.send(f"Now playing: {info['title']}")
    else:
        queue[ctx.guild.id].append(audio_url)
        await ctx.send(f"Added to queue: {info['title']}")


def play_next(ctx):
    guild_id = ctx.guild.id
    if queue[guild_id]:
        next_song = queue[guild_id].pop(0)
        ctx.voice_client.play(
            discord.FFmpegPCMAudio(next_song, **FFMPEG_OPTIONS),
            after=lambda e: play_next(ctx),
        )
    else:
        asyncio.run_coroutine_threadsafe(ctx.voice_client.disconnect(), bot.loop)


@bot.command()
async def skip(ctx):
    if ctx.voice_client.is_playing():
        ctx.voice_client.stop()
        await ctx.send("Skipped ⏭")
    else:
        await ctx.send("No song is currently playing.")


@bot.command()
async def stop(ctx):
    if ctx.voice_client:
        queue[ctx.guild.id] = []
        ctx.voice_client.stop()
        await ctx.send("Music stopped. Queue cleared.")
        await ctx.voice_client.disconnect()
    else:
        await ctx.send("I'm not playing any music.")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    #Test message
    msg = message.content
    if msg.startswith("hello"):
        await message.channel.send("hello")
        print("Message sent in channel: " + str(message.channel.name))

    await bot.process_commands(message)


bot.run(DISCORD_TOKEN)