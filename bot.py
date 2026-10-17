
import os
import logging
import asyncio

from dotenv import load_dotenv
from google import genai
from google.genai import types

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN or not GEMINI_API_KEY:
    raise RuntimeError("Missing required environment variables.")

client = genai.Client(api_key=GEMINI_API_KEY)

# If this model is unavailable for your API key,
# we can change it later.
MODEL = "gemini-2.5-flash"

PLATFORMS = {
    "universal": "Universal AI",
    "gemini": "Google Gemini",
    "chatgpt": "ChatGPT Images",
    "midjourney": "Midjourney",
    "flux": "FLUX",
}

BASE_PROMPT = """
Analyze the reference image like an expert visual artist,
photographer and image-prompt engineer.

Write a detailed standalone image-generation prompt that
recreates the visible image as closely as possible.

Describe:
- Subject, appearance, pose, expression and clothing
- Composition, framing, camera angle and perspective
- Lighting direction, shadows and highlights
- Colors, contrast, textures and materials
- Background, atmosphere and depth of field
- Art style or photographic style
- Visible text and typography, if readable
- Approximate aspect ratio

Be specific. Preserve distinctive visual details.
Do not invent invisible details or claim to know the
original prompt, model, camera or generation settings.

Write a prompt suitable for the selected AI platform.
Return the prompt only, without extra commentary.
"""

GUIDANCE = {
    "universal": """
Use detailed natural language compatible with many image generators.
""",
    "gemini": """
Optimize for Gemini image generation using clear natural language.
Describe layout, spatial relationships, lighting and readable text.
""",
    "chatgpt": """
Optimize for ChatGPT image generation. Clearly describe the scene,
visual hierarchy, composition, materials and any visible text.
""",
    "midjourney": """
Make a descriptive Midjourney prompt. Put the subject and composition
first, then lighting, colors and style. Add an appropriate --ar
parameter only when the image proportions support it.
""",
    "flux": """
Optimize for FLUX using concrete visual descriptions, spatial
relationships, materials, lighting and texture.
""",
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "👋 Welcome to Image Prompt Bot!\n\n"
        "Send me an image and I'll create a detailed prompt "
        "for your chosen AI image generator.\n\n"
        "Supported: Universal, Gemini, ChatGPT, Midjourney and FLUX.\n"
        "Use /help for instructions."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.effective_message.reply_text(
        "1. Send a photo or image file.\n"
        "2. Choose your target AI.\n"
        "3. Copy the generated prompt into that AI.\n\n"
        "Send another image to analyze a new reference."
    )


async def receive_image(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    message = update.effective_message

    try:
        if message.photo:
            file = await message.photo[-1].get_file()
            image_bytes = bytes(await file.download_as_bytearray())
            mime_type = "image/jpeg"

        elif message.document:
            document = message.document
            mime_type = document.mime_type or ""

            if mime_type not in (
                "image/jpeg", "image/png", "image/webp"
            ):
                await message.reply_text(
                    "Please send a JPG, PNG or WebP image."
                )
                return

            if document.file_size and document.file_size > 15 * 1024 * 1024:
                await message.reply_text(
                    "Please send an image smaller than 15 MB."
                )
                return

            file = await document.get_file()
            image_bytes = bytes(await file.download_as_bytearray())

        else:
            return

        if len(image_bytes) > 15 * 1024 * 1024:
            await message.reply_text(
                "Please send an image smaller than 15 MB."
            )
            return

        context.user_data["image_bytes"] = image_bytes
        context.user_data["image_mime"] = mime_type

        keyboard = [
            [
                InlineKeyboardButton(
                    "🌐 Universal", callback_data="p:universal"
                ),
                InlineKeyboardButton(
                    "✨ Gemini", callback_data="p:gemini"
                ),
            ],
            [
                InlineKeyboardButton(
                    "🧠 ChatGPT", callback_data="p:chatgpt"
                ),
                InlineKeyboardButton(
                    "🎨 Midjourney", callback_data="p:midjourney"
                ),
            ],
            [
                InlineKeyboardButton(
                    "⚡ FLUX", callback_data="p:flux"
                ),
            ],
        ]

        await message.reply_text(
            "✅ Image received! Choose your target AI:",
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    except Exception:
        logger.exception("Image processing failed")
        await message.reply_text(
            "I couldn't read that image. Please try again."
        )


def make_prompt(image_bytes, mime_type, platform):
    instruction = (
        BASE_PROMPT
        + "\nTARGET AI: " + PLATFORMS[platform]
        + "\n" + GUIDANCE[platform]
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            ),
            instruction,
        ],
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=1400,
        ),
    )

    if not response.text:
        raise RuntimeError("Empty response from Gemini.")

    return response.text.strip()


async def choose_platform(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    query = update.callback_query
    await query.answer()

    platform = query.data.split(":", 1)[1]

    image_bytes = context.user_data.get("image_bytes")
    mime_type = context.user_data.get("image_mime")

    if platform not in PLATFORMS or not image_bytes or not mime_type:
        await query.message.reply_text(
            "Please send your image again, then choose an AI."
        )
        return

    await query.edit_message_text(
        "🔍 Analyzing your image for "
        + PLATFORMS[platform] + "..."
    )

    try:
        prompt = await asyncio.to_thread(
            make_prompt, image_bytes, mime_type, platform
        )

        if len(prompt) > 3900:
            prompt = prompt[:3900]

        await query.message.reply_text(
            "🎯 YOUR IMAGE PROMPT\n\n"
            + prompt
            + "\n\nCopy this prompt into your chosen image generator."
        )

    except Exception:
        logger.exception("Gemini prompt generation failed")
        await query.message.reply_text(
            "Sorry, generation failed. Please check the Gemini "
            "API key, model availability and API quota in Railway."
        )


async def unknown_text(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    await update.effective_message.reply_text(
        "Please send an image to get started."
    )


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))

    app.add_handler(
        MessageHandler(
            filters.PHOTO | filters.Document.IMAGE,
            receive_image,
        )
    )

    app.add_handler(
        CallbackQueryHandler(choose_platform, pattern=r"^p:")
    )

    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, unknown_text)
    )

    print("Bot is running!")
    app.run_polling()


if __name__ == "__main__":
    main()
  
