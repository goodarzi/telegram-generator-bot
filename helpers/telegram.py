import os
from telethon.custom import (
    MessageButton,
    Forward,
    Message,
    Button,
    InlineBuilder,
    InlineResult,
    InlineResults,
    Conversation,
)

from telethon import TelegramClient, hints
from telethon.hints import MarkupLike

# TypeMessageEntity = Union[MessageEntityUnknown,MessageEntityMention,MessageEntityHashtag,MessageEntityBotCommand,MessageEntityUrl,MessageEntityEmail,MessageEntityBold,MessageEntityItalic,MessageEntityCode,MessageEntityPre,MessageEntityTextUrl,MessageEntityMentionName,InputMessageEntityMentionName,MessageEntityPhone,MessageEntityCashtag,MessageEntityUnderline,MessageEntityStrike,MessageEntityBankCard,MessageEntitySpoiler,MessageEntityCustomEmoji,MessageEntityBlockquote]


# message_button = MessageButton()
# client = TelegramClient
# client.send_message()
# client.parse_mode
# client.build_reply_markup
# client.edit_message()


def file_id(message: Message):
    if not message.file:
        return None
    file = {}
    file["ext"] = message.file.ext
    if message.photo:
        file["id"] = message.media.photo.id
        file["name"] = str(file["id"])
        file["base_name"] = file["name"] + file["ext"]
    else:
        file["id"] = message.media.document.id
        file["name"], ext = os.path.splitext(message.file.name)
        file["base_name"] = file["name"] + "_" + str(file["id"]) + file["ext"]
    return file


async def download_image(message: Message, path: str):
    file = file_id(message)
    if not file:
        return
    file_path = os.path.join(path, file["base_name"])
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == message.file.size:
            return file_path
    return await message.download_media(file=file_path)


def button_inline_list(array_list: list):
    inline_list = []
    for i in array_list:
        inline_list.append([Button.inline(i)])
    return inline_list


def buttons_to_list(buttons):
    button_list = []
    if isinstance(buttons, list):
        for row in buttons:
            if isinstance(row, list):
                for button in row:
                    button_list.append(button.text)
            else:
                button_list.append(row)
    else:
        button_list.append(buttons)
    return button_list


def build_respond(response):
    if isinstance(response, list):
        return button_inline_list(response)
