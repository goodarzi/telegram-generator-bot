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
