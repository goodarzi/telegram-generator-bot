import os, logging
from typing import Union
from signal import SIGINT, SIGUSR1, SIGTERM, SIGQUIT
from telethon import TelegramClient, types
from telethon.hints import MarkupLike
import os
from telethon.events import NewMessage, MessageEdited, CallbackQuery
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


class TelegramBot(TelegramClient):

    def __init__(
        self,
        session_path: str,
        api_id: int,
        api_hash: str,
        allowed_chat_ids: list,
        log_level: Union[int, str] = "ERROR",
        proxy: dict = None,
        retry_delay=1,
    ):

        self.allowed_chat_ids = allowed_chat_ids

        logging.basicConfig()
        self.logger = logging.getLogger(os.path.basename(__file__)).getChild(
            __class__.__name__
        )
        self.logger.setLevel(log_level)
        super().__init__(
            session_path,
            api_id,
            api_hash,
            proxy=proxy,
            retry_delay=retry_delay,
            base_logger=self.logger,
        )

        self.loop.add_signal_handler(SIGINT, self.sigint_handler)
        self.loop.add_signal_handler(SIGTERM, self.sigint_handler)
        self.loop.add_signal_handler(SIGQUIT, self.sigint_handler)

        self.add_event_handler(
            self.respond_not_allowed,
            NewMessage(
                chats=self.allowed_chat_ids, blacklist_chats=True, incoming=True
            ),
        )

    def sigint_handler(self):
        self.logger.warning("SIGINT or CTRL-C detected. Exiting gracefully")
        self.disconnect()
        exit(0)

    async def respond_not_allowed(self, event: NewMessage.Event):
        await event.respond(str(event.chat_id))

    @staticmethod
    def get_bot_command(message: Message):
        commands = []
        for c in message.get_entities_text(types.MessageEntityBotCommand):
            commands.append(c)
            # print(ent, ent.length, txt,len(message.text))
        if not commands:
            return None

        if len(commands[0][1]) == len(message.text):
            return (commands[0][1], "")
        else:
            cmd_args = []
            for index, command in enumerate(commands):
                cmd = command[1]
                args_offset = int(command[0].offset) + int(command[0].length)
                if len(message.text) > args_offset:
                    if len(commands) > index + 1:
                        args = (
                            message.text[args_offset : commands[index + 1][0].offset]
                            .lstrip()
                            .rstrip()
                        )
                    else:
                        args = message.text[args_offset:].lstrip().rstrip()
                else:
                    args = None
                cmd_args.append((cmd, args))
            return cmd_args

    @staticmethod
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

    @staticmethod
    async def download_image(message: Message, path: str):
        file = TelegramBot.file_id(message)
        if not file:
            return
        file_path = os.path.join(path, file["base_name"])
        if os.path.exists(file_path):
            if os.path.getsize(file_path) == message.file.size:
                return file_path
        return await message.download_media(file=file_path)

    @staticmethod
    def button_inline_list(array_list: list):
        inline_list = []
        for i in array_list:
            inline_list.append([Button.inline(i)])
        return inline_list

    @staticmethod
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

    @staticmethod
    def build_respond(response):
        if isinstance(response, list):
            return TelegramBot.button_inline_list(response)

    @staticmethod
    def media_is_png(event):
        if event.file:
            print(f"{event.file.mime_type=}")
            if isinstance(event.file.media, types.Document):
                if event.file.mime_type == "image/png":
                    return True

    @staticmethod
    def media_is_photo(event):
        if event.file:
            if isinstance(event.file.media, types.Photo):
                return True
