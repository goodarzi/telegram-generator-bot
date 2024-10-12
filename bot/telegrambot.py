import os, logging
from typing import Union
from signal import SIGINT, SIGUSR1, SIGTERM, SIGQUIT
from telethon import TelegramClient, types
from telethon.hints import MarkupLike
import os, re
from telethon.types import MessageEntityBotCommand, Document, Photo
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

dir_path = os.path.dirname(os.path.realpath(__file__))


def _download_path():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    downloads_dir = os.path.join(dir_path, "downloads")
    if not os.path.exists(downloads_dir):
        os.mkdir(downloads_dir)
    return downloads_dir


class TelegramBot(TelegramClient):

    def __init__(self, config):

        session = os.path.basename(__file__).split(".")[0]
        session_path = os.path.join(dir_path, session)

        if not "api_id" in config:
            raise ValueError
        api_id = config["api_id"]
        if not "api_hash" in config:
            raise ValueError
        api_hash = config["api_hash"]
        if not "bot_token" in config:
            raise ValueError
        bot_token = config["bot_token"]

        proxy = config["proxy"] if "proxy" in config else None

        self.allow_chats = config["allow_chats"] if "allow_chats" in config else []
        log_level = config["log_level"] if "log_level" in config else 2

        retry_delay = config["retry_delay"] if "retry_delay" in config else 1

        logging.basicConfig()
        self.logger = logging.getLogger(os.path.basename(__file__)).getChild(
            __class__.__name__
        )
        self.logger.setLevel(log_level)
        super().__init__(
            session=session_path,
            api_id=api_id,
            api_hash=api_hash,
            proxy=proxy,
            retry_delay=retry_delay,
            base_logger=self.logger,
        )

        self.loop.add_signal_handler(SIGINT, self.sigint_handler)
        self.loop.add_signal_handler(SIGTERM, self.sigint_handler)
        self.loop.add_signal_handler(SIGQUIT, self.sigint_handler)

        self.add_event_handler(
            self.respond_not_allowed,
            NewMessage(chats=self.allow_chats, blacklist_chats=True, incoming=True),
        )

        self.download_path = _download_path()

    def sigint_handler(self):
        self.logger.warning("SIGINT or CTRL-C detected. Exiting gracefully")
        self.disconnect()
        exit(0)

    @staticmethod
    async def respond_not_allowed(event: NewMessage.Event):
        await event.respond(str(event.chat_id))

    @staticmethod
    def get_bot_command(message: Message):
        commands = []
        for c in message.get_entities_text(MessageEntityBotCommand):
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
    def media_id(message: Message):
        if not message.file:
            return None
        return message.media.photo.id if message.photo else message.media.document.id

    async def download_image(self, message: Message, path: str = None):
        media_id = TelegramBot.media_id(message)
        if not path:
            path = self.download_path
        if not media_id:
            return None
        file_name = str(media_id) + message.file.ext
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            if os.path.getsize(file_path) == message.file.size:
                return file_path
        return await message.download_media(file=file_path)

    @staticmethod
    def button_inline_list(array_list: list):
        if len(array_list) > 98:
            result = []
            for i in range(0, 96, 2):
                result.append(
                    [Button.inline(array_list[i]), Button.inline(array_list[i + 1])]
                )
            return result
        return [[Button.inline(i)] for i in array_list]

    @staticmethod
    def media_is_png(event):
        if event.file:
            print(f"{event.file.mime_type=}")
            if isinstance(event.file.media, Document):
                if event.file.mime_type == "image/png":
                    return True

    @staticmethod
    def media_is_photo(event):
        if event.file:
            if isinstance(event.file.media, Photo):
                return True

    @staticmethod
    def message_head(message):
        pattern = re.compile("/?([\w\s]+):")
        current_menu = re.search(pattern, message)
        if current_menu:
            return current_menu[1]

    @staticmethod
    def event_not_reply(event: Union[NewMessage.Event, MessageEdited.Event]):
        return not event.message.is_reply
