import os, logging
import typing
from typing import Union
from signal import SIGINT, SIGUSR1, SIGTERM, SIGQUIT

from telethon import TelegramClient

from telethon import  functions, types, errors, utils, custom, events
from telethon.events import MessageEdited
from telethon.custom import Message
from telethon.custom import Button
from telethon.events import NewMessage

class TelegramBot(TelegramClient):

    def __init__(self,
                 session_path: str,
                 api_id: int,
                 api_hash: str,
                 allowed_chat_ids: list,
                 log_level: Union[int, str]='ERROR',
                 proxy: dict=None,
                 retry_delay = 1
                 ):

        self.allowed_chat_ids = allowed_chat_ids

        logging.basicConfig()
        self.logger = logging.getLogger(os.path.basename(__file__)).getChild(__class__.__name__)
        self.logger.setLevel(log_level)
        super().__init__(session_path, api_id, api_hash,
            proxy=proxy, retry_delay=retry_delay, base_logger=self.logger
        )

        self.loop.add_signal_handler(SIGINT, self.sigint_handler)
        self.loop.add_signal_handler(SIGTERM, self.sigint_handler)
        self.loop.add_signal_handler(SIGQUIT, self.sigint_handler)

        self.add_event_handler(self.respond_not_allowed, 
            events.NewMessage(chats=self.allowed_chat_ids,
                              blacklist_chats=True,incoming=True))
        
    def sigint_handler(self):
        self.logger.warning('SIGINT or CTRL-C detected. Exiting gracefully')
        self.disconnect()
        exit(0)
        

    async def respond_not_allowed(self, event: events.NewMessage):
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
            return (commands[0][1], '')
        else:
            cmd_args = []
            for index, command in enumerate(commands):
                cmd = command[1]
                args_offset = int(command[0].offset) + int(command[0].length)
                if len(message.text) > args_offset:
                    if len(commands) > index+1:
                        args = message.text[args_offset:commands[index+1][0].offset].lstrip().rstrip()
                    else:
                        args = message.text[args_offset:].lstrip().rstrip()
                else:
                    args = None
                cmd_args.append((cmd, args))
            return cmd_args


