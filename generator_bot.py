import os
import logging
import asyncio
import base64
import re
import uuid
import json
from http import HTTPStatus
from httpx import ConnectError, BasicAuth
from typing import Union
from .generators.webui import WebuiClient
from .helpers import utils
from .helpers.telegram import button_inline_list, Button

from .telegrambot import TelegramBot
from telethon import events, errors

dir_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(dir_path, "config.yaml")
if config_file:
    config = utils.load_config(config_file)
else:
    config = {}

output_dir = os.path.join(dir_path, config["extensions"]["image_generator_output_dir"])


def get_image_generator(event):

    if config["extensions"]["image_generator"] == "webui":
        auth = BasicAuth(username="user1", password="pass1")
        return WebuiClient(
            base_url=config["extensions"]["webui"]["base_url"],
            chat_id=event.chat_id,
            out_dir=output_dir,
            auth=auth,
        )


def message_to_prompt(message: str):
    msg_split = message.split("\n")
    print(msg_split)
    negative_prompt: str = None
    if len(msg_split) > 1:
        prompt = msg_split[1].strip("`")
        if msg_split[1].lower().startswith("negative prompt:"):
            negative_prompt = msg_split[1][16:]
        elif msg_split[1].lower().startswith("negative:"):
            negative_prompt = msg_split[1][9:]
        if negative_prompt:
            return (prompt, negative_prompt)
        else:
            return (prompt, "")
    else:
        return (message, "")


def message_head(message):
    pattern = re.compile("/?([\w\s]+):")
    current_menu = re.search(pattern, message)
    if current_menu:
        return current_menu[1]


def save_live_image(image: str):
    file_path = "/tmp/generation-bot-thumb.png"
    with open(file_path, "wb") as output:
        output.write(base64.b64decode(image))
    return file_path


async def update_progress(generator_client, event, id_task):
    if event.pattern_match:
        status_text = f"generating...\n`{event.pattern_match[1]}` \n"
    else:
        status_text = "**generating...** \n"
    picframe = os.path.join(dir_path, "helpers/folder-adwaita-pictures.svg")
    status = await event.respond(
        message=status_text, file=picframe, reply_to=event._message_id
    )

    # async for s in generator_client.task_progress(id_task):
    async for s in generator_client.progress(id_task):

        text = f"{status_text}\n"
        text += f"\n📥 : [{utils.create_progress_bar(s[0]*100)}]"
        try:
            if s[1]:
                await status.edit(text=text, file=save_live_image(s[1]))
            else:
                await status.edit(text=text)
        except errors.MessageNotModifiedError as e:
            print(e)
    return status


async def generate_txt2img(
    event: Union[events.NewMessage.Event, events.MessageEdited.Event], message=None
):
    generator_client = get_image_generator(event)
    payload = generator_client.txt2img_payload
    payload.force_task_id = uuid.uuid4().hex
    if message:
        payload.prompt, payload.negative_prompt = message_to_prompt(message)
    elif event.pattern_match:
        print(event.pattern_match[1])
        payload.prompt = event.pattern_match[1]
    else:
        payload.prompt = event.message.text
    generate = asyncio.create_task(generator_client.txt2img(payload))
    progress = asyncio.create_task(
        update_progress(generator_client, event, payload.force_task_id)
    )
    await generate
    await progress
    info = json.loads(generate.result().info)

    # print(infotext)
    progress_text = progress.result().message
    # print(progress_text)
    for index, image in enumerate(generate.result().images):

        infotext = f"Steps: {info['steps']}, Sampler: {info['sampler_name']}, CFG scale: {info['cfg_scale']}, Seed: {info['all_seeds']}, Size: {info['height']}x{info['width']}, Model: {info['sd_model_name']}, Clip skip: {info['clip_skip']}, "
        if len(info["all_negative_prompts"]) >= index + 1:
            if info["all_negative_prompts"][index]:
                infotext = (
                    f"Negative prompt: {info['all_negative_prompts'][index]}\n"
                    + infotext
                )
        if len(info["all_prompts"]) >= index + 1:
            if info["all_prompts"][index]:
                infotext = (
                    f"**txt2img:** \n`{info['all_prompts'][index]}` \n" + infotext
                )
        if "ADetailer model" in info["extra_generation_params"].keys():
            infotext += f"ADetailer model: {info['extra_generation_params']['ADetailer model']}, "
        if len(infotext) > 4096:
            infotext = infotext[:4096]

        img_file = await event.client.upload_file(
            base64.b64decode(image),
            file_name=f"txt2img-{utils.timestamp()}-{index}.png",
        )
        await progress.result().edit(
            file=img_file,
            # reply_to = event._message_id,
            # caption = self.payload['prompt'] if self.payload["prompt"] else infotexts[index],
            text=f"{infotext}",
            force_document=False,
            buttons=[Button.inline("Regen"), Button.inline("File")],
        )


async def check_reply(
    event: Union[events.NewMessage.Event, events.MessageEdited.Event]
):
    reply_message = await event.message.get_reply_message()
    if reply_message.photo:
        await generate_img2img(event, reply_message)


async def generate_img2img(
    event: Union[events.NewMessage.Event, events.MessageEdited.Event],
    message=None,
    in_memory=False,
):

    generator_client = get_image_generator(event)
    payload = generator_client.img2img_payload
    payload.force_task_id = uuid.uuid4().hex

    if isinstance(event, events.CallbackQuery.Event):
        msg = await event.get_message()
        msg_txt = msg.text
        img = await msg.download_media()
    else:
        msg_txt = event.message.text
        if message:
            img = await message.download_media()
        else:
            img = await event.message.download_media()

    payload.prompt, payload.negative_prompt = message_to_prompt(msg_txt)

    with open(img, "rb") as file:
        b64img = base64.b64encode(file.read()).decode("utf-8")
    init_images = [
        b64img,
    ]

    # if in_memory:
    #     bytes_arr = bytearray()
    #     async for chunk in event.client.iter_download(event.photo):
    #         bytes_arr.extend(chunk)
    #     init_images = [
    #         base64.b64encode(bytes_arr).decode("utf-8"),
    #     ]
    # else:
    #     img = await event.message.download_media()

    #     with open(img, "rb") as file:
    #         b64img = base64.b64encode(file.read()).decode("utf-8")
    #     init_images = [
    #         b64img,
    #     ]
    payload.init_images = init_images

    generate = asyncio.create_task(generator_client.img2img(payload))
    progress = asyncio.create_task(
        update_progress(generator_client, event, payload.force_task_id)
    )
    await generate
    await progress
    info = json.loads(generate.result().info)
    # print(infotext)
    progress_text = progress.result().message
    # print(progress_text)
    for index, image in enumerate(generate.result().images):
        infotext = f"Steps: {info['steps']}, Sampler: {info['sampler_name']}, CFG scale: {info['cfg_scale']}, Seed: {info['all_seeds']}, Size: {info['height']}x{info['width']}, Model: {info['sd_model_name']}, Clip skip: {info['clip_skip']}, "
        infotext += f"Denoising strength: {info['denoising_strength']}, "
        if len(info["all_negative_prompts"]) >= index + 1:
            if info["all_negative_prompts"][index]:
                infotext = (
                    f"Negative prompt: {info['all_negative_prompts'][index]}\n"
                    + infotext
                )
        if len(info["all_prompts"]) >= index + 1:
            if info["all_prompts"][index]:
                infotext = (
                    f"**img2img:** \n`{info['all_prompts'][index]}` \n" + infotext
                )
        if "ADetailer model" in info["extra_generation_params"].keys():
            infotext += f"ADetailer model: {info['extra_generation_params']['ADetailer model']}, "
        if len(infotext) > 4096:
            infotext = infotext[:4096]

        img_file = await event.client.upload_file(
            base64.b64decode(image),
            file_name=f"txt2img-{utils.timestamp()}-{index}.png",
        )
        await progress.result().edit(
            file=img_file,
            # reply_to = event._message_id,
            # caption = self.payload['prompt'] if self.payload["prompt"] else infotexts[index],
            text=f"{infotext}",
            force_document=False,
            buttons=[Button.inline("Regen"), Button.inline("File")],
        )


async def memory(event: events.NewMessage.Event):
    generator_client = get_image_generator(event)
    result = generator_client.memory()
    await event.respond(str(result))


async def model(event: events.NewMessage.Event):
    generator_client = get_image_generator(event)
    result = await generator_client.model()
    buttons = button_inline_list(result)
    buttons.append(
        [
            Button.inline("Refresh"),
            Button.inline("Reload"),
            Button.inline("Unload"),
            Button.inline("Current Model"),
        ]
    )
    await event.respond(message="Select Model:", buttons=buttons)


async def start_command(event: events.NewMessage.Event):
    buttons = [
        Button.text("Checkpoints"),
        Button.text("Lora"),
        Button.text("Memory info"),
    ]
    await event.respond(message="Welcome:", buttons=buttons)


async def menu(event, generator_client=None):
    text = f"Menu:\n"
    if generator_client:
        text += f"**Stable Diffusion checkpoint:** {await generator_client.model(current=True)}"
    buttons = [
        [Button.inline("Stable Diffusion checkpoint:")],
        [Button.inline("txt2img")],
        [Button.inline("img2img")],
        [Button.inline("Extras")],
        [Button.inline("PNG Info")],
        [Button.inline("lora")],
        [Button.inline("Memory Info")],
        [Button.inline("Text Modes")],
    ]
    if isinstance(event, events.NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    elif isinstance(event, events.CallbackQuery.Event):
        await event.edit(text=text, buttons=buttons)


async def menu_stable_diffusion_checkpoint(
    event, generator_client, data_str: str = None
):
    if data_str:
        if data_str == "Refresh":
            pass
        elif data_str == "Reload":
            pass
        elif data_str == "Unload":
            pass
        else:
            result: HTTPStatus = await generator_client.model(data_str)
            await event.answer(str(result), cache_time=2)
            await menu(event, generator_client)
    else:
        text = f"Menu/Stable Diffusion checkpoint:\n"
        result = await generator_client.model()
        buttons = button_inline_list(result)
        buttons.append(
            [
                Button.inline("Back"),
                Button.inline("Refresh"),
                Button.inline("Reload"),
                Button.inline("Unload"),
            ]
        )
        await event.edit(text=text, buttons=buttons)
        # await event.delete()
        # await event.reply("Select Model:", buttons=buttons)
        # chat = await event.get_chat()
        # telethon.tl.types.User
        # print(vars(chat))


async def menu_txt2img(event, generator_client):
    text = f"Menu/txt2img:\n"
    txt2img_obj = generator_client.txt2img_payload
    # txt2img_keys = [{a: getattr(txt2img_obj, a)} for a in dir(txt2img_obj) if not a.startswith('__')]
    width = txt2img_obj.width
    height = txt2img_obj.height
    cfg_scale = txt2img_obj.cfg_scale
    steps = txt2img_obj.steps
    sampler_name = txt2img_obj.sampler_name
    text += f"{width=}\n{height=}\n{cfg_scale=}\n{steps=}\n"
    text += f"Enter settings in following format:\n"
    text += f"`/txt2img width ` [INT]:\n"
    text += f"`/txt2img height ` [INT]:\n"
    text += f"`/txt2img cfg_scale ` [FLOAT]:\n"
    text += f"`/txt2img steps ` [INT]:\n"
    buttons = [
        [Button.inline(f"{sampler_name=}")],
        [Button.inline(f"{steps=}")],
        [Button.inline(f"{cfg_scale=}")],
        [Button.inline(f"{width=}")],
        [Button.inline(f"{height=}")],
    ]
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_txt2img_sampler_name(event, generator_client, data_str: str = None):
    if data_str:
        result = generator_client.txt2img_sampler(data_str)
        return await menu_txt2img(event, generator_client)

    text = f"**Menu/txt2img/sampler_name:**\n"
    current_sampler = generator_client.txt2img_payload.sampler_name
    samplers = generator_client.txt2img_sampler()
    # for s in samplers:
    #     text += f'{s}\n'
    # pprint.pprint(samplers)
    buttons = button_inline_list(samplers)
    # buttons = []
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def txt2img_settings(event):
    generator_client = get_image_generator(event)
    message: str = event.message.text
    option = message.split()
    if option[1] == "steps":
        steps = int(option[2])
        if steps > 2 and steps < 200:
            generator_client.txt2img_payload.steps = steps
    elif option[1] == "height":
        height = int(option[2])
        generator_client.txt2img_payload.height = height
    elif option[1] == "width":
        width = int(option[2])
        generator_client.txt2img_payload.width = width
    elif option[1] == "cfg_scale":
        cfg_scale = float(option[2])
        if cfg_scale > 0.1 and cfg_scale < 10:
            generator_client.txt2img_payload.cfg_scale = cfg_scale


async def img2img_settings(event):
    generator_client = get_image_generator(event)
    message: str = event.message.text
    option = message.split()
    if option[1] == "steps":
        steps = int(option[2])
        if steps > 2 and steps < 200:
            generator_client.img2img_payload.steps = steps
    elif option[1] == "height":
        height = int(option[2])
        generator_client.img2img_payload.height = height
    elif option[1] == "width":
        width = int(option[2])
        generator_client.img2img_payload.width = width
    elif option[1] == "cfg_scale":
        cfg_scale = float(option[2])
        if cfg_scale > 0.1 and cfg_scale < 10:
            generator_client.img2img_payload.cfg_scale = cfg_scale
    elif option[1] == "denoising_strength":
        denoising_strength = float(option[2])
        if denoising_strength > 0.1 and denoising_strength <= 1:
            generator_client.img2img_payload.denoising_strength = denoising_strength


async def menu_img2img(event, generator_client):
    text = f"Menu/img2img:\n"
    img2img = generator_client.img2img_payload
    width = img2img.width
    height = img2img.height
    cfg_scale = img2img.cfg_scale
    steps = img2img.steps
    sampler_name = img2img.sampler_name
    denoising_strength = img2img.denoising_strength

    text += f"{width=}\n{height=}\n{cfg_scale=}\n{steps=}\n{denoising_strength=}\n"
    text += f"Enter settings in following format:\n"
    text += f"`/img2img width ` [INT]:\n"
    text += f"`/img2img height ` [INT]:\n"
    text += f"`/img2img cfg_scale ` [FLOAT]:\n"
    text += f"`/img2img steps ` [INT]:\n"
    text += f"`/img2img denoising_strength ` [0-100]\n"
    buttons = [
        [Button.inline(f"{sampler_name=}")],
        [Button.inline(f"{steps=}")],
        [Button.inline(f"{cfg_scale=}")],
        [Button.inline(f"{width=}")],
        [Button.inline(f"{height=}")],
        [Button.inline(f"{denoising_strength=}")],
    ]
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_extras(event, generator_client):
    text = f"Menu/Extras:\n"
    buttons = []
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_png_info(event, generator_client):
    text = f"Menu/PNG Info:\n"
    buttons = []
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_lora(event, generator_client):
    text = f"**Menu/lora:**\n\n"
    buttons = button_inline_list(generator_client.lora())
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_memory_info(event, generator_client):
    text = f"Menu/Memory Info:\n"
    memory_info = generator_client.memory()
    text += f"{memory_info}"
    buttons = []
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def regen(event, generator_client):
    await event.answer("ok")
    message = await event.get_message()
    print(message.text)
    message_title = message_head(message.text)
    if message_title:
        print(message_title)
        if message_title == "img2img":
            await generate_img2img(event, message=message)
        elif message_title == "txt2img":
            await generate_txt2img(event, message=message.text)


async def menu_text_modes(event, generator_client):
    text = f"Menu/Text Modes:\n"
    text += f"||spoiler||\n"
    text += f"[inline URL](http://www.example.com/)\n"
    text += f"![👍](tg://emoji?id=5368324170671202286)\n"
    text += f"`inline fixed-width code`"
    text += f"```\npre-formatted fixed-width code block\n```\n"
    text += f"```python\npre-formatted fixed-width code block written in the Python programming language\n```\n"
    text += f">Block quotation started\n"
    text += f">Block quotation started\n"
    text += f">The last line of the block quotation\n"
    text += f"**>The expandable block quotation started right after the previous block quotation\n"
    text += f""
    # text2 =
    buttons = []
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def callback_query_handler(event: events.CallbackQuery.Event):
    generator_client = get_image_generator(event)
    event_str = event.data.decode("utf-8")
    #     events.CallbackQuery(data=b'Unload')
    #     events.CallbackQuery(data=b'Refresh')
    #     events.CallbackQuery(data=b'Current')
    try:
        if event.data == b"Regen":
            await regen(event, generator_client)
        elif event.data.startswith(b"Stable Diffusion checkpoint:"):
            await menu_stable_diffusion_checkpoint(event, generator_client)
        elif event.data == b"txt2img":
            await menu_txt2img(event, generator_client)
        elif event.data == b"img2img":
            await menu_img2img(event, generator_client)
        elif event.data == b"Extras":
            await menu_extras(event, generator_client)
        elif event.data == b"PNG Info":
            await menu_png_info(event, generator_client)
        elif event.data == b"lora":
            await menu_lora(event, generator_client)
        elif event.data == b"Memory Info":
            await menu_memory_info(event, generator_client)
        elif event.data.startswith(b"sampler_name="):
            await menu_txt2img_sampler_name(event, generator_client)
        elif event.data == b"Text Modes":
            await menu_text_modes(event, generator_client)

        elif event.data == b"Back":
            message = await event.get_message()
            pattern = re.compile("([\w\s]+)/[\w\s]+:")
            back_menu = re.search(pattern, message.text)
            if back_menu:
                if back_menu[1] == "Menu":
                    await menu(event, generator_client)
                elif back_menu[1] == "txt2img":
                    await menu_txt2img(event, generator_client)
        else:
            message = await event.get_message()
            pattern = re.compile("/?([\w\s]+):")
            current_menu = re.search(pattern, message.text)
            print(message.text)
            if current_menu:
                if current_menu[1] == "sampler_name":
                    await menu_txt2img_sampler_name(event, generator_client, event_str)
                if current_menu[1] == "Stable Diffusion checkpoint":
                    await menu_stable_diffusion_checkpoint(
                        event, generator_client, event_str
                    )
    except ConnectError as e:
        await event.answer(str(e))


async def main():

    session = os.path.basename(__file__).split(".")[0]
    session_path = os.path.join(dir_path, session)

    logging.basicConfig()
    # logger = logging.getLogger(os.path.basename(__file__)).getChild(__class__.__name__)
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(config["log_level"])
    allowed_chat_ids = config["allowed_chat_ids"]

    telegram_client: TelegramBot = TelegramBot(
        session_path,
        config["api_id"],
        config["api_hash"],
        allowed_chat_ids=allowed_chat_ids,
        log_level=config["log_level"],
        proxy=config["proxy"],
        retry_delay=5,
    )

    await telegram_client.start(bot_token=config["bot_token"])

    telegram_client.add_event_handler(
        callback=start_command,
        event=events.NewMessage(
            chats=allowed_chat_ids, incoming=True, pattern="(?i)/start$"
        ),
    )

    telegram_client.add_event_handler(
        callback=menu,
        event=events.NewMessage(
            chats=allowed_chat_ids, incoming=True, pattern="(?i)/menu$"
        ),
    )

    telegram_client.add_event_handler(
        callback=generate_txt2img,
        event=events.NewMessage(
            chats=allowed_chat_ids, incoming=True, pattern="/img\s(.*)$"
        ),
    )

    telegram_client.add_event_handler(
        callback=txt2img_settings,
        event=events.NewMessage(
            chats=allowed_chat_ids, incoming=True, pattern="/txt2img\s(.*)$"
        ),
    )

    telegram_client.add_event_handler(
        callback=img2img_settings,
        event=events.NewMessage(
            chats=allowed_chat_ids, incoming=True, pattern="/img2img\s(.*)$"
        ),
    )

    telegram_client.add_event_handler(
        callback=generate_img2img,
        event=events.NewMessage(
            chats=allowed_chat_ids, incoming=True, func=lambda e: e.photo
        ),
    )

    telegram_client.add_event_handler(
        callback=check_reply,
        event=events.NewMessage(
            chats=allowed_chat_ids, incoming=True, func=lambda e: e.message.is_reply
        ),
    )

    telegram_client.add_event_handler(
        callback_query_handler, events.CallbackQuery(chats=allowed_chat_ids)
    )

    logger.info("Generator Bot is active.")

    await telegram_client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
