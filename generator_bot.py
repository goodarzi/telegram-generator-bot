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
from generators.webui import WebuiClient
from generators.forge import ForgeClient
from helpers import utils
from bot import (
    TelegramBot,
    Message,
    Button,
    NewMessage,
    MessageEdited,
    CallbackQuery,
)

dir_path = os.path.dirname(os.path.realpath(__file__))
config_file = os.path.join(dir_path, "config.yaml")
if config_file:
    config = utils.load_config(config_file)
else:
    config = {}

output_dir = os.path.join(dir_path, config["extensions"]["image_generator_output_dir"])


def get_image_generator(event):

    if config["extensions"]["image_generator"] == "webui":
        if (
            "username" in config["extensions"]["webui"]
            and "password" in config["extensions"]["webui"]
        ):
            auth = BasicAuth(
                username=config["extensions"]["webui"]["username"],
                password=config["extensions"]["webui"]["password"],
            )
        else:
            auth = None
        return WebuiClient(
            base_url=config["extensions"]["webui"]["base_url"],
            chat_id=event.chat_id,
            out_dir=output_dir,
            auth=auth,
        )
    elif config["extensions"]["image_generator"] == "forge":
        if (
            "username" in config["extensions"]["forge"]
            and "password" in config["extensions"]["forge"]
        ):
            auth = BasicAuth(
                username=config["extensions"]["forge"]["username"],
                password=config["extensions"]["forge"]["password"],
            )
        else:
            auth = None
        return ForgeClient(
            base_url=config["extensions"]["forge"]["base_url"],
            chat_id=event.chat_id,
            out_dir=output_dir,
            auth=auth,
        )


def message_to_prompt(message: str):
    msg_split = message.split("\n")
    print(msg_split)
    negative_prompt: str = None
    if len(msg_split) > 1:
        prompt = msg_split[1].strip()
        prompt = prompt.strip("`")
        prompt = prompt.lstrip("/img ")
        if len(msg_split) > 2:
            msg_split[2] = msg_split[2].strip()
            if msg_split[2].lower().startswith("negative prompt:"):
                negative_prompt = msg_split[2][16:]
            elif msg_split[2].lower().startswith("negative:"):
                negative_prompt = msg_split[2][9:]
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


def save_live_image(image):
    file_path = f"/tmp/{image[0]}.png"
    with open(file_path, "wb") as output:
        output.write(base64.b64decode(image[1]))
    return file_path


def get_full_path(path):
    if os.path.exists(path):
        return os.path.realpath(path)
    else:
        if os.path.exists(os.path.dirname(path)):
            os.mkdir(path)
            return os.path.realpath(path)
        else:
            full_path = os.path.join(os.path.realpath(""), path)
            if not os.path.exists(full_path):
                os.mkdir(full_path)
            return full_path


async def update_progress(generator_client, message: Message, id_task) -> Message:
    status: Message = None
    waited = None
    buttons = [Button.inline("Intrrupt"), Button.inline("Skip")]
    last_text = ""
    async for s in generator_client.task_progress(id_task):
        status_text = f"**{s[2]}** \n"
        if not status:
            if s[1]:
                picframe = save_live_image(s[1])
            elif waited:
                picframe = os.path.join(dir_path, "helpers/folder-adwaita-pictures.png")
            else:
                asyncio.sleep(1)
                waited = True
                continue
            status = await message.reply(
                message=status_text,
                file=picframe,
                thumb=None,
                buttons=buttons,
            )
            continue

        text = f"{status_text}\n"
        text += f"\n📥 : [{utils.create_progress_bar(s[0]*100)}]"
        if last_text != text:
            last_text = text
        else:
            continue
        try:
            if s[1]:
                # await status.edit(
                #     text=text, buttons=buttons, file=save_live_image(s[1])
                # )
                await status.edit(text=text, buttons=buttons)
            else:
                await status.edit(text=text, buttons=buttons)
        except Exception as e:
            await status.reply(str(e))
            break
    return status


def create_infotext(info, images, gentype: str):
    infotext = f"Steps: {info['steps']}, Sampler: {info['sampler_name']}, CFG scale: {info['cfg_scale']}, Seed: {info['all_seeds']}, Size: {info['height']}x{info['width']}, Model: {info['sd_model_name']}, Clip skip: {info['clip_skip']}, "
    if gentype == "img2img":
        infotext += f"Denoising strength: {info['denoising_strength']}, "
    for index, image in enumerate(images):
        if len(info["all_negative_prompts"]) >= index + 1:
            if info["all_negative_prompts"][index]:
                infotext = (
                    f"Negative prompt: {info['all_negative_prompts'][index]}\n"
                    + infotext
                )
        if len(info["all_prompts"]) >= index + 1:
            if info["all_prompts"][index]:
                infotext = (
                    f"**{gentype}:** \n`/img {info['all_prompts'][index]}` \n"
                    + infotext
                )
        if "ADetailer model" in info["extra_generation_params"].keys():
            infotext += f"ADetailer model: {info['extra_generation_params']['ADetailer model']}, "
        if len(infotext) > 4096:
            infotext = infotext[:4096]
        yield infotext, image


async def generate_txt2img(
    event: Union[NewMessage.Event, MessageEdited.Event, CallbackQuery.Event],
    message: Message = None,
    text: str = None,
):
    generator_client = get_image_generator(event)
    payload = generator_client.txt2img_payload
    payload.force_task_id = uuid.uuid4().hex
    if not message:
        message = event.message
    if text:
        payload.prompt, payload.negative_prompt = message_to_prompt(text)
    elif event.pattern_match:
        payload.prompt = event.pattern_match[1]
    else:
        payload.prompt, payload.negative_prompt = message_to_prompt(message.text)

    async with asyncio.TaskGroup() as tg:
        generate = tg.create_task(generator_client.txt2img(payload))
        progress = tg.create_task(
            update_progress(generator_client, message, payload.force_task_id)
        )

    info = json.loads(generate.result().info)
    buttons = [Button.inline("Regen"), Button.inline("File")]
    upload_files = []
    info_texts = []
    for infotext, image in create_infotext(info, generate.result().images, "txt2img"):
        upload_file = await event.client.upload_file(
            base64.b64decode(image),
            file_name=f"txt2img-{utils.timestamp()}.png",
        )
        upload_files.append(upload_file)
        info_texts.append(infotext)

    if len(upload_files) > 1:
        # print(info_texts)
        edit_msg = await progress.result().edit(
            text=f"{info_texts[0]}",
            buttons=buttons,
        )
        photoalbum = await event.respond(
            file=upload_files,
            reply_to=edit_msg.id,
        )
    else:
        edit_msg = await progress.result().edit(
            file=upload_files[0],
            text=f"{info_texts[0]}",
            buttons=buttons,
        )
    generator_client.tg_msg_id_input_files[edit_msg.id] = upload_files


async def check_reply(event: Union[NewMessage.Event, MessageEdited.Event]):
    reply_message = await event.message.get_reply_message()
    if reply_message.file.mime_type in ["image/png", "image/jpg", "image/jpeg"]:
        await generate_img2img(event, reply_message, text=event.message.text)


async def clip(event: NewMessage.Event):
    generator_client = get_image_generator(event)
    img = await event.message.download_media()
    with open(img, "rb") as file:
        b64img = base64.b64encode(file.read()).decode("utf-8")

    # result = await generator_client.interrogate_post(b64img, "deepdanbooru")
    result = await generator_client.interrogate_post(b64img, "clip")
    await event.respond(str(result))


async def png_info(event: NewMessage.Event):
    generator_client = get_image_generator(event)
    img = await event.message.download_media()
    with open(img, "rb") as file:
        b64img = base64.b64encode(file.read()).decode("utf-8")

    result = await generator_client.png_info_post(b64img)
    if result.info:
        await event.respond(str(result.info))
    else:
        await event.respond("no info")


async def generate_img2img(event, message, text: str = None):

    generator_client = get_image_generator(event)
    payload = generator_client.img2img_payload
    payload.force_task_id = uuid.uuid4().hex

    if not message.media:
        await event.respond("no message media!")
        return

    img = await TelegramBot.download_image(message, os.path.realpath(""))
    print(img)
    payload.prompt, payload.negative_prompt = message_to_prompt(text)

    with open(img, "rb") as file:
        b64img = base64.b64encode(file.read()).decode("utf-8")
    init_images = [
        b64img,
    ]

    payload.init_images = init_images

    generate = asyncio.create_task(generator_client.img2img(payload))
    progress = asyncio.create_task(
        update_progress(generator_client, message, payload.force_task_id)
    )
    await generate
    await progress
    info = json.loads(generate.result().info)
    # progress_text = progress.result().message
    buttons = [Button.inline("Regen"), Button.inline("File")]
    upload_files = []
    info_texts = []
    for infotext, image in create_infotext(info, generate.result().images, "img2img"):
        upload_file = await event.client.upload_file(
            base64.b64decode(image),
            file_name=f"txt2img-{utils.timestamp()}.png",
        )
        upload_files.append(upload_file)
        info_texts.append(infotext)

    if len(upload_files) > 1:
        # print(info_texts)
        edit_msg = await progress.result().edit(
            text=f"{info_texts[0]}",
            buttons=buttons,
        )
        photoalbum = await event.respond(
            file=upload_files,
            reply_to=edit_msg.id,
        )
    else:
        edit_msg = await progress.result().edit(
            file=upload_files[0],
            text=f"{info_texts[0]}",
            buttons=buttons,
        )
    generator_client.tg_msg_id_input_files[edit_msg.id] = upload_files


async def start_command(event: NewMessage.Event):
    text = f"Welcome:\n"
    text += f"`/menu` : Main menu\n"
    text += f"Generate txt2img:\n"
    text += f"Send prompt with following format , negative prompt is optional:\n"
    text += f"`/p [prompt] /n [negative prompt]` : Generate Image\n"
    text += f"`/txt2img [option] [value]` : Set txt2img options\n"
    text += f"Generate img2img:\n"
    text += (
        f"Reply any image file in chat with prompt will run img2img for that image\n"
    )
    text += f"`/img2img [option] [value]` : Set img2img options\n"

    await event.respond(message=text)


async def menu(event, generator_client=None):
    text = f"**Menu:**\n"
    buttons = [
        [Button.inline("Stable Diffusion checkpoints")],
        [Button.inline("txt2img"), Button.inline("img2img")],
        [Button.inline("Extras"), Button.inline("PNG Info"), Button.inline("lora")],
        [Button.inline("Memory Info"), Button.inline("Text Modes")],
    ]

    try:
        if not generator_client:
            generator_client = get_image_generator(event)

        options = generator_client.options_get()
    except Exception as e:
        if isinstance(event, NewMessage.Event):
            await event.respond(message=str(e), buttons=buttons)
        elif isinstance(event, CallbackQuery.Event):
            await event.edit(text=str(e), buttons=buttons)
        return
    text += f"**├─Checkpoint:** {options.sd_model_checkpoint}\n"

    if isinstance(generator_client, ForgeClient):
        text += f"**├─Forge preset:** {options.forge_preset}\n"
        buttons.insert(0, [Button.inline("forge")])
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    elif isinstance(event, CallbackQuery.Event):
        await event.edit(text=text, buttons=buttons)


async def menu_forge(
    event, generator_client: Union[WebuiClient, ForgeClient], data_str: str = None
):
    if data_str:
        if data_str in ["sd", "xl", "flux", "all"]:
            try:
                generator_client.set_forge_preset(data_str)
                await event.answer(data_str, cache_time=2)
                await menu_forge(event, generator_client)
            except Exception as e:
                await event.answer(str(e), cache_time=2)
    else:
        options = generator_client.options_get()
        forge_async_loading = options.forge_async_loading
        forge_pin_shared_memory = options.forge_pin_shared_memory
        forge_inference_memory = options.forge_inference_memory
        forge_unet_storage_dtype = options.forge_unet_storage_dtype
        forge_additional_modules = generator_client.sd_modules(current=True)
        clip_stop_at_last_layers = options.clip_stop_at_last_layers
        text = f"Menu/forge:\n"
        text += f"VAE / Text Encoder:\n"
        for i in forge_additional_modules:
            text += f"  - {i}\n"
        buttons = [
            [
                Button.inline("sd"),
                Button.inline("xl"),
                Button.inline("flux"),
                Button.inline("all"),
            ]
        ]
        if options.forge_preset == "flux":
            text += f"`/options forge_inference_memory `: {forge_inference_memory}\n"
            buttons.append([Button.inline(f"{forge_async_loading=}")])
            buttons.append([Button.inline(f"{forge_pin_shared_memory=}")])
            buttons.append([Button.inline(f"{forge_unet_storage_dtype=}")])
            buttons.append([Button.inline(f"forge_additional_modules=")])
        elif options.forge_preset == "xl":
            text += f"`/options forge_inference_memory `: {forge_inference_memory}\n"
            buttons.append([Button.inline(f"{forge_unet_storage_dtype=}")])
            buttons.append([Button.inline(f"forge_additional_modules=")])
        elif options.forge_preset == "sd":
            text += (
                f"`/options CLIP_stop_at_last_layers `: {clip_stop_at_last_layers}\n"
            )
            buttons.append([Button.inline(f"forge_additional_modules=")])
        elif options.forge_preset == "all":
            text += (
                f"`/options CLIP_stop_at_last_layers `: {clip_stop_at_last_layers}\n"
            )
            text += f"`/options forge_inference_memory `: {forge_inference_memory}"
            buttons.append([Button.inline(f"{forge_async_loading=}")])
            buttons.append([Button.inline(f"{forge_pin_shared_memory=}")])
            buttons.append([Button.inline(f"{forge_unet_storage_dtype=}")])
            buttons.append([Button.inline(f"forge_additional_modules=")])

        buttons.append([Button.inline("Back")])
        await event.edit(text=text, buttons=buttons)


async def menu_forge_async_loading(event, generator_client, data_str: str = None):
    if data_str:
        body = {"forge_async_loading": data_str}
        try:
            generator_client.options_post(body)
            await event.answer(data_str, cache_time=2)
            return await menu_forge(event, generator_client)
        except Exception as e:
            await event.answer(str(e), cache_time=2)
    text = f"**Menu/forge/forge_async_loading:**\n"
    buttons = [[Button.inline("Queue"), Button.inline("Async")]]
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_forge_pin_shared_memory(event, generator_client, data_str: str = None):
    if data_str:
        body = {"forge_pin_shared_memory": data_str}
        try:
            generator_client.options_post(body)
            await event.answer(data_str, cache_time=2)
            return await menu_forge(event, generator_client)
        except Exception as e:
            await event.answer(str(e), cache_time=2)
    text = f"**Menu/forge/forge_pin_shared_memory:**\n"
    buttons = [[Button.inline("CPU"), Button.inline("Shared")]]
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_forge_unet_storage_dtype(event, generator_client, data_str: str = None):
    if data_str:
        body = {"forge_unet_storage_dtype": data_str}
        try:
            generator_client.options_post(body)
            await event.answer(data_str, cache_time=2)
            return await menu_forge(event, generator_client)
        except Exception as e:
            await event.answer(str(e), cache_time=2)
    text = f"**Menu/forge/forge_unet_storage_dtype:**\n"
    buttons = [[Button.inline("Automatic"), Button.inline("Automatic (fp16 LoRA)")]]
    buttons.append([Button.inline("bnb-nf4"), Button.inline("bnb-nf4 (fp16 LoRA)")])
    buttons.append(
        [Button.inline("float8-e4m3fn"), Button.inline("float8-e4m3fn (fp16 LoRA)")]
    )
    buttons.append([Button.inline("bnb-fp4"), Button.inline("bnb-fp4 (fp16 LoRA)")])
    buttons.append(
        [Button.inline("float8-e5m2"), Button.inline("float8-e5m2 (fp16 LoRA)")]
    )
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_forge_additional_modules(event, generator_client, data_str: str = None):
    if data_str:
        try:
            generator_client.sd_modules(data_str)
            await event.answer(data_str, cache_time=2)
            return await menu_forge_additional_modules(event, generator_client)
        except Exception as e:
            await event.answer(str(e), cache_time=2)
    text = f"**Menu/forge/forge_additional_modules:**\n"
    forge_additional_modules = generator_client.sd_modules(current=True)
    text += f"VAE / Text Encoder:\n"
    for i in forge_additional_modules:
        text += f"  - {i}\n"
    buttons = TelegramBot.button_inline_list(generator_client.sd_modules())
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def options(event: NewMessage.Event):
    generator_client = get_image_generator(event)
    message: str = event.message.text
    option = message.split()
    try:
        new_options = {option[1]: option[2]}
        generator_client.set_options(new_options)
        await event.respond("ok")
    except Exception as e:
        await event.respond(message=str(e))


async def menu_stable_diffusion_checkpoint(
    event, generator_client, data_str: str = None
):
    if data_str:
        if data_str == "Refresh":
            await generator_client.refresh_checkpoints_post()
            await event.answer(data_str, cache_time=2)
            await menu_stable_diffusion_checkpoint(event, generator_client)
        elif data_str == "Reload":
            await generator_client.reload_checkpoint_post()
            await event.answer(data_str, cache_time=2)
            await menu_stable_diffusion_checkpoint(event, generator_client)
        elif data_str == "Unload":
            await generator_client.unload_checkpoint_post()
            await event.answer(data_str, cache_time=2)
            await menu_stable_diffusion_checkpoint(event, generator_client)
        else:
            result: HTTPStatus = generator_client.model(data_str)
            await event.answer(str(result), cache_time=2)
            await menu(event, generator_client)
    else:
        text = f"Menu/Stable Diffusion checkpoints:\n"
        result = generator_client.model()
        buttons = TelegramBot.button_inline_list(result)
        buttons.append(
            [
                Button.inline("Back"),
                Button.inline("Refresh"),
                Button.inline("Reload"),
                Button.inline("Unload"),
            ]
        )
        await event.edit(text=text, buttons=buttons)


async def menu_txt2img(event, generator_client):
    text = f"Menu/txt2img:\n"
    text += f"Enter settings in following format:\n"
    for k in generator_client.txt2img_settings():
        text += f"`/txt2img {k} `: {getattr(generator_client.txt2img_payload, k)} \n"
    # txt2img_obj = generator_client.txt2img_payload
    # txt2img_keys = [{a: getattr(txt2img_obj, a)} for a in dir(txt2img_obj) if not a.startswith('__')]
    sampler_name = generator_client.txt2img_payload.sampler_name
    scheduler = generator_client.txt2img_payload.scheduler
    buttons = [
        [Button.inline(f"{sampler_name=}")],
        [Button.inline("Back")],
    ]
    if isinstance(generator_client, ForgeClient):
        buttons.insert(1, [Button.inline(f"{scheduler=}")])
    await event.edit(text=text, buttons=buttons)


async def menu_txt2img_sampler_name(event, generator_client, data_str: str = None):
    if data_str:
        generator_client.txt2img_payload.sampler_name = data_str
        return await menu_txt2img(event, generator_client)
    text = f"**Menu/txt2img/sampler_name:**\n"
    current_sampler = generator_client.txt2img_payload.sampler_name
    samplers = generator_client.get_samplers()
    buttons = TelegramBot.button_inline_list(samplers)
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_txt2img_scheduler(event, generator_client, data_str: str = None):
    if data_str:
        generator_client.txt2img_payload.scheduler = data_str
        return await menu_txt2img(event, generator_client)
    text = f"**Menu/txt2img/scheduler:**\n"
    current_scheduler = generator_client.txt2img_payload.scheduler
    schedulers = generator_client.get_schedulers()
    buttons = TelegramBot.button_inline_list(schedulers)
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def set_txt2img_payload(event: NewMessage.Event):
    generator_client = get_image_generator(event)
    message: str = event.message.text
    option = message.split()
    try:
        setattr(generator_client.txt2img_payload, option[1], option[2])
        print(type(generator_client.txt2img_payload.__getattribute__(option[1])))
    except (AttributeError, TypeError, ValueError) as e:
        await event.respond(message=str(e))


async def menu_img2img(event, generator_client):
    text = f"Menu/img2img:\n"
    text += f"Enter settings in following format:\n"
    for k in generator_client.img2img_settings():
        text += f"`/img2img {k} `: {getattr(generator_client.img2img_payload, k)} \n"
    # txt2img_obj = generator_client.img2img_payload
    # txt2img_keys = [{a: getattr(img2img_obj, a)} for a in dir(img2img_obj) if not a.startswith('__')]
    sampler_name = generator_client.img2img_payload.sampler_name
    scheduler = generator_client.img2img_payload.scheduler
    buttons = [
        [Button.inline(f"{sampler_name=}")],
        [Button.inline("Back")],
    ]
    if isinstance(generator_client, ForgeClient):
        buttons.insert(1, [Button.inline(f"{scheduler=}")])
    await event.edit(text=text, buttons=buttons)


async def menu_img2img_sampler_name(event, generator_client, data_str: str = None):
    if data_str:
        generator_client.img2img_payload.sampler_name = data_str
        return await menu_img2img(event, generator_client)
    text = f"**Menu/img2img/sampler_name:**\n"
    current_sampler = generator_client.img2img_payload.sampler_name
    samplers = generator_client.get_samplers()
    buttons = TelegramBot.button_inline_list(samplers)
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_img2img_scheduler(event, generator_client, data_str: str = None):
    if data_str:
        generator_client.img2img_payload.scheduler = data_str
        return await menu_img2img(event, generator_client)
    text = f"**Menu/img2img/scheduler:**\n"
    current_scheduler = generator_client.img2img_payload.scheduler
    schedulers = generator_client.get_schedulers()
    buttons = TelegramBot.button_inline_list(schedulers)
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def set_img2img_payload(event: NewMessage.Event):
    generator_client = get_image_generator(event)
    message: str = event.message.text
    option = message.split()
    try:
        utils.check_obj_attr_type(
            generator_client.img2img_payload, option[1], option[2]
        )
        attr_type = type(getattr(generator_client.img2img_payload, option[1]))
        value_intype = attr_type(option[2])
        generator_client.img2img_payload.__setattr__(option[1], value_intype)
    except (AttributeError, TypeError) as e:
        await event.respond(message=str(e))


async def menu_extras(event, generator_client):
    text = f"Menu/Extras:\n"
    buttons = []
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_png_info(event, generator_client):
    text = f"Menu/PNG Info:\n"
    text += f"Send the photo as a file,\n"
    text += (
        f'To do this, turn off the "Compress the image" option when sending a photo\n'
    )
    text += f"This bot always runs png info for photos that are sent as files.\n"
    buttons = []
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_lora(event, generator_client):
    text = f"**Menu/lora:**\n\n"
    buttons = TelegramBot.button_inline_list(generator_client.lora())
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def menu_memory_info(event, generator_client):
    text = f"**Menu/Memory Info:**\n"
    result = generator_client.get_memory()
    text += "**├─RAM**\n"
    text += f"│ ├─Free:{utils.format_byte(result['ram']['free'])} \tUsed:{utils.format_byte(result['ram']['used'])} \tTotal:{utils.format_byte(result['ram']['total'])}\n"
    text += "**├─CUDA**\n"
    text += f"│ ├─Free:{utils.format_byte(result['cuda']['free'])} \tUsed:{utils.format_byte(result['cuda']['used'])} \tTotal:{utils.format_byte(result['cuda']['total'])}\n"
    buttons = []
    buttons.append([Button.inline("Back")])
    await event.edit(text=text, buttons=buttons)


async def regen(event):
    await event.answer("ok")
    message = await event.get_message()
    message_title = message_head(message.text)
    if message_title:
        print(message_title)
        if message_title == "img2img":
            reply_to = await message.get_reply_message()
            await generate_img2img(event, message=reply_to, text=message.text)
        elif message_title == "txt2img":
            await generate_txt2img(event, message=message, text=message.text)


async def send_as_file(
    event: CallbackQuery.Event, generator_client: Union[WebuiClient, ForgeClient]
):
    message = await event.get_message()
    if message.id in generator_client.tg_msg_id_input_files:
        await event.answer("Sending as File")
        await event.reply(
            file=generator_client.tg_msg_id_input_files[message.id], force_document=True
        )
    else:
        await event.answer("File not Found")


async def intrrupt(
    event: CallbackQuery.Event, generator_client: Union[WebuiClient, ForgeClient]
):
    result = generator_client.interrupt_post()
    print(result)
    await event.answer(str(result.status_code))


async def skip(
    event: CallbackQuery.Event, generator_client: Union[WebuiClient, ForgeClient]
):
    result = generator_client.skip_post()
    print(result)
    await event.answer(str(result.status_code))


# async def menu_text_modes(event, generator_client):
#     text = f"Menu/Text Modes:\n"
#     text += f"||spoiler||\n"
#     text += f"[inline URL](http://www.example.com/)\n"
#     text += f"![👍](tg://emoji?id=5368324170671202286)\n"
#     text += f"`inline fixed-width code`"
#     text += f"```\npre-formatted fixed-width code block\n```\n"
#     text += f"```python\npre-formatted fixed-width code block written in the Python programming language\n```\n"
#     buttons = []
#     buttons.append([Button.inline("Back")])
#     await event.edit(text=text, buttons=buttons)


async def callback_query_handler(event: CallbackQuery.Event):
    generator_client = get_image_generator(event)
    event_str = event.data.decode("utf-8")
    message = await event.get_message()
    back_menu_pattern = re.compile("([\w\s]+)/[\w\s]+:")
    back_menu = re.search(back_menu_pattern, message.text)
    current_menu_pattern = re.compile("/?([\w\s]+):")
    current_menu = re.search(current_menu_pattern, message.text)
    try:
        if event.data == b"Regen":
            await regen(event)
        elif event.data == b"File":
            await send_as_file(event, generator_client)
        elif event.data == b"Intrrupt":
            await intrrupt(event, generator_client)
        elif event.data == b"Skip":
            await skip(event, generator_client)
        elif event.data == b"Stable Diffusion checkpoints":
            await menu_stable_diffusion_checkpoint(event, generator_client)
        elif event.data == b"forge":
            await menu_forge(event, generator_client)
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

        # elif event.data == b"Text Modes":
        #     await menu_text_modes(event, generator_client)

        elif event.data == b"Back" and back_menu:
            if back_menu[1] == "Menu":
                await menu(event, generator_client)
            elif back_menu[1] == "forge":
                await menu_forge(event, generator_client)
            elif back_menu[1] == "txt2img":
                await menu_txt2img(event, generator_client)
            elif back_menu[1] == "img2img":
                await menu_img2img(event, generator_client)
        elif current_menu:
            if current_menu[1] == "Stable Diffusion checkpoints":
                await menu_stable_diffusion_checkpoint(
                    event, generator_client, event_str
                )
            elif current_menu[1] == "forge":
                if event.data.startswith(b"forge_async_loading="):
                    await menu_forge_async_loading(event, generator_client)
                elif event.data.startswith(b"forge_pin_shared_memory="):
                    await menu_forge_pin_shared_memory(event, generator_client)
                elif event.data.startswith(b"forge_unet_storage_dtype="):
                    await menu_forge_unet_storage_dtype(event, generator_client)
                elif event.data.startswith(b"forge_additional_modules="):
                    await menu_forge_additional_modules(event, generator_client)
                else:
                    await menu_forge(event, generator_client, event_str)
            elif current_menu[1] == "txt2img":
                if event.data.startswith(b"sampler_name="):
                    await menu_txt2img_sampler_name(event, generator_client)
                elif event.data.startswith(b"scheduler="):
                    await menu_txt2img_scheduler(event, generator_client)
            elif current_menu[1] == "img2img":
                if event.data.startswith(b"sampler_name="):
                    await menu_img2img_sampler_name(event, generator_client)
                elif event.data.startswith(b"scheduler="):
                    await menu_img2img_scheduler(event, generator_client)
            elif current_menu[1] == "forge_async_loading":
                await menu_forge_async_loading(event, generator_client, event_str)
            elif current_menu[1] == "forge_pin_shared_memory":
                await menu_forge_pin_shared_memory(event, generator_client, event_str)
            elif current_menu[1] == "forge_unet_storage_dtype":
                await menu_forge_unet_storage_dtype(event, generator_client, event_str)
            elif current_menu[1] == "forge_additional_modules":
                await menu_forge_additional_modules(event, generator_client, event_str)
            elif current_menu[1] == "sampler_name" and back_menu[1] == "txt2img":
                await menu_txt2img_sampler_name(event, generator_client, event_str)
            elif current_menu[1] == "sampler_name" and back_menu[1] == "img2img":
                await menu_img2img_sampler_name(event, generator_client, event_str)
            elif current_menu[1] == "scheduler" and back_menu[1] == "txt2img":
                await menu_txt2img_scheduler(event, generator_client, event_str)
            elif current_menu[1] == "scheduler" and back_menu[1] == "img2img":
                await menu_img2img_scheduler(event, generator_client, event_str)
            elif current_menu[1] == "Stable Diffusion checkpoints":
                await menu_stable_diffusion_checkpoint(
                    event, generator_client, event_str
                )

    except Exception as e:
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
        event=NewMessage(chats=allowed_chat_ids, incoming=True, pattern="(?i)/start$"),
    )

    telegram_client.add_event_handler(
        callback=menu,
        event=NewMessage(chats=allowed_chat_ids, incoming=True, pattern="(?i)/menu$"),
    )

    telegram_client.add_event_handler(
        callback=generate_txt2img,
        event=NewMessage(chats=allowed_chat_ids, incoming=True, pattern="/img\s(.*)$"),
    )

    telegram_client.add_event_handler(
        callback=generate_txt2img,
        event=MessageEdited(
            chats=allowed_chat_ids, incoming=True, pattern="/img\s(.*)$"
        ),
    )

    telegram_client.add_event_handler(
        callback=set_txt2img_payload,
        event=NewMessage(
            chats=allowed_chat_ids, incoming=True, pattern="/txt2img\s(.*)$"
        ),
    )

    telegram_client.add_event_handler(
        callback=set_img2img_payload,
        event=NewMessage(
            chats=allowed_chat_ids, incoming=True, pattern="/img2img\s(.*)$"
        ),
    )

    telegram_client.add_event_handler(
        callback=options,
        event=NewMessage(
            chats=allowed_chat_ids, incoming=True, pattern="/options\s(.*)$"
        ),
    )

    telegram_client.add_event_handler(
        # callback=generate_img2img,
        callback=clip,
        event=NewMessage(chats=allowed_chat_ids, incoming=True, func=lambda e: e.photo),
    )

    telegram_client.add_event_handler(
        callback=png_info,
        event=NewMessage(
            chats=allowed_chat_ids, incoming=True, func=TelegramBot.media_is_png
        ),
    )

    telegram_client.add_event_handler(
        callback=check_reply,
        event=NewMessage(
            chats=allowed_chat_ids, incoming=True, func=lambda e: e.message.is_reply
        ),
    )

    telegram_client.add_event_handler(
        callback_query_handler, CallbackQuery(chats=allowed_chat_ids)
    )

    logger.info("Generator Bot is active.")

    await telegram_client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
