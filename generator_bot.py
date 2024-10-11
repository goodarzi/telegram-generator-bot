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
from generators.generator_client import GeneratorClient
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


async def update_progress(message: Message, id_task, payload_info=None) -> Message:
    status: Message = None
    buttons = [Button.inline("Intrrupt"), Button.inline("Skip")]
    last_text = ""
    generator_client = GeneratorClient(message.chat_id).image
    async for progress, livephoto, info in generator_client.task_progress(id_task):
        text = payload_info if payload_info else ""
        if info:
            text += f"{info}\n"
        text += f"\n📥 : [{utils.create_progress_bar(progress*100)}]"
        if last_text != text:
            last_text = text
        else:
            continue
        if not status:
            upload_livephoto = (
                await message.client.upload_file(
                    livephoto[1],
                    file_name=f"{livephoto[0]}",
                )
                if livephoto
                else b"0"
            )

            status = await message.reply(
                message=text,
                file=upload_livephoto,
                thumb=None,
                buttons=buttons,
            )
            continue
        try:
            if livephoto:
                if not status.photo:
                    file = await message.client.upload_file(
                        livephoto[1], file_name=f"{livephoto[0]}"
                    )
                    status = await status.edit(text=text, buttons=buttons, file=file)
                else:
                    await status.edit(text=text, buttons=buttons)
            else:
                await status.edit(text=text, buttons=buttons)
        except Exception as e:
            await status.reply(str(e))
            break
    return status


async def generate_txt2img(
    event: Union[NewMessage.Event, MessageEdited.Event, CallbackQuery.Event],
    message: Message = None,
    text: str = None,
):
    buttons = [Button.inline("Regen"), Button.inline("File")]
    generator_client = GeneratorClient(event.chat_id).image
    payload = generator_client.txt2img_payload
    payload.force_task_id = uuid.uuid4().hex

    if not message:
        message = event.message
    if text:
        payload.prompt, payload.negative_prompt = GeneratorClient.txt2prompt(text)
    elif event.pattern_match:
        payload.prompt = event.pattern_match[1]
    else:
        payload.prompt, payload.negative_prompt = GeneratorClient.txt2prompt(
            message.text
        )

    payload_info = (
        f"`/p {payload.prompt}`\n"
        if not payload.negative_prompt
        else f"`/p {payload.prompt}` `/n {payload.negative_prompt}`\n"
    )
    payload_info = f"**txt2img:** \n" + payload_info

    for k in generator_client.txt2img_info():
        payload_info += (
            f"`/txt2img {k} {getattr(generator_client.txt2img_payload, k)}` \n"
        )

    async with asyncio.TaskGroup() as tg:
        generate = tg.create_task(generator_client.txt2img(payload))
        progress = tg.create_task(
            update_progress(message, payload.force_task_id, payload_info)
        )

    info = json.loads(generate.result().info)
    payload_info += f"`/txt2img seed {info['all_seeds']}`\n"

    upload_files = [
        await event.client.upload_file(
            base64.b64decode(image),
            file_name=f"txt2img-{utils.timestamp()}.png",
        )
        for image in generate.result().images
    ]

    if len(upload_files) > 1 or (not progress.result().photo):
        # print(info_texts)
        edit_msg = await progress.result().edit(
            text=f"{payload_info}",
            buttons=buttons,
        )
        photoalbum = await event.respond(
            file=upload_files,
            reply_to=edit_msg.id,
        )
    else:
        edit_msg = await progress.result().edit(
            file=upload_files[0],
            text=f"{payload_info}",
            buttons=buttons,
        )
    generator_client.tg_msg_id_input_files[edit_msg.id] = upload_files


async def check_reply(event: Union[NewMessage.Event, MessageEdited.Event]):
    reply_message = await event.message.get_reply_message()
    if reply_message.file.mime_type in ["image/png", "image/jpg", "image/jpeg"]:
        await generate_img2img(event, reply_message, text=event.message.text)


async def clip(event: NewMessage.Event):
    generator_client = GeneratorClient(event.chat_id).image
    img = await event.message.download_media()
    with open(img, "rb") as file:
        b64img = base64.b64encode(file.read()).decode("utf-8")

    # result = await generator_client.interrogate_post(b64img, "deepdanbooru")
    result = await generator_client.interrogate_post(b64img, "clip")
    await event.respond(str(result))


async def png_info(event: NewMessage.Event):
    generator_client = GeneratorClient(event.chat_id).image
    img = await event.message.download_media()
    with open(img, "rb") as file:
        b64img = base64.b64encode(file.read()).decode("utf-8")

    result = await generator_client.png_info_post(b64img)
    if result.info:
        await event.respond(str(result.info))
    else:
        await event.respond("no info")


async def generate_img2img(event, message, text: str = None):
    buttons = [Button.inline("Regen"), Button.inline("File")]
    generator_client = GeneratorClient(event.chat_id).image
    payload = generator_client.img2img_payload
    payload.force_task_id = uuid.uuid4().hex

    if not message.media:
        await event.respond("no message media!")
        return

    img = await event.client.download_image(message)
    print(img)
    payload.prompt, payload.negative_prompt = GeneratorClient.txt2prompt(text)

    with open(img, "rb") as file:
        b64img = base64.b64encode(file.read()).decode("utf-8")
    init_images = [
        b64img,
    ]

    payload.init_images = init_images

    payload_info = (
        f"`/p {payload.prompt}`\n"
        if not payload.negative_prompt
        else f"`/p {payload.prompt}` `/n {payload.negative_prompt}`\n"
    )
    payload_info = f"**img2img:** \n" + payload_info

    for k in generator_client.img2img_info():
        payload_info += (
            f"`/img2img {k} {str(getattr(generator_client.txt2img_payload, k))}` \n"
        )

    async with asyncio.TaskGroup() as tg:
        generate = tg.create_task(generator_client.img2img(payload))
        progress = tg.create_task(
            update_progress(message, payload.force_task_id, payload_info)
        )

    info = json.loads(generate.result().info)
    payload_info += f"`/txt2img seed {info['all_seeds']}`\n"

    upload_files = [
        await event.client.upload_file(
            base64.b64decode(image),
            file_name=f"txt2img-{utils.timestamp()}.png",
        )
        for image in generate.result().images
    ]

    if len(upload_files) > 1 or (not progress.result().photo):
        # print(info_texts)
        edit_msg = await progress.result().edit(
            text=f"{payload_info}",
            buttons=buttons,
        )
        photoalbum = await event.respond(
            file=upload_files,
            reply_to=edit_msg.id,
        )
    else:
        edit_msg = await progress.result().edit(
            file=upload_files[0],
            text=f"{payload_info}",
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
    buttons = [
        [
            Button.text("/menu", resize=True, single_use=True),
            Button.text("/sd_model", resize=True, single_use=True),
            Button.text("/txt2img", resize=True, single_use=True),
            Button.text("/img2img", resize=True, single_use=True),
        ],
        [
            Button.text("/extras", resize=True, single_use=True),
            Button.text("/meminfo", resize=True, single_use=True),
            Button.text("/lora", resize=True, single_use=True),
        ],
    ]

    await event.respond(message=text, buttons=buttons)


async def clear_buttons(event: NewMessage.Event):
    await event.respond(message="Buttons cleared", buttons=Button.clear())


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
            generator_client = GeneratorClient(event.chat_id).image

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
    generator_client = GeneratorClient(event.chat_id).image
    message: str = event.message.text
    option = message.split()
    try:
        new_options = {option[1]: option[2]}
        generator_client.set_options(new_options)
        await event.respond("ok")
    except Exception as e:
        await event.respond(message=str(e))


async def menu_stable_diffusion_checkpoint(
    event, generator_client=None, data_str: str = None
):
    if not generator_client:
        generator_client = GeneratorClient(event.chat_id).image
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
        if isinstance(event, NewMessage.Event):
            await event.respond(message=text, buttons=buttons)
        else:
            await event.edit(text=text, buttons=buttons)


async def menu_txt2img(event, generator_client=None):
    text = f"Menu/txt2img:\n"
    text += f"Enter settings in following format:\n"
    if not generator_client:
        generator_client = GeneratorClient(event.chat_id).image
    for k in generator_client.txt2img_settings():
        text += f"`/txt2img {k} `: {getattr(generator_client.txt2img_payload, k)} \n"
    sampler_name = generator_client.txt2img_payload.sampler_name
    buttons = [
        [Button.inline(f"{sampler_name=}")],
        [Button.inline("Back")],
    ]
    if isinstance(generator_client, ForgeClient):
        scheduler = generator_client.txt2img_payload.scheduler
        buttons.insert(1, [Button.inline(f"{scheduler=}")])
    if isinstance(event, CallbackQuery.Event):
        await event.edit(text=text, buttons=buttons)
    else:
        await event.respond(message=text, buttons=buttons)


async def menu_txt2img_sampler_name(event, generator_client, data_str: str = None):
    if data_str:
        generator_client.txt2img_payload.sampler_name = data_str
        return await menu_txt2img(event, generator_client)
    text = f"**Menu/txt2img/sampler_name:**\n"
    current_sampler = generator_client.txt2img_payload.sampler_name
    samplers = generator_client.get_samplers()
    buttons = TelegramBot.button_inline_list(samplers)
    buttons.append([Button.inline("Back")])
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    else:
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
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    else:
        await event.edit(text=text, buttons=buttons)


async def set_txt2img_payload(event: NewMessage.Event):
    generator_client = GeneratorClient(event.chat_id).image
    message: str = event.message.text
    option = message.split()
    try:
        setattr(generator_client.txt2img_payload, option[1], option[2])
        print(type(generator_client.txt2img_payload.__getattribute__(option[1])))
    except (AttributeError, TypeError, ValueError) as e:
        await event.respond(message=str(e))


async def menu_img2img(event, generator_client=None):
    text = f"Menu/img2img:\n"
    text += f"Enter settings in following format:\n"
    if not generator_client:
        generator_client = GeneratorClient(event.chat_id).image
    for k in generator_client.img2img_settings():
        text += f"`/img2img {k} `: {getattr(generator_client.img2img_payload, k)} \n"
    sampler_name = generator_client.img2img_payload.sampler_name
    buttons = [
        [Button.inline(f"{sampler_name=}")],
        [Button.inline("Back")],
    ]
    if isinstance(generator_client, ForgeClient):
        scheduler = generator_client.img2img_payload.scheduler
        buttons.insert(1, [Button.inline(f"{scheduler=}")])
    if isinstance(event, CallbackQuery.Event):
        await event.edit(text=text, buttons=buttons)
    else:
        await event.respond(message=text, buttons=buttons)


async def menu_img2img_sampler_name(event, generator_client, data_str: str = None):
    if data_str:
        generator_client.img2img_payload.sampler_name = data_str
        return await menu_img2img(event, generator_client)
    text = f"**Menu/img2img/sampler_name:**\n"
    current_sampler = generator_client.img2img_payload.sampler_name
    samplers = generator_client.get_samplers()
    buttons = TelegramBot.button_inline_list(samplers)
    buttons.append([Button.inline("Back")])
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    else:
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
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    else:
        await event.edit(text=text, buttons=buttons)


async def set_img2img_payload(event: NewMessage.Event):
    generator_client = GeneratorClient(event.chat_id).image
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
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    else:
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
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    else:
        await event.edit(text=text, buttons=buttons)


async def menu_lora(event, generator_client):
    text = f"**Menu/lora:**\n\n"
    buttons = TelegramBot.button_inline_list(generator_client.lora())
    buttons.append([Button.inline("Back")])
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    else:
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
    if isinstance(event, NewMessage.Event):
        await event.respond(message=text, buttons=buttons)
    else:
        await event.edit(text=text, buttons=buttons)


async def regen(event):
    await event.answer("ok")
    message = await event.get_message()
    message_title = TelegramBot.message_head(message.text)
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


async def callback_query_handler(event: CallbackQuery.Event):
    generator_client = GeneratorClient(event.chat_id).image
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
    config = utils.load_config()
    logging.basicConfig()
    # logger = logging.getLogger(os.path.basename(__file__)).getChild(__class__.__name__)
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(config["log_level"])

    telegram_client = TelegramBot(config["telegram"])
    await telegram_client.start(bot_token=config["telegram"]["bot_token"])
    allow_chats = telegram_client.allow_chats

    telegram_client.add_event_handler(
        callback=start_command,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="(?i)/start$"),
    )

    telegram_client.add_event_handler(
        callback=clear_buttons,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="(?i)/clear$"),
    )

    telegram_client.add_event_handler(
        callback=menu,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="(?i)/menu$"),
    )

    telegram_client.add_event_handler(
        callback=menu_txt2img,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="(?i)/txt2img$"),
    )

    telegram_client.add_event_handler(
        callback=menu_img2img,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="(?i)/img2img$"),
    )

    telegram_client.add_event_handler(
        callback=generate_txt2img,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="/p\s(.*)$"),
    )

    telegram_client.add_event_handler(
        callback=generate_txt2img,
        event=MessageEdited(chats=allow_chats, incoming=True, pattern="/img\s(.*)$"),
    )

    telegram_client.add_event_handler(
        callback=set_txt2img_payload,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="/txt2img\s(.*)$"),
    )

    telegram_client.add_event_handler(
        callback=set_img2img_payload,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="/img2img\s(.*)$"),
    )

    telegram_client.add_event_handler(
        callback=options,
        event=NewMessage(chats=allow_chats, incoming=True, pattern="/options\s(.*)$"),
    )

    telegram_client.add_event_handler(
        # callback=generate_img2img,
        callback=clip,
        event=NewMessage(chats=allow_chats, incoming=True, func=lambda e: e.photo),
    )

    telegram_client.add_event_handler(
        callback=png_info,
        event=NewMessage(
            chats=allow_chats, incoming=True, func=TelegramBot.media_is_png
        ),
    )

    telegram_client.add_event_handler(
        callback=check_reply,
        event=NewMessage(
            chats=allow_chats, incoming=True, func=lambda e: e.message.is_reply
        ),
    )

    telegram_client.add_event_handler(
        callback_query_handler, CallbackQuery(chats=allow_chats)
    )

    logger.info("Generator Bot is active.")

    await telegram_client.run_until_disconnected()


if __name__ == "__main__":
    asyncio.run(main())
