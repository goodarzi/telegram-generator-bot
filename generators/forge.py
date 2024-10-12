import os
import logging
import asyncio
import pprint
import json
from httpx import BasicAuth
from typing import Union

# from .utils import utils

from .stable_diffusion_webui_forge_client.client import Client as ForgeBaseClient
from .stable_diffusion_webui_forge_client.client import (
    AuthenticatedClient as WebuiAuthenticatedBaseClient,
)
from .stable_diffusion_webui_forge_client.types import Response, HTTPStatus


from .stable_diffusion_webui_forge_client.api.default import (
    field_lambda_internal_ping_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_pending_tasks_internal_pending_tasks_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    download_sysinfo_internal_sysinfo_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    field_lambda_internal_sysinfo_download_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    progressapi_internal_progress_post,
)

from .stable_diffusion_webui_forge_client.api.default import (
    text2imgapi_sdapi_v1_txt2img_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    img2imgapi_sdapi_v1_img2img_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    extras_single_image_api_sdapi_v1_extra_single_image_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    extras_batch_images_api_sdapi_v1_extra_batch_images_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    pnginfoapi_sdapi_v1_png_info_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    progressapi_sdapi_v1_progress_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    interrogateapi_sdapi_v1_interrogate_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    interruptapi_sdapi_v1_interrupt_post,
)
from .stable_diffusion_webui_forge_client.api.default import skip_sdapi_v1_skip_post
from .stable_diffusion_webui_forge_client.api.default import (
    get_config_sdapi_v1_options_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    set_config_sdapi_v1_options_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_cmd_flags_sdapi_v1_cmd_flags_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_samplers_sdapi_v1_samplers_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_upscalers_sdapi_v1_upscalers_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_latent_upscale_modes_sdapi_v1_latent_upscale_modes_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_sd_models_sdapi_v1_sd_models_get,
)

from .stable_diffusion_webui_forge_client.api.default import (
    get_hypernetworks_sdapi_v1_hypernetworks_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_face_restorers_sdapi_v1_face_restorers_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_realesrgan_models_sdapi_v1_realesrgan_models_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_prompt_styles_sdapi_v1_prompt_styles_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_embeddings_sdapi_v1_embeddings_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    refresh_embeddings_sdapi_v1_refresh_embeddings_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    refresh_checkpoints_sdapi_v1_refresh_checkpoints_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    refresh_vae_sdapi_v1_refresh_vae_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    create_embedding_sdapi_v1_create_embedding_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    create_hypernetwork_sdapi_v1_create_hypernetwork_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_memory_sdapi_v1_memory_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    unloadapi_sdapi_v1_unload_checkpoint_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    reloadapi_sdapi_v1_reload_checkpoint_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_scripts_list_sdapi_v1_scripts_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_script_info_sdapi_v1_script_info_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_extensions_list_sdapi_v1_extensions_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_loras_sdapi_v1_loras_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    refresh_loras_sdapi_v1_refresh_loras_post,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_sd_vaes_and_text_encoders_sdapi_v1_sd_modules_get,
)
from .stable_diffusion_webui_forge_client.api.default import (
    get_schedulers_sdapi_v1_schedulers_get,
)

from .stable_diffusion_webui_forge_client.models import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    ExtrasSingleImageRequest,
    ExtrasBatchImagesRequest,
    PNGInfoRequest,
    InterrogateRequest,
    SetConfigSdapiV1OptionsPostReq,
    CreateEmbeddingSdapiV1CreateEmbeddingPostArgs,
    CreateHypernetworkSdapiV1CreateHypernetworkPostArgs,
    StableDiffusionProcessingTxt2ImgAlwaysonScripts,
    StableDiffusionProcessingImg2ImgAlwaysonScripts,
    SDModelItem,
    Options,
)

from .stable_diffusion_webui_forge_client.models import (
    StableDiffusionProcessingTxt2Img as WebuiTxt2Img,
)
from .stable_diffusion_webui_forge_client.models import (
    StableDiffusionProcessingImg2Img as WebuiImg2Img,
)
from .stable_diffusion_webui_forge_client.models import (
    StableDiffusionProcessingTxt2ImgAlwaysonScripts as WebuiTxt2ImgAlwaysonScripts,
)
from .stable_diffusion_webui_forge_client.models import (
    ProgressRequest as WebuiProgressRequest,
)
from .stable_diffusion_webui_forge_client.models import (
    StableDiffusionProcessingImg2ImgOverrideSettingsType0,
)
import os
import base64
from collections.abc import AsyncIterator


class ForgeClient:

    instances = []

    def __new__(cls, **kwargs):
        if len(ForgeClient.instances) == 0:
            self = object.__new__(cls)
            ForgeClient.instances.append(self)
            return self
        else:
            chat_id = kwargs["chat_id"]
            for inst in ForgeClient.instances:
                if hasattr(inst, "chat_id"):
                    if inst.chat_id == chat_id:
                        return inst

            self = object.__new__(cls)
            ForgeClient.instances.append(self)
            return self

    def __init__(self, base_url: str, out_dir: str, auth: dict = None, **kwargs):
        if os.path.exists(out_dir):
            self.out_dir = out_dir
        else:
            os.mkdir(out_dir)

        self.out_dir_t2i = os.path.join(self.out_dir, "txt2img")
        self.img2img_dir = os.path.join(self.out_dir, "img2img")
        self.base_url = base_url
        if auth:
            if {"username", "password"} <= auth.keys():
                self.httpx_args = {
                    "auth": BasicAuth(
                        username=auth["username"], password=auth["password"]
                    )
                }
        else:
            self.httpx_args = {}

        if not hasattr(self, "id"):
            if hasattr(ForgeClient.instances[-1], "id"):
                self.id = ForgeClient.instances[-1].id + 1
            else:
                self.id = 1
            self.chat_id: int = kwargs["chat_id"]

            logging.basicConfig()
            self.logger = logging.getLogger(os.path.basename(__file__)).getChild(
                __class__.__name__
            )
            if "log_level" in kwargs:
                self.logger.setLevel(kwargs["log_level"])

            self.tg_msg_id_input_files = {}
            self.img2img_payload = WebuiImg2Img(
                styles=[], sampler_name="", scheduler=""
            )
            self.txt2img_payload = WebuiTxt2Img(
                styles=[], sampler_name="", scheduler=""
            )

            self.preset_list = ("sd", "xl", "flux", "all", "lightning", "hyper")
            self.preset = "lightning"

            self.set_preset(self.preset)

    @property
    def options(self) -> Options:
        return self.options_get()

    @staticmethod
    def encode_file_to_base64(path):
        with open(path, "rb") as file:
            return base64.b64encode(file.read()).decode("utf-8")

    @staticmethod
    async def iter_to_base64(iterator: AsyncIterator):
        bytes_arr = bytearray()
        async for chunk in iterator:
            bytes_arr.extend(chunk)
        return base64.b64encode(bytes_arr).decode("utf-8")

    async def txt2img_post(self, body: StableDiffusionProcessingTxt2Img):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await text2imgapi_sdapi_v1_txt2img_post.asyncio(
                client=client, body=body
            )

    async def img2img_post(self, body: StableDiffusionProcessingImg2Img):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await img2imgapi_sdapi_v1_img2img_post.asyncio(
                client=client, body=body
            )

    async def extra_single_image_post(self, body: ExtrasSingleImageRequest):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return (
                await extras_single_image_api_sdapi_v1_extra_single_image_post.asyncio(
                    client=client, body=body
                )
            )

    async def extra_batch_images_post(self, body: ExtrasBatchImagesRequest):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return (
                await extras_batch_images_api_sdapi_v1_extra_batch_images_post.asyncio(
                    client=client, body=body
                )
            )

    async def png_info_post(self, image: str):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            body = PNGInfoRequest(image)
            result = await pnginfoapi_sdapi_v1_png_info_post.asyncio(
                client=client, body=body
            )
            return result

    def progress_get(self, skip_current_image: bool = False):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return progressapi_sdapi_v1_progress_get.sync(
                client=client, skip_current_image=skip_current_image
            )

    async def interrogate_post(self, image: str, model: str = "clip"):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            body = InterrogateRequest(image, model)
            result = await interrogateapi_sdapi_v1_interrogate_post.asyncio(
                client=client, body=body
            )
            if result:
                return result["caption"]

    def interrupt_post(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return interruptapi_sdapi_v1_interrupt_post.sync_detailed(client=client)

    def skip_post(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return skip_sdapi_v1_skip_post.sync_detailed(client=client)

    def options_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_config_sdapi_v1_options_get.sync(client=client)

    def options_post(self, body: dict):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return set_config_sdapi_v1_options_post.sync(
                client=client, body=SetConfigSdapiV1OptionsPostReq.from_dict(body)
            )

    def cmd_flags_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_cmd_flags_sdapi_v1_cmd_flags_get.sync(client=client)

    def samplers_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_samplers_sdapi_v1_samplers_get.sync(client=client)

    def upscalers_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_upscalers_sdapi_v1_upscalers_get.sync(client=client)

    def latent_upscale_modes_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_latent_upscale_modes_sdapi_v1_latent_upscale_modes_get.sync(
                client=client
            )

    def sd_models_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_sd_models_sdapi_v1_sd_models_get.sync(client=client)

    def hypernetworks_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_hypernetworks_sdapi_v1_hypernetworks_get.sync(client=client)

    def face_restorers_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_face_restorers_sdapi_v1_face_restorers_get.sync(client=client)

    def realesrgan_models_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_realesrgan_models_sdapi_v1_realesrgan_models_get.sync(
                client=client
            )

    def prompt_styles_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_prompt_styles_sdapi_v1_prompt_styles_get.sync(client=client)

    def embeddings_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_embeddings_sdapi_v1_embeddings_get.sync(client=client)

    async def refresh_embeddings_post(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await refresh_embeddings_sdapi_v1_refresh_embeddings_post.asyncio_detailed(
                client=client
            )

    async def refresh_checkpoints_post(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await refresh_checkpoints_sdapi_v1_refresh_checkpoints_post.asyncio_detailed(
                client=client
            )

    async def refresh_vae_post(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await refresh_vae_sdapi_v1_refresh_vae_post.asyncio_detailed(
                client=client
            )

    async def create_embedding_post(
        self, body: CreateEmbeddingSdapiV1CreateEmbeddingPostArgs
    ):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await create_embedding_sdapi_v1_create_embedding_post.asyncio(
                client=client, body=body
            )

    async def create_hypernetwork_post(
        self, body: CreateHypernetworkSdapiV1CreateHypernetworkPostArgs
    ):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await create_hypernetwork_sdapi_v1_create_hypernetwork_post.asyncio(
                client=client, body=body
            )

    def memory_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_memory_sdapi_v1_memory_get.sync(client=client)

    async def unload_checkpoint_post(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await unloadapi_sdapi_v1_unload_checkpoint_post.asyncio_detailed(
                client=client
            )

    async def reload_checkpoint_post(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await reloadapi_sdapi_v1_reload_checkpoint_post.asyncio_detailed(
                client=client
            )

    def scripts_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_scripts_list_sdapi_v1_scripts_get.sync(client=client)

    def script_info_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_script_info_sdapi_v1_script_info_get.sync(client=client)

    def extensions_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_extensions_list_sdapi_v1_extensions_get.sync(client=client)

    def loras_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_loras_sdapi_v1_loras_get.sync_detailed(client=client)

    async def refresh_loras_post(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await refresh_loras_sdapi_v1_refresh_loras_post.asyncio_detailed(
                client=client
            )

    async def progress_post(self, body: WebuiProgressRequest):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await progressapi_internal_progress_post.asyncio_detailed(
                client=client, body=body
            )

    def sd_modules_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_sd_vaes_and_text_encoders_sdapi_v1_sd_modules_get.sync(
                client=client
            )

    def schedulers_get(self):
        client = ForgeBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_schedulers_sdapi_v1_schedulers_get.sync(client=client)

    #
    #

    async def txt2img(self, payload):
        txt2img_result = await self.txt2img_post(body=payload)
        return txt2img_result

    async def img2img(self, payload):
        txt2img_result = await self.img2img_post(body=payload)
        return txt2img_result

    async def progress_current(self, progress_request):
        progress_response = await self.progress_post(body=progress_request)
        return json.loads(progress_response.content)

    async def task_progress(self, id_task=None):
        live_previews_enable = self.options.live_previews_enable
        refresh_period = (
            self.options.live_preview_refresh_period / 1000
            if self.options.live_preview_refresh_period > 0
            else 1
        )
        progress_request = WebuiProgressRequest(
            id_task=id_task, live_preview=live_previews_enable
        )
        last_progress = 0.0
        last_live_preview_id = None
        last_live_preview = None
        while True:
            # progress = await self.progress_current(progress_request)
            response = await self.progress_post(progress_request)
            progress: dict = json.loads(response.content)
            # if (not progress["active"]) and (not progress["queued"]):
            #     break
            if (
                progress["active"]
                and live_previews_enable
                and (progress["id_live_preview"] > -1)
            ):
                if progress["id_live_preview"] != last_live_preview_id:
                    last_live_preview_id = progress["id_live_preview"]
                    live_preview = progress["live_preview"].split(",")
                    del progress["live_preview"]
                    if len(live_preview) > 1:
                        live_preview = [
                            f'{id_task}-{progress["id_live_preview"]}.{self.options.live_previews_image_format}',
                            base64.b64decode(live_preview[1]),
                        ]
                    else:
                        live_preview = None
                else:
                    live_preview = None
            else:
                live_preview = None

            progress_current = self.progress_get()
            pprint.pprint(progress_current.progress)
            pprint.pprint(progress_current.state)
            pprint.pprint(self.options.live_preview_refresh_period)
            if progress["completed"]:
                yield (1.0, live_preview, progress["textinfo"])
                break
            elif progress["progress"] != last_progress:
                last_progress = progress["progress"]
                if progress["queued"]:
                    yield (0.0, None, progress["textinfo"])
                elif progress["progress"] > 0:
                    yield (progress["progress"], live_preview, progress["textinfo"])

            await asyncio.sleep(refresh_period)

    def get_memory(self):
        meminfo = self.memory_get()
        ram = meminfo.ram.additional_properties
        if "system" in meminfo.cuda.additional_properties:
            cuda = meminfo.cuda.additional_properties["system"]
        else:
            cuda = {"free": None, "used": None, "total": None}
        result = {
            "ram": ram,
            "cuda": cuda,
        }
        return result

    def get_samplers(self):
        response = self.samplers_get()
        return [i.name for i in response]

    def get_schedulers(self):
        response = self.schedulers_get()
        return [i.name for i in response]

    def lora(self):
        response: Response = self.loras_get()
        loras = json.loads(response.content)
        lora_list = []
        for i in loras:
            lora_list.append(i["name"])
        return lora_list

    def model(self, model: str = None, current: bool = False):
        if model:
            option_payload = {
                "sd_model_checkpoint": model,
            }
            self.options_post(body=option_payload)
            return
        elif current:
            options = self.options_get()
            return options.sd_model_checkpoint
        else:
            models = self.sd_models_get()
            return [i.model_name for i in models]

    async def forge_options(self, option, val):
        if hasattr(Options, option):
            option_payload = {option: val}
            result = await self.options_post(body=option_payload)
            return result.status_code

    def txt2img_info(self):
        payload = [
            "sampler_name",
            "scheduler",
            "steps",
            "cfg_scale",
            "width",
            "height",
            "seed",
            "styles",
        ]
        if self.options.forge_preset in ["flux", "all"]:
            payload.insert(5, "distilled_cfg_scale")
        return payload

    def img2img_info(self):
        payload = [
            "sampler_name",
            "steps",
            "width",
            "height",
            "cfg_scale",
            "denoising_strength",
            "styles",
        ]
        return payload

    def txt2img_settings(self):
        payload = [
            "sampler_name",
            "scheduler",
            "steps",
            "width",
            "height",
            "cfg_scale",
            "n_iter",
            "batch_size",
            "seed",
            "styles",
        ]
        if self.options.forge_preset in ["flux", "all"]:
            payload.insert(5, "distilled_cfg_scale")
        return payload

    def img2img_settings(self):
        inpaint = [
            "mask_blur",
            "inpaint_full_res",
            "inpaint_full_res_padding",
            "inpainting_mask_invert",
        ]
        # resize_mode = ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
        # resize_mode.index("Just resize")
        payload = [
            "resize_mode",
            "sampler_name",
            "scheduler",
            "steps",
            "width",
            "height",
            "cfg_scale",
            "n_iter",
            "batch_size",
            "denoising_strength",
            "seed",
            "styles",
        ]
        if self.options.forge_preset in ["flux", "all"]:
            payload.insert(5, "distilled_cfg_scale")
        return payload

    def sd_modules(self, module=None, current=False):
        options: Options = self.options_get()
        selected_modules: list = options.forge_additional_modules
        avail_modules = self.sd_modules_get()
        if current:
            current_modules_names = [
                m.model_name for m in avail_modules if m.filename in selected_modules
            ]
            return current_modules_names
        elif module:
            module_files = [m.filename for m in avail_modules if module == m.model_name]
            if module_files:
                if module_files[0] in selected_modules:
                    selected_modules.remove(module_files[0])
                else:
                    selected_modules.append(module_files[0])
                self.options_post({"forge_additional_modules": selected_modules})
                return "ok"
        else:
            return [i.model_name for i in avail_modules]

    def styles(self, style=None, current=False):
        avail_styles = self.prompt_styles_get()
        selected_styles: list = (
            []
            if isinstance(self.txt2img_payload.styles, Unset)
            else self.txt2img_payload.styles
        )

        if current:
            return [s for s in selected_styles]
        elif style:
            if style in selected_styles:
                selected_styles.remove(style)
            else:
                selected_styles.append(style)
            self.txt2img_payload.styles = selected_styles
            return "ok"
        else:
            return [i.name for i in avail_styles]

    def set_options(self, new_options: dict):
        options = self.options_get()
        options_payload = {}
        for k, v in new_options.items():
            k_lower = k.lower()
            if not hasattr(options, k_lower):
                raise AttributeError(
                    "%s attribute not found on %s"
                    % (
                        k,
                        type(options).__name__,
                    )
                )
            attr_type = type(getattr(options, k_lower))
            value_intype = attr_type(v)
            options_payload[k] = value_intype
        self.options_post(options_payload)

    def set_txt2img_payload(self, k, v):
        print(type(self.txt2img_payload.cfg_scale))
        k_lower = k.lower()
        if not hasattr(self.txt2img_payload, k_lower):
            raise AttributeError(
                "%s attribute not found on %s"
                % (
                    k,
                    type(self.txt2img_payload).__name__,
                )
            )
        attr_type = type(getattr(self.txt2img_payload, k_lower))
        value_intype = attr_type(v)
        self.txt2img_payload = value_intype

    def set_preset(self, preset: str):
        print(f"{self.id=}")
        for inst in ForgeClient.instances:
            if hasattr(inst, "chat_id"):
                print(f"{inst.chat_id=}")

        new_options = {"CLIP_stop_at_last_layers": 1}

        if preset == "sd":
            self.txt2img_payload.height = 640
            self.txt2img_payload.width = 512
            self.txt2img_payload.cfg_scale = 7.0
            self.txt2img_payload.sampler_name = "Euler a"
            self.txt2img_payload.scheduler = "Automatic"
            self.txt2img_payload.steps = 20

            self.img2img_payload.resize_mode = 1
            self.img2img_payload.cfg_scale = 7.0
            self.img2img_payload.sampler_name = "Euler a"
            self.img2img_payload.scheduler = "Automatic"
            self.img2img_payload.steps = 20
            self.preset = "sd"
            new_options["forge_preset"] = "sd"

        elif preset == "xl":
            self.txt2img_payload.height = 1152
            self.txt2img_payload.width = 896
            self.txt2img_payload.cfg_scale = 5.0
            self.txt2img_payload.sampler_name = "DPM++ 2M SDE"
            self.txt2img_payload.scheduler = "Karras"
            self.txt2img_payload.steps = 20

            self.img2img_payload.resize_mode = 1
            self.img2img_payload.height = 1152
            self.img2img_payload.width = 896
            self.img2img_payload.cfg_scale = 5.0
            self.img2img_payload.sampler_name = "DPM++ 2M SDE"
            self.img2img_payload.scheduler = "Karras"
            self.img2img_payload.steps = 20
            self.preset = "xl"
            new_options["forge_preset"] = "xl"

        elif preset == "flux":
            self.txt2img_payload.height = 1152
            self.txt2img_payload.width = 896
            self.txt2img_payload.cfg_scale = 1.0
            self.txt2img_payload.sampler_name = "Euler"
            self.txt2img_payload.scheduler = "Simple"
            self.txt2img_payload.steps = 20

            self.img2img_payload.resize_mode = 1
            self.img2img_payload.height = 1152
            self.img2img_payload.width = 896
            self.img2img_payload.cfg_scale = 1.0
            self.img2img_payload.sampler_name = "Euler"
            self.img2img_payload.scheduler = "Simple"
            self.img2img_payload.steps = 20
            self.preset = "flux"
            new_options["forge_preset"] = "flux"
        elif preset == "all":
            self.txt2img_payload.sampler_name = "DPM++ 2M"
            self.txt2img_payload.scheduler = "Automatic"
            self.txt2img_payload.steps = 20

            self.img2img_payload.resize_mode = 1.0
            self.img2img_payload.sampler_name = "DPM++ 2M"
            self.img2img_payload.scheduler = "Automatic"
            self.img2img_payload.steps = 20
            self.preset = "all"
            new_options["forge_preset"] = "all"
        elif preset == "lightning":
            self.txt2img_payload.height = 1152
            self.txt2img_payload.width = 896
            self.txt2img_payload.cfg_scale = 1.0
            self.txt2img_payload.sampler_name = "DPM++ SDE"
            self.txt2img_payload.scheduler = "Karras"
            self.txt2img_payload.steps = 5

            self.img2img_payload.height = 1152
            self.img2img_payload.width = 896
            self.img2img_payload.resize_mode = 1.0
            self.img2img_payload.sampler_name = "DPM++ SDE"
            self.img2img_payload.scheduler = "Karras"
            self.img2img_payload.steps = 5
            self.preset = "lightning"
            new_options["forge_preset"] = "xl"
        elif preset == "hyper":
            self.txt2img_payload.height = 1216
            self.txt2img_payload.width = 832
            self.txt2img_payload.cfg_scale = 1.0
            self.txt2img_payload.sampler_name = "DPM++ SDE"
            self.txt2img_payload.scheduler = "Karras"
            self.txt2img_payload.steps = 6

            self.img2img_payload.height = 1216
            self.img2img_payload.width = 832
            self.img2img_payload.resize_mode = 1.0
            self.img2img_payload.sampler_name = "DPM++ SDE"
            self.img2img_payload.scheduler = "Karras"
            self.img2img_payload.steps = 5
            self.preset = "hyper"
            new_options["forge_preset"] = "xl"

        self.set_options(new_options)
