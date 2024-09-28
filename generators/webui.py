import os
import time
import logging
import asyncio
import pprint
import json
from httpx import BasicAuth
from typing import Union

# from .utils import utils

from .stable_diffusion_webui_client.client import Client as WebuiBaseClient
from .stable_diffusion_webui_client.client import (
    AuthenticatedClient as WebuiAuthenticatedBaseClient,
)
from .stable_diffusion_webui_forge_client.types import Response, HTTPStatus


from .stable_diffusion_webui_client.api.default import field_lambda_internal_ping_get
from .stable_diffusion_webui_client.api.default import (
    get_pending_tasks_internal_pending_tasks_get,
)
from .stable_diffusion_webui_client.api.default import (
    download_sysinfo_internal_sysinfo_get,
)
from .stable_diffusion_webui_client.api.default import (
    field_lambda_internal_sysinfo_download_get,
)
from .stable_diffusion_webui_client.api.default import (
    progressapi_internal_progress_post,
)

from .stable_diffusion_webui_client.api.default import text2imgapi_sdapi_v1_txt2img_post
from .stable_diffusion_webui_client.api.default import img2imgapi_sdapi_v1_img2img_post
from .stable_diffusion_webui_client.api.default import (
    extras_single_image_api_sdapi_v1_extra_single_image_post,
)
from .stable_diffusion_webui_client.api.default import (
    extras_batch_images_api_sdapi_v1_extra_batch_images_post,
)
from .stable_diffusion_webui_client.api.default import pnginfoapi_sdapi_v1_png_info_post
from .stable_diffusion_webui_client.api.default import progressapi_sdapi_v1_progress_get
from .stable_diffusion_webui_client.api.default import (
    interrogateapi_sdapi_v1_interrogate_post,
)
from .stable_diffusion_webui_client.api.default import (
    interruptapi_sdapi_v1_interrupt_post,
)
from .stable_diffusion_webui_client.api.default import skip_sdapi_v1_skip_post
from .stable_diffusion_webui_client.api.default import get_config_sdapi_v1_options_get
from .stable_diffusion_webui_client.api.default import set_config_sdapi_v1_options_post
from .stable_diffusion_webui_client.api.default import (
    get_cmd_flags_sdapi_v1_cmd_flags_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_samplers_sdapi_v1_samplers_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_upscalers_sdapi_v1_upscalers_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_latent_upscale_modes_sdapi_v1_latent_upscale_modes_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_sd_models_sdapi_v1_sd_models_get,
)
from .stable_diffusion_webui_client.api.default import get_sd_vaes_sdapi_v1_sd_vae_get
from .stable_diffusion_webui_client.api.default import (
    get_hypernetworks_sdapi_v1_hypernetworks_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_face_restorers_sdapi_v1_face_restorers_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_realesrgan_models_sdapi_v1_realesrgan_models_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_prompt_styles_sdapi_v1_prompt_styles_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_embeddings_sdapi_v1_embeddings_get,
)
from .stable_diffusion_webui_client.api.default import (
    refresh_embeddings_sdapi_v1_refresh_embeddings_post,
)
from .stable_diffusion_webui_client.api.default import (
    refresh_checkpoints_sdapi_v1_refresh_checkpoints_post,
)
from .stable_diffusion_webui_client.api.default import (
    refresh_vae_sdapi_v1_refresh_vae_post,
)
from .stable_diffusion_webui_client.api.default import (
    create_embedding_sdapi_v1_create_embedding_post,
)
from .stable_diffusion_webui_client.api.default import (
    create_hypernetwork_sdapi_v1_create_hypernetwork_post,
)
from .stable_diffusion_webui_client.api.default import (
    train_embedding_sdapi_v1_train_embedding_post,
)
from .stable_diffusion_webui_client.api.default import (
    train_hypernetwork_sdapi_v1_train_hypernetwork_post,
)
from .stable_diffusion_webui_client.api.default import get_memory_sdapi_v1_memory_get
from .stable_diffusion_webui_client.api.default import (
    unloadapi_sdapi_v1_unload_checkpoint_post,
)
from .stable_diffusion_webui_client.api.default import (
    reloadapi_sdapi_v1_reload_checkpoint_post,
)
from .stable_diffusion_webui_client.api.default import (
    get_scripts_list_sdapi_v1_scripts_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_script_info_sdapi_v1_script_info_get,
)
from .stable_diffusion_webui_client.api.default import (
    get_extensions_list_sdapi_v1_extensions_get,
)
from .stable_diffusion_webui_client.api.default import get_loras_sdapi_v1_loras_get
from .stable_diffusion_webui_client.api.default import (
    refresh_loras_sdapi_v1_refresh_loras_post,
)


from .stable_diffusion_webui_client.models import (
    StableDiffusionProcessingTxt2Img,
    StableDiffusionProcessingImg2Img,
    ExtrasSingleImageRequest,
    ExtrasBatchImagesRequest,
    PNGInfoRequest,
    InterrogateRequest,
    SetConfigSdapiV1OptionsPostReq,
    CreateEmbeddingSdapiV1CreateEmbeddingPostArgs,
    CreateHypernetworkSdapiV1CreateHypernetworkPostArgs,
    TrainEmbeddingSdapiV1TrainEmbeddingPostArgs,
    TrainHypernetworkSdapiV1TrainHypernetworkPostArgs,
    StableDiffusionProcessingTxt2ImgAlwaysonScripts,
    StableDiffusionProcessingTxt2ImgOverrideSettings,
    StableDiffusionProcessingImg2ImgAlwaysonScripts,
    StableDiffusionProcessingImg2ImgOverrideSettings,
    SDModelItem,
)

from .stable_diffusion_webui_client.models import (
    StableDiffusionProcessingTxt2Img as WebuiTxt2Img,
)
from .stable_diffusion_webui_client.models import (
    StableDiffusionProcessingImg2Img as WebuiImg2Img,
)
from .stable_diffusion_webui_client.models import (
    StableDiffusionProcessingTxt2ImgAlwaysonScripts as WebuiTxt2ImgAlwaysonScripts,
)
from .stable_diffusion_webui_client.models import (
    StableDiffusionProcessingTxt2ImgOverrideSettings as WebuiTxt2ImgOverrideSettings,
)
from .stable_diffusion_webui_client.models import (
    ProgressRequest as WebuiProgressRequest,
)

import os
import base64
from collections.abc import AsyncIterator


class WebuiClient:

    instances = []

    def __new__(cls, **kwargs):
        if len(WebuiClient.instances) == 0:
            self = object.__new__(cls)
            WebuiClient.instances.append(self)
            return self
        else:
            chat_id = kwargs["chat_id"]
            for inst in WebuiClient.instances:
                if hasattr(inst, "chat_id"):
                    if inst.chat_id == chat_id:
                        return inst

            self = object.__new__(cls)
            WebuiClient.instances.append(self)
            return self

    def __init__(self, base_url: str, out_dir: str, auth: BasicAuth = None, **kwargs):

        if not hasattr(self, "id"):
            if hasattr(WebuiClient.instances[-1], "id"):
                self.id = WebuiClient.instances[-1].id + 1
            else:
                self.id = 1
            self.chat_id: int = kwargs["chat_id"]
            # self.image_generator: Union[WebuiClient, ForgeClient] = kwargs['image_generator']

            logging.basicConfig()
            self.logger = logging.getLogger(os.path.basename(__file__)).getChild(
                __class__.__name__
            )
            if "log_level" in kwargs:
                self.logger.setLevel(kwargs["log_level"])

            self.txt2img_payload = WebuiTxt2Img(
                prompt="",
                negative_prompt="",
                width=832,
                height=1216,
                cfg_scale=2,
                steps=5,
                sampler_name="DPM++ SDE Karras",
            )

        if os.path.exists(out_dir):
            self.out_dir = out_dir
        else:
            os.mkdir(out_dir)
            # raise ValueError

        self.out_dir_t2i = os.path.join(self.out_dir, "txt2img")
        self.img2img_dir = os.path.join(self.out_dir, "img2img")
        self.base_url = base_url
        self.httpx_args = {"auth": auth}
        # self.client = WebuiBaseClient(base_url=base_url, httpx_args={'auth': auth})

    @staticmethod
    def txt2img_payload():
        return StableDiffusionProcessingTxt2Img(
            prompt="",
            negative_prompt="",
            width=1216,
            height=1832,
            cfg_scale=2,
            steps=5,
            sampler_name="DPM++ SDE Karras",
        )

    @staticmethod
    def img2img_payload(image):
        return StableDiffusionProcessingImg2Img(
            prompt="",
            negative_prompt="",
            width=1216,
            height=1832,
            cfg_scale=2,
            steps=5,
            sampler_name="DPM++ SDE Karras",
            init_images=[image],
        )

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
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await text2imgapi_sdapi_v1_txt2img_post.asyncio(
                client=client, body=body
            )

    async def img2img_post(self, body: StableDiffusionProcessingImg2Img):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await img2imgapi_sdapi_v1_img2img_post.asyncio(
                client=client, body=body
            )

    async def extra_single_image_post(self, body: ExtrasSingleImageRequest):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return (
                await extras_single_image_api_sdapi_v1_extra_single_image_post.asyncio(
                    client=client, body=body
                )
            )

    async def extra_batch_images_post(self, body: ExtrasBatchImagesRequest):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return (
                await extras_batch_images_api_sdapi_v1_extra_batch_images_post.asyncio(
                    client=client, body=body
                )
            )

    async def png_info_post(self, body: PNGInfoRequest):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await pnginfoapi_sdapi_v1_png_info_post.asyncio(
                client=client, body=body
            )

    def progress_get(self, skip_current_image: bool = False):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return progressapi_sdapi_v1_progress_get.sync(
                client=client, skip_current_image=skip_current_image
            )

    async def interrogateapi_sdapi_v1_interrogate_post(self, body: InterrogateRequest):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await interrogateapi_sdapi_v1_interrogate_post.asyncio(
                client=client, body=body
            )

    async def interrupt_post(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await interruptapi_sdapi_v1_interrupt_post.asyncio_detailed(
                client=client
            )

    def skip_post(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return skip_sdapi_v1_skip_post.sync_detailed(client=client)

    def options_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_config_sdapi_v1_options_get.sync(client=client)

    async def options_post(self, body: dict):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await set_config_sdapi_v1_options_post.asyncio_detailed(
                client=client, body=SetConfigSdapiV1OptionsPostReq.from_dict(body)
            )

    def cmd_flags_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_cmd_flags_sdapi_v1_cmd_flags_get.sync(client=client)

    def samplers_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_samplers_sdapi_v1_samplers_get.sync(client=client)

    def upscalers_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_upscalers_sdapi_v1_upscalers_get.sync(client=client)

    def latent_upscale_modes_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_latent_upscale_modes_sdapi_v1_latent_upscale_modes_get.sync(
                client=client
            )

    def sd_models_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_sd_models_sdapi_v1_sd_models_get.sync(client=client)

    def sd_vae_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_sd_vaes_sdapi_v1_sd_vae_get.sync(client=client)

    def hypernetworks_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_hypernetworks_sdapi_v1_hypernetworks_get.sync(client=client)

    def face_restorers_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_face_restorers_sdapi_v1_face_restorers_get.sync(client=client)

    def realesrgan_models_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_realesrgan_models_sdapi_v1_realesrgan_models_get.sync(
                client=client
            )

    def prompt_styles_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_prompt_styles_sdapi_v1_prompt_styles_get.sync(client=client)

    def embeddings_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_embeddings_sdapi_v1_embeddings_get.sync(client=client)

    async def refresh_embeddings_post(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await refresh_embeddings_sdapi_v1_refresh_embeddings_post.asyncio_detailed(
                client=client
            )

    async def refresh_checkpoints_post(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await refresh_checkpoints_sdapi_v1_refresh_checkpoints_post.asyncio_detailed(
                client=client
            )

    async def refresh_vae_post(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await refresh_vae_sdapi_v1_refresh_vae_post.asyncio_detailed(
                client=client
            )

    async def create_embedding_post(
        self, body: CreateEmbeddingSdapiV1CreateEmbeddingPostArgs
    ):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await create_embedding_sdapi_v1_create_embedding_post.asyncio(
                client=client, body=body
            )

    async def create_hypernetwork_post(
        self, body: CreateHypernetworkSdapiV1CreateHypernetworkPostArgs
    ):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await create_hypernetwork_sdapi_v1_create_hypernetwork_post.asyncio(
                client=client, body=body
            )

    async def train_embedding_post(
        self, body: TrainEmbeddingSdapiV1TrainEmbeddingPostArgs
    ):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await train_embedding_sdapi_v1_train_embedding_post.asyncio(
                client=client, body=body
            )

    async def train_hypernetwork_post(
        self, body: TrainHypernetworkSdapiV1TrainHypernetworkPostArgs
    ):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return train_hypernetwork_sdapi_v1_train_hypernetwork_post.asyncio(
                client=client, body=body
            )

    def memory_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_memory_sdapi_v1_memory_get.sync(client=client)

    async def unload_checkpoint_post(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await unloadapi_sdapi_v1_unload_checkpoint_post.asyncio_detailed(
                client=client
            )

    async def reload_checkpoint_post(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await reloadapi_sdapi_v1_reload_checkpoint_post.asyncio_detailed(
                client=client
            )

    def scripts_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_scripts_list_sdapi_v1_scripts_get.sync(client=client)

    def script_info_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_script_info_sdapi_v1_script_info_get.sync(client=client)

    def extensions_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_extensions_list_sdapi_v1_extensions_get.sync(client=client)

    def loras_get(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        with client as client:
            return get_loras_sdapi_v1_loras_get.sync_detailed(client=client)

    async def refresh_loras_post(self):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await refresh_loras_sdapi_v1_refresh_loras_post.asyncio_detailed(
                client=client
            )

    async def progress_post(self, body: WebuiProgressRequest):
        client = WebuiBaseClient(base_url=self.base_url, httpx_args=self.httpx_args)
        async with client as client:
            return await progressapi_internal_progress_post.asyncio_detailed(
                client=client, body=body
            )

    #
    #

    async def txt2img(self, payload):
        txt2img_result = await self.txt2img_post(body=payload)
        return txt2img_result

    async def progress_current(self, progress_request):
        progress_response = await self.progress_post(body=progress_request)
        return json.loads(progress_response.content)

    async def task_progress(self, id_task=None):
        # progress_request = WebuiProgressRequest(id_task=id_task, live_preview=False)
        progress_request = WebuiProgressRequest(id_task=id_task)
        progress = await self.progress_current(progress_request)
        if progress["active"]:
            last_progress = 0.0
            while True:
                await asyncio.sleep(1)
                progress = await self.progress_current(progress_request)
                pprint.pprint(progress)
                if progress["completed"]:
                    yield (1.0, progress["live_preview"])
                    break
                elif progress["progress"] != last_progress:
                    last_progress = progress["progress"]
                    yield (progress["progress"], progress["live_preview"])

    async def progress(self, task_id=None):
        last_progress = 0
        last_image = ""
        # if progress_request == 1
        while True:
            await asyncio.sleep(1)
            progress_current = self.progress_get()
            # pprint.pp(progress_current.progress)
            # pprint.pp(progress_current.state)
            # pprint.pp(progress_current.textinfo)
            # pprint.pp(progress_current.detail)
            if (not progress_current.progress) or (progress_current.progress == 1.0):
                yield (1.0, None)
                break
            if progress_current.progress != last_progress:
                last_progress = progress_current.progress
                if progress_current.current_image == last_image:
                    yield (progress_current.progress, None)
                else:
                    yield (progress_current.progress, progress_current.current_image)

    def memory(self):
        return self.memory_get()

    def txt2img_sampler(self, sampler_name: str = None):
        if sampler_name:
            self.txt2img_payload.sampler_name = sampler_name
        else:
            response = self.samplers_get()
            return [i.name for i in response]

    def lora(self):
        response: Response = self.loras_get()
        loras = json.loads(response.content)
        lora_list = []
        for i in loras:
            lora_list.append(i["name"])
        return lora_list

    async def model(self, model: str = None, current: bool = False):
        if model:
            option_payload = {
                "sd_model_checkpoint": model,
                "CLIP_stop_at_last_layers": 2,
            }
            response = await self.options_post(body=option_payload)
            return response.status_code

        if current:
            response = self.options_get()
            return response.sd_model_checkpoint

        response = self.sd_models_get()
        model_list = []
        for i in response:
            model_list.append(i.model_name)
        return model_list
