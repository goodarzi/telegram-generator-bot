import os, logging, base64, uuid, asyncio, json
from typing import Literal, Union

from . import ForgeClient, WebuiClient
from . import utils

GEN_IMAGE = Literal["forge", "webui"]
GEN_TEXT = Literal[None]


class GeneratorClient:

    instances = []

    def __new__(cls, chat_id: int, **kwargs):
        if len(GeneratorClient.instances) == 0:
            self = object.__new__(cls)
            GeneratorClient.instances.append(self)
            return self
        else:
            # chat_id = kwargs["chat_id"]
            for inst in GeneratorClient.instances:
                if hasattr(inst, "chat_id"):
                    if inst.chat_id == chat_id:
                        return inst

            self = object.__new__(cls)
            GeneratorClient.instances.append(self)
            return self

    def __init__(self, chat_id: int, **kwargs):
        if not hasattr(self, "id"):
            if hasattr(GeneratorClient.instances[-1], "id"):
                self.id = GeneratorClient.instances[-1].id + 1
            else:
                self.id = 1
            # self.chat_id: int = kwargs["chat_id"]
            self.chat_id = chat_id

            all_config = utils.load_config()
            config = all_config["generators"]
            self.config = config
            self.gen_image: GEN_IMAGE = config["image"]

            log_level = config["log_level"] if "log_level" in config else 2
            logging.basicConfig()
            self.logger = logging.getLogger(os.path.basename(__file__)).getChild(
                __class__.__name__
            )
            self.logger.setLevel(log_level)

            # if not os.path.exists(out_dir):
            #     if os.path.exists(os.path.dirname(out_dir)):
            #         os.mkdir(out_dir)
            #     else:
            #         out_dir = os.path.join(os.path.realpath(""), out_dir)
            #         if not os.path.exists(out_dir):
            #             os.mkdir(out_dir)

    @property
    def image(self) -> Union[ForgeClient, WebuiClient]:
        if self.gen_image == "webui":
            if not "webui" in self.config:
                raise ValueError
            if not "base_url" in self.config["webui"]:
                raise ValueError
            base_url = self.config["webui"]["base_url"]
            out_dir = (
                self.config["webui"]["out_dir"]
                if "out_dir" in self.config["webui"]
                else None
            )
            auth = (
                self.config["webui"]["auth"] if "auth" in self.config["webui"] else None
            )
            return WebuiClient(
                base_url=base_url, out_dir=out_dir, auth=auth, chat_id=self.chat_id
            )
        elif self.gen_image == "forge":
            if not "forge" in self.config:
                raise ValueError
            if not "base_url" in self.config["forge"]:
                raise ValueError
            base_url = self.config["forge"]["base_url"]
            out_dir = (
                self.config["forge"]["out_dir"]
                if "out_dir" in self.config["forge"]
                else None
            )
            auth = (
                self.config["forge"]["auth"] if "auth" in self.config["forge"] else None
            )
            return ForgeClient(
                base_url=base_url, out_dir=out_dir, auth=auth, chat_id=self.chat_id
            )

    @staticmethod
    def txt2prompt(text: str):
        msg_split = text.split("\n")
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
            return (text, "")

    @staticmethod
    def save_live_image(image):
        file_path = f"/tmp/{image[0]}.png"
        with open(file_path, "wb") as output:
            output.write(base64.b64decode(image[1]))
        return file_path

    @staticmethod
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
