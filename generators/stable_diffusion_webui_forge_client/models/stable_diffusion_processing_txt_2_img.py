from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.stable_diffusion_processing_txt_2_img_alwayson_scripts import (
        StableDiffusionProcessingTxt2ImgAlwaysonScripts,
    )
    from ..models.stable_diffusion_processing_txt_2_img_comments_type_0 import (
        StableDiffusionProcessingTxt2ImgCommentsType0,
    )
    from ..models.stable_diffusion_processing_txt_2_img_override_settings_type_0 import (
        StableDiffusionProcessingTxt2ImgOverrideSettingsType0,
    )


T = TypeVar("T", bound="StableDiffusionProcessingTxt2Img")


@_attrs_define
class StableDiffusionProcessingTxt2Img:
    """
    Attributes:
        prompt (Union[None, Unset, str]):  Default: ''.
        negative_prompt (Union[None, Unset, str]):  Default: ''.
        styles (Union[List[str], None, Unset]):
        seed (Union[None, Unset, int]):  Default: -1.
        subseed (Union[None, Unset, int]):  Default: -1.
        subseed_strength (Union[None, Unset, float]):  Default: 0.0.
        seed_resize_from_h (Union[None, Unset, int]):  Default: -1.
        seed_resize_from_w (Union[None, Unset, int]):  Default: -1.
        sampler_name (Union[None, Unset, str]):
        scheduler (Union[None, Unset, str]):
        batch_size (Union[None, Unset, int]):  Default: 1.
        n_iter (Union[None, Unset, int]):  Default: 1.
        steps (Union[None, Unset, int]):  Default: 50.
        cfg_scale (Union[None, Unset, float]):  Default: 7.0.
        distilled_cfg_scale (Union[None, Unset, float]):  Default: 3.5.
        width (Union[None, Unset, int]):  Default: 512.
        height (Union[None, Unset, int]):  Default: 512.
        restore_faces (Union[None, Unset, bool]):
        tiling (Union[None, Unset, bool]):
        do_not_save_samples (Union[None, Unset, bool]):  Default: False.
        do_not_save_grid (Union[None, Unset, bool]):  Default: False.
        eta (Union[None, Unset, float]):
        denoising_strength (Union[None, Unset, float]):
        s_min_uncond (Union[None, Unset, float]):
        s_churn (Union[None, Unset, float]):
        s_tmax (Union[None, Unset, float]):
        s_tmin (Union[None, Unset, float]):
        s_noise (Union[None, Unset, float]):
        override_settings (Union['StableDiffusionProcessingTxt2ImgOverrideSettingsType0', None, Unset]):
        override_settings_restore_afterwards (Union[None, Unset, bool]):  Default: True.
        refiner_checkpoint (Union[None, Unset, str]):
        refiner_switch_at (Union[None, Unset, float]):
        disable_extra_networks (Union[None, Unset, bool]):  Default: False.
        firstpass_image (Union[None, Unset, str]):
        comments (Union['StableDiffusionProcessingTxt2ImgCommentsType0', None, Unset]):
        enable_hr (Union[None, Unset, bool]):  Default: False.
        firstphase_width (Union[None, Unset, int]):  Default: 0.
        firstphase_height (Union[None, Unset, int]):  Default: 0.
        hr_scale (Union[None, Unset, float]):  Default: 2.0.
        hr_upscaler (Union[None, Unset, str]):
        hr_second_pass_steps (Union[None, Unset, int]):  Default: 0.
        hr_resize_x (Union[None, Unset, int]):  Default: 0.
        hr_resize_y (Union[None, Unset, int]):  Default: 0.
        hr_checkpoint_name (Union[None, Unset, str]):
        hr_sampler_name (Union[None, Unset, str]):
        hr_scheduler (Union[None, Unset, str]):
        hr_prompt (Union[None, Unset, str]):  Default: ''.
        hr_negative_prompt (Union[None, Unset, str]):  Default: ''.
        force_task_id (Union[None, Unset, str]):
        sampler_index (Union[Unset, str]):  Default: 'Euler'.
        script_name (Union[None, Unset, str]):
        script_args (Union[Unset, List[Any]]):
        send_images (Union[Unset, bool]):  Default: True.
        save_images (Union[Unset, bool]):  Default: False.
        alwayson_scripts (Union[Unset, StableDiffusionProcessingTxt2ImgAlwaysonScripts]):
        infotext (Union[None, Unset, str]):
    """

    prompt: Union[None, Unset, str] = ""
    negative_prompt: Union[None, Unset, str] = ""
    styles: Union[List[str], None, Unset] = UNSET
    seed: Union[None, Unset, int] = -1
    subseed: Union[None, Unset, int] = -1
    subseed_strength: Union[None, Unset, float] = 0.0
    seed_resize_from_h: Union[None, Unset, int] = -1
    seed_resize_from_w: Union[None, Unset, int] = -1
    sampler_name: Union[None, Unset, str] = UNSET
    scheduler: Union[None, Unset, str] = UNSET
    batch_size: Union[None, Unset, int] = 1
    n_iter: Union[None, Unset, int] = 1
    steps: Union[None, Unset, int] = 50
    cfg_scale: Union[None, Unset, float] = 7.0
    distilled_cfg_scale: Union[None, Unset, float] = 3.5
    width: Union[None, Unset, int] = 512
    height: Union[None, Unset, int] = 512
    restore_faces: Union[None, Unset, bool] = UNSET
    tiling: Union[None, Unset, bool] = UNSET
    do_not_save_samples: Union[None, Unset, bool] = False
    do_not_save_grid: Union[None, Unset, bool] = False
    eta: Union[None, Unset, float] = UNSET
    denoising_strength: Union[None, Unset, float] = UNSET
    s_min_uncond: Union[None, Unset, float] = UNSET
    s_churn: Union[None, Unset, float] = UNSET
    s_tmax: Union[None, Unset, float] = UNSET
    s_tmin: Union[None, Unset, float] = UNSET
    s_noise: Union[None, Unset, float] = UNSET
    override_settings: Union["StableDiffusionProcessingTxt2ImgOverrideSettingsType0", None, Unset] = UNSET
    override_settings_restore_afterwards: Union[None, Unset, bool] = True
    refiner_checkpoint: Union[None, Unset, str] = UNSET
    refiner_switch_at: Union[None, Unset, float] = UNSET
    disable_extra_networks: Union[None, Unset, bool] = False
    firstpass_image: Union[None, Unset, str] = UNSET
    comments: Union["StableDiffusionProcessingTxt2ImgCommentsType0", None, Unset] = UNSET
    enable_hr: Union[None, Unset, bool] = False
    firstphase_width: Union[None, Unset, int] = 0
    firstphase_height: Union[None, Unset, int] = 0
    hr_scale: Union[None, Unset, float] = 2.0
    hr_upscaler: Union[None, Unset, str] = UNSET
    hr_second_pass_steps: Union[None, Unset, int] = 0
    hr_resize_x: Union[None, Unset, int] = 0
    hr_resize_y: Union[None, Unset, int] = 0
    hr_checkpoint_name: Union[None, Unset, str] = UNSET
    hr_sampler_name: Union[None, Unset, str] = UNSET
    hr_scheduler: Union[None, Unset, str] = UNSET
    hr_prompt: Union[None, Unset, str] = ""
    hr_negative_prompt: Union[None, Unset, str] = ""
    force_task_id: Union[None, Unset, str] = UNSET
    sampler_index: Union[Unset, str] = "Euler"
    script_name: Union[None, Unset, str] = UNSET
    script_args: Union[Unset, List[Any]] = UNSET
    send_images: Union[Unset, bool] = True
    save_images: Union[Unset, bool] = False
    alwayson_scripts: Union[Unset, "StableDiffusionProcessingTxt2ImgAlwaysonScripts"] = UNSET
    infotext: Union[None, Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        from ..models.stable_diffusion_processing_txt_2_img_comments_type_0 import (
            StableDiffusionProcessingTxt2ImgCommentsType0,
        )
        from ..models.stable_diffusion_processing_txt_2_img_override_settings_type_0 import (
            StableDiffusionProcessingTxt2ImgOverrideSettingsType0,
        )

        prompt: Union[None, Unset, str]
        if isinstance(self.prompt, Unset):
            prompt = UNSET
        else:
            prompt = self.prompt

        negative_prompt: Union[None, Unset, str]
        if isinstance(self.negative_prompt, Unset):
            negative_prompt = UNSET
        else:
            negative_prompt = self.negative_prompt

        styles: Union[List[str], None, Unset]
        if isinstance(self.styles, Unset):
            styles = UNSET
        elif isinstance(self.styles, list):
            styles = self.styles

        else:
            styles = self.styles

        seed: Union[None, Unset, int]
        if isinstance(self.seed, Unset):
            seed = UNSET
        else:
            seed = self.seed

        subseed: Union[None, Unset, int]
        if isinstance(self.subseed, Unset):
            subseed = UNSET
        else:
            subseed = self.subseed

        subseed_strength: Union[None, Unset, float]
        if isinstance(self.subseed_strength, Unset):
            subseed_strength = UNSET
        else:
            subseed_strength = self.subseed_strength

        seed_resize_from_h: Union[None, Unset, int]
        if isinstance(self.seed_resize_from_h, Unset):
            seed_resize_from_h = UNSET
        else:
            seed_resize_from_h = self.seed_resize_from_h

        seed_resize_from_w: Union[None, Unset, int]
        if isinstance(self.seed_resize_from_w, Unset):
            seed_resize_from_w = UNSET
        else:
            seed_resize_from_w = self.seed_resize_from_w

        sampler_name: Union[None, Unset, str]
        if isinstance(self.sampler_name, Unset):
            sampler_name = UNSET
        else:
            sampler_name = self.sampler_name

        scheduler: Union[None, Unset, str]
        if isinstance(self.scheduler, Unset):
            scheduler = UNSET
        else:
            scheduler = self.scheduler

        batch_size: Union[None, Unset, int]
        if isinstance(self.batch_size, Unset):
            batch_size = UNSET
        else:
            batch_size = self.batch_size

        n_iter: Union[None, Unset, int]
        if isinstance(self.n_iter, Unset):
            n_iter = UNSET
        else:
            n_iter = self.n_iter

        steps: Union[None, Unset, int]
        if isinstance(self.steps, Unset):
            steps = UNSET
        else:
            steps = self.steps

        cfg_scale: Union[None, Unset, float]
        if isinstance(self.cfg_scale, Unset):
            cfg_scale = UNSET
        else:
            cfg_scale = self.cfg_scale

        distilled_cfg_scale: Union[None, Unset, float]
        if isinstance(self.distilled_cfg_scale, Unset):
            distilled_cfg_scale = UNSET
        else:
            distilled_cfg_scale = self.distilled_cfg_scale

        width: Union[None, Unset, int]
        if isinstance(self.width, Unset):
            width = UNSET
        else:
            width = self.width

        height: Union[None, Unset, int]
        if isinstance(self.height, Unset):
            height = UNSET
        else:
            height = self.height

        restore_faces: Union[None, Unset, bool]
        if isinstance(self.restore_faces, Unset):
            restore_faces = UNSET
        else:
            restore_faces = self.restore_faces

        tiling: Union[None, Unset, bool]
        if isinstance(self.tiling, Unset):
            tiling = UNSET
        else:
            tiling = self.tiling

        do_not_save_samples: Union[None, Unset, bool]
        if isinstance(self.do_not_save_samples, Unset):
            do_not_save_samples = UNSET
        else:
            do_not_save_samples = self.do_not_save_samples

        do_not_save_grid: Union[None, Unset, bool]
        if isinstance(self.do_not_save_grid, Unset):
            do_not_save_grid = UNSET
        else:
            do_not_save_grid = self.do_not_save_grid

        eta: Union[None, Unset, float]
        if isinstance(self.eta, Unset):
            eta = UNSET
        else:
            eta = self.eta

        denoising_strength: Union[None, Unset, float]
        if isinstance(self.denoising_strength, Unset):
            denoising_strength = UNSET
        else:
            denoising_strength = self.denoising_strength

        s_min_uncond: Union[None, Unset, float]
        if isinstance(self.s_min_uncond, Unset):
            s_min_uncond = UNSET
        else:
            s_min_uncond = self.s_min_uncond

        s_churn: Union[None, Unset, float]
        if isinstance(self.s_churn, Unset):
            s_churn = UNSET
        else:
            s_churn = self.s_churn

        s_tmax: Union[None, Unset, float]
        if isinstance(self.s_tmax, Unset):
            s_tmax = UNSET
        else:
            s_tmax = self.s_tmax

        s_tmin: Union[None, Unset, float]
        if isinstance(self.s_tmin, Unset):
            s_tmin = UNSET
        else:
            s_tmin = self.s_tmin

        s_noise: Union[None, Unset, float]
        if isinstance(self.s_noise, Unset):
            s_noise = UNSET
        else:
            s_noise = self.s_noise

        override_settings: Union[Dict[str, Any], None, Unset]
        if isinstance(self.override_settings, Unset):
            override_settings = UNSET
        elif isinstance(self.override_settings, StableDiffusionProcessingTxt2ImgOverrideSettingsType0):
            override_settings = self.override_settings.to_dict()
        else:
            override_settings = self.override_settings

        override_settings_restore_afterwards: Union[None, Unset, bool]
        if isinstance(self.override_settings_restore_afterwards, Unset):
            override_settings_restore_afterwards = UNSET
        else:
            override_settings_restore_afterwards = self.override_settings_restore_afterwards

        refiner_checkpoint: Union[None, Unset, str]
        if isinstance(self.refiner_checkpoint, Unset):
            refiner_checkpoint = UNSET
        else:
            refiner_checkpoint = self.refiner_checkpoint

        refiner_switch_at: Union[None, Unset, float]
        if isinstance(self.refiner_switch_at, Unset):
            refiner_switch_at = UNSET
        else:
            refiner_switch_at = self.refiner_switch_at

        disable_extra_networks: Union[None, Unset, bool]
        if isinstance(self.disable_extra_networks, Unset):
            disable_extra_networks = UNSET
        else:
            disable_extra_networks = self.disable_extra_networks

        firstpass_image: Union[None, Unset, str]
        if isinstance(self.firstpass_image, Unset):
            firstpass_image = UNSET
        else:
            firstpass_image = self.firstpass_image

        comments: Union[Dict[str, Any], None, Unset]
        if isinstance(self.comments, Unset):
            comments = UNSET
        elif isinstance(self.comments, StableDiffusionProcessingTxt2ImgCommentsType0):
            comments = self.comments.to_dict()
        else:
            comments = self.comments

        enable_hr: Union[None, Unset, bool]
        if isinstance(self.enable_hr, Unset):
            enable_hr = UNSET
        else:
            enable_hr = self.enable_hr

        firstphase_width: Union[None, Unset, int]
        if isinstance(self.firstphase_width, Unset):
            firstphase_width = UNSET
        else:
            firstphase_width = self.firstphase_width

        firstphase_height: Union[None, Unset, int]
        if isinstance(self.firstphase_height, Unset):
            firstphase_height = UNSET
        else:
            firstphase_height = self.firstphase_height

        hr_scale: Union[None, Unset, float]
        if isinstance(self.hr_scale, Unset):
            hr_scale = UNSET
        else:
            hr_scale = self.hr_scale

        hr_upscaler: Union[None, Unset, str]
        if isinstance(self.hr_upscaler, Unset):
            hr_upscaler = UNSET
        else:
            hr_upscaler = self.hr_upscaler

        hr_second_pass_steps: Union[None, Unset, int]
        if isinstance(self.hr_second_pass_steps, Unset):
            hr_second_pass_steps = UNSET
        else:
            hr_second_pass_steps = self.hr_second_pass_steps

        hr_resize_x: Union[None, Unset, int]
        if isinstance(self.hr_resize_x, Unset):
            hr_resize_x = UNSET
        else:
            hr_resize_x = self.hr_resize_x

        hr_resize_y: Union[None, Unset, int]
        if isinstance(self.hr_resize_y, Unset):
            hr_resize_y = UNSET
        else:
            hr_resize_y = self.hr_resize_y

        hr_checkpoint_name: Union[None, Unset, str]
        if isinstance(self.hr_checkpoint_name, Unset):
            hr_checkpoint_name = UNSET
        else:
            hr_checkpoint_name = self.hr_checkpoint_name

        hr_sampler_name: Union[None, Unset, str]
        if isinstance(self.hr_sampler_name, Unset):
            hr_sampler_name = UNSET
        else:
            hr_sampler_name = self.hr_sampler_name

        hr_scheduler: Union[None, Unset, str]
        if isinstance(self.hr_scheduler, Unset):
            hr_scheduler = UNSET
        else:
            hr_scheduler = self.hr_scheduler

        hr_prompt: Union[None, Unset, str]
        if isinstance(self.hr_prompt, Unset):
            hr_prompt = UNSET
        else:
            hr_prompt = self.hr_prompt

        hr_negative_prompt: Union[None, Unset, str]
        if isinstance(self.hr_negative_prompt, Unset):
            hr_negative_prompt = UNSET
        else:
            hr_negative_prompt = self.hr_negative_prompt

        force_task_id: Union[None, Unset, str]
        if isinstance(self.force_task_id, Unset):
            force_task_id = UNSET
        else:
            force_task_id = self.force_task_id

        sampler_index = self.sampler_index

        script_name: Union[None, Unset, str]
        if isinstance(self.script_name, Unset):
            script_name = UNSET
        else:
            script_name = self.script_name

        script_args: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.script_args, Unset):
            script_args = self.script_args

        send_images = self.send_images

        save_images = self.save_images

        alwayson_scripts: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.alwayson_scripts, Unset):
            alwayson_scripts = self.alwayson_scripts.to_dict()

        infotext: Union[None, Unset, str]
        if isinstance(self.infotext, Unset):
            infotext = UNSET
        else:
            infotext = self.infotext

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if prompt is not UNSET:
            field_dict["prompt"] = prompt
        if negative_prompt is not UNSET:
            field_dict["negative_prompt"] = negative_prompt
        if styles is not UNSET:
            field_dict["styles"] = styles
        if seed is not UNSET:
            field_dict["seed"] = seed
        if subseed is not UNSET:
            field_dict["subseed"] = subseed
        if subseed_strength is not UNSET:
            field_dict["subseed_strength"] = subseed_strength
        if seed_resize_from_h is not UNSET:
            field_dict["seed_resize_from_h"] = seed_resize_from_h
        if seed_resize_from_w is not UNSET:
            field_dict["seed_resize_from_w"] = seed_resize_from_w
        if sampler_name is not UNSET:
            field_dict["sampler_name"] = sampler_name
        if scheduler is not UNSET:
            field_dict["scheduler"] = scheduler
        if batch_size is not UNSET:
            field_dict["batch_size"] = batch_size
        if n_iter is not UNSET:
            field_dict["n_iter"] = n_iter
        if steps is not UNSET:
            field_dict["steps"] = steps
        if cfg_scale is not UNSET:
            field_dict["cfg_scale"] = cfg_scale
        if distilled_cfg_scale is not UNSET:
            field_dict["distilled_cfg_scale"] = distilled_cfg_scale
        if width is not UNSET:
            field_dict["width"] = width
        if height is not UNSET:
            field_dict["height"] = height
        if restore_faces is not UNSET:
            field_dict["restore_faces"] = restore_faces
        if tiling is not UNSET:
            field_dict["tiling"] = tiling
        if do_not_save_samples is not UNSET:
            field_dict["do_not_save_samples"] = do_not_save_samples
        if do_not_save_grid is not UNSET:
            field_dict["do_not_save_grid"] = do_not_save_grid
        if eta is not UNSET:
            field_dict["eta"] = eta
        if denoising_strength is not UNSET:
            field_dict["denoising_strength"] = denoising_strength
        if s_min_uncond is not UNSET:
            field_dict["s_min_uncond"] = s_min_uncond
        if s_churn is not UNSET:
            field_dict["s_churn"] = s_churn
        if s_tmax is not UNSET:
            field_dict["s_tmax"] = s_tmax
        if s_tmin is not UNSET:
            field_dict["s_tmin"] = s_tmin
        if s_noise is not UNSET:
            field_dict["s_noise"] = s_noise
        if override_settings is not UNSET:
            field_dict["override_settings"] = override_settings
        if override_settings_restore_afterwards is not UNSET:
            field_dict["override_settings_restore_afterwards"] = override_settings_restore_afterwards
        if refiner_checkpoint is not UNSET:
            field_dict["refiner_checkpoint"] = refiner_checkpoint
        if refiner_switch_at is not UNSET:
            field_dict["refiner_switch_at"] = refiner_switch_at
        if disable_extra_networks is not UNSET:
            field_dict["disable_extra_networks"] = disable_extra_networks
        if firstpass_image is not UNSET:
            field_dict["firstpass_image"] = firstpass_image
        if comments is not UNSET:
            field_dict["comments"] = comments
        if enable_hr is not UNSET:
            field_dict["enable_hr"] = enable_hr
        if firstphase_width is not UNSET:
            field_dict["firstphase_width"] = firstphase_width
        if firstphase_height is not UNSET:
            field_dict["firstphase_height"] = firstphase_height
        if hr_scale is not UNSET:
            field_dict["hr_scale"] = hr_scale
        if hr_upscaler is not UNSET:
            field_dict["hr_upscaler"] = hr_upscaler
        if hr_second_pass_steps is not UNSET:
            field_dict["hr_second_pass_steps"] = hr_second_pass_steps
        if hr_resize_x is not UNSET:
            field_dict["hr_resize_x"] = hr_resize_x
        if hr_resize_y is not UNSET:
            field_dict["hr_resize_y"] = hr_resize_y
        if hr_checkpoint_name is not UNSET:
            field_dict["hr_checkpoint_name"] = hr_checkpoint_name
        if hr_sampler_name is not UNSET:
            field_dict["hr_sampler_name"] = hr_sampler_name
        if hr_scheduler is not UNSET:
            field_dict["hr_scheduler"] = hr_scheduler
        if hr_prompt is not UNSET:
            field_dict["hr_prompt"] = hr_prompt
        if hr_negative_prompt is not UNSET:
            field_dict["hr_negative_prompt"] = hr_negative_prompt
        if force_task_id is not UNSET:
            field_dict["force_task_id"] = force_task_id
        if sampler_index is not UNSET:
            field_dict["sampler_index"] = sampler_index
        if script_name is not UNSET:
            field_dict["script_name"] = script_name
        if script_args is not UNSET:
            field_dict["script_args"] = script_args
        if send_images is not UNSET:
            field_dict["send_images"] = send_images
        if save_images is not UNSET:
            field_dict["save_images"] = save_images
        if alwayson_scripts is not UNSET:
            field_dict["alwayson_scripts"] = alwayson_scripts
        if infotext is not UNSET:
            field_dict["infotext"] = infotext

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.stable_diffusion_processing_txt_2_img_alwayson_scripts import (
            StableDiffusionProcessingTxt2ImgAlwaysonScripts,
        )
        from ..models.stable_diffusion_processing_txt_2_img_comments_type_0 import (
            StableDiffusionProcessingTxt2ImgCommentsType0,
        )
        from ..models.stable_diffusion_processing_txt_2_img_override_settings_type_0 import (
            StableDiffusionProcessingTxt2ImgOverrideSettingsType0,
        )

        d = src_dict.copy()

        def _parse_prompt(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        prompt = _parse_prompt(d.pop("prompt", UNSET))

        def _parse_negative_prompt(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        negative_prompt = _parse_negative_prompt(d.pop("negative_prompt", UNSET))

        def _parse_styles(data: object) -> Union[List[str], None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                styles_type_0 = cast(List[str], data)

                return styles_type_0
            except:  # noqa: E722
                pass
            return cast(Union[List[str], None, Unset], data)

        styles = _parse_styles(d.pop("styles", UNSET))

        def _parse_seed(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        seed = _parse_seed(d.pop("seed", UNSET))

        def _parse_subseed(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        subseed = _parse_subseed(d.pop("subseed", UNSET))

        def _parse_subseed_strength(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        subseed_strength = _parse_subseed_strength(d.pop("subseed_strength", UNSET))

        def _parse_seed_resize_from_h(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        seed_resize_from_h = _parse_seed_resize_from_h(d.pop("seed_resize_from_h", UNSET))

        def _parse_seed_resize_from_w(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        seed_resize_from_w = _parse_seed_resize_from_w(d.pop("seed_resize_from_w", UNSET))

        def _parse_sampler_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sampler_name = _parse_sampler_name(d.pop("sampler_name", UNSET))

        def _parse_scheduler(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        scheduler = _parse_scheduler(d.pop("scheduler", UNSET))

        def _parse_batch_size(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        batch_size = _parse_batch_size(d.pop("batch_size", UNSET))

        def _parse_n_iter(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        n_iter = _parse_n_iter(d.pop("n_iter", UNSET))

        def _parse_steps(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        steps = _parse_steps(d.pop("steps", UNSET))

        def _parse_cfg_scale(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        cfg_scale = _parse_cfg_scale(d.pop("cfg_scale", UNSET))

        def _parse_distilled_cfg_scale(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        distilled_cfg_scale = _parse_distilled_cfg_scale(d.pop("distilled_cfg_scale", UNSET))

        def _parse_width(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        width = _parse_width(d.pop("width", UNSET))

        def _parse_height(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        height = _parse_height(d.pop("height", UNSET))

        def _parse_restore_faces(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        restore_faces = _parse_restore_faces(d.pop("restore_faces", UNSET))

        def _parse_tiling(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        tiling = _parse_tiling(d.pop("tiling", UNSET))

        def _parse_do_not_save_samples(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        do_not_save_samples = _parse_do_not_save_samples(d.pop("do_not_save_samples", UNSET))

        def _parse_do_not_save_grid(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        do_not_save_grid = _parse_do_not_save_grid(d.pop("do_not_save_grid", UNSET))

        def _parse_eta(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        eta = _parse_eta(d.pop("eta", UNSET))

        def _parse_denoising_strength(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        denoising_strength = _parse_denoising_strength(d.pop("denoising_strength", UNSET))

        def _parse_s_min_uncond(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        s_min_uncond = _parse_s_min_uncond(d.pop("s_min_uncond", UNSET))

        def _parse_s_churn(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        s_churn = _parse_s_churn(d.pop("s_churn", UNSET))

        def _parse_s_tmax(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        s_tmax = _parse_s_tmax(d.pop("s_tmax", UNSET))

        def _parse_s_tmin(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        s_tmin = _parse_s_tmin(d.pop("s_tmin", UNSET))

        def _parse_s_noise(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        s_noise = _parse_s_noise(d.pop("s_noise", UNSET))

        def _parse_override_settings(
            data: object,
        ) -> Union["StableDiffusionProcessingTxt2ImgOverrideSettingsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                override_settings_type_0 = StableDiffusionProcessingTxt2ImgOverrideSettingsType0.from_dict(data)

                return override_settings_type_0
            except:  # noqa: E722
                pass
            return cast(Union["StableDiffusionProcessingTxt2ImgOverrideSettingsType0", None, Unset], data)

        override_settings = _parse_override_settings(d.pop("override_settings", UNSET))

        def _parse_override_settings_restore_afterwards(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        override_settings_restore_afterwards = _parse_override_settings_restore_afterwards(
            d.pop("override_settings_restore_afterwards", UNSET)
        )

        def _parse_refiner_checkpoint(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        refiner_checkpoint = _parse_refiner_checkpoint(d.pop("refiner_checkpoint", UNSET))

        def _parse_refiner_switch_at(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        refiner_switch_at = _parse_refiner_switch_at(d.pop("refiner_switch_at", UNSET))

        def _parse_disable_extra_networks(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        disable_extra_networks = _parse_disable_extra_networks(d.pop("disable_extra_networks", UNSET))

        def _parse_firstpass_image(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        firstpass_image = _parse_firstpass_image(d.pop("firstpass_image", UNSET))

        def _parse_comments(data: object) -> Union["StableDiffusionProcessingTxt2ImgCommentsType0", None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, dict):
                    raise TypeError()
                comments_type_0 = StableDiffusionProcessingTxt2ImgCommentsType0.from_dict(data)

                return comments_type_0
            except:  # noqa: E722
                pass
            return cast(Union["StableDiffusionProcessingTxt2ImgCommentsType0", None, Unset], data)

        comments = _parse_comments(d.pop("comments", UNSET))

        def _parse_enable_hr(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        enable_hr = _parse_enable_hr(d.pop("enable_hr", UNSET))

        def _parse_firstphase_width(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        firstphase_width = _parse_firstphase_width(d.pop("firstphase_width", UNSET))

        def _parse_firstphase_height(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        firstphase_height = _parse_firstphase_height(d.pop("firstphase_height", UNSET))

        def _parse_hr_scale(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        hr_scale = _parse_hr_scale(d.pop("hr_scale", UNSET))

        def _parse_hr_upscaler(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        hr_upscaler = _parse_hr_upscaler(d.pop("hr_upscaler", UNSET))

        def _parse_hr_second_pass_steps(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        hr_second_pass_steps = _parse_hr_second_pass_steps(d.pop("hr_second_pass_steps", UNSET))

        def _parse_hr_resize_x(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        hr_resize_x = _parse_hr_resize_x(d.pop("hr_resize_x", UNSET))

        def _parse_hr_resize_y(data: object) -> Union[None, Unset, int]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, int], data)

        hr_resize_y = _parse_hr_resize_y(d.pop("hr_resize_y", UNSET))

        def _parse_hr_checkpoint_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        hr_checkpoint_name = _parse_hr_checkpoint_name(d.pop("hr_checkpoint_name", UNSET))

        def _parse_hr_sampler_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        hr_sampler_name = _parse_hr_sampler_name(d.pop("hr_sampler_name", UNSET))

        def _parse_hr_scheduler(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        hr_scheduler = _parse_hr_scheduler(d.pop("hr_scheduler", UNSET))

        def _parse_hr_prompt(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        hr_prompt = _parse_hr_prompt(d.pop("hr_prompt", UNSET))

        def _parse_hr_negative_prompt(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        hr_negative_prompt = _parse_hr_negative_prompt(d.pop("hr_negative_prompt", UNSET))

        def _parse_force_task_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        force_task_id = _parse_force_task_id(d.pop("force_task_id", UNSET))

        sampler_index = d.pop("sampler_index", UNSET)

        def _parse_script_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        script_name = _parse_script_name(d.pop("script_name", UNSET))

        script_args = cast(List[Any], d.pop("script_args", UNSET))

        send_images = d.pop("send_images", UNSET)

        save_images = d.pop("save_images", UNSET)

        _alwayson_scripts = d.pop("alwayson_scripts", UNSET)
        alwayson_scripts: Union[Unset, StableDiffusionProcessingTxt2ImgAlwaysonScripts]
        if isinstance(_alwayson_scripts, Unset):
            alwayson_scripts = UNSET
        else:
            alwayson_scripts = StableDiffusionProcessingTxt2ImgAlwaysonScripts.from_dict(_alwayson_scripts)

        def _parse_infotext(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        infotext = _parse_infotext(d.pop("infotext", UNSET))

        stable_diffusion_processing_txt_2_img = cls(
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=styles,
            seed=seed,
            subseed=subseed,
            subseed_strength=subseed_strength,
            seed_resize_from_h=seed_resize_from_h,
            seed_resize_from_w=seed_resize_from_w,
            sampler_name=sampler_name,
            scheduler=scheduler,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
            distilled_cfg_scale=distilled_cfg_scale,
            width=width,
            height=height,
            restore_faces=restore_faces,
            tiling=tiling,
            do_not_save_samples=do_not_save_samples,
            do_not_save_grid=do_not_save_grid,
            eta=eta,
            denoising_strength=denoising_strength,
            s_min_uncond=s_min_uncond,
            s_churn=s_churn,
            s_tmax=s_tmax,
            s_tmin=s_tmin,
            s_noise=s_noise,
            override_settings=override_settings,
            override_settings_restore_afterwards=override_settings_restore_afterwards,
            refiner_checkpoint=refiner_checkpoint,
            refiner_switch_at=refiner_switch_at,
            disable_extra_networks=disable_extra_networks,
            firstpass_image=firstpass_image,
            comments=comments,
            enable_hr=enable_hr,
            firstphase_width=firstphase_width,
            firstphase_height=firstphase_height,
            hr_scale=hr_scale,
            hr_upscaler=hr_upscaler,
            hr_second_pass_steps=hr_second_pass_steps,
            hr_resize_x=hr_resize_x,
            hr_resize_y=hr_resize_y,
            hr_checkpoint_name=hr_checkpoint_name,
            hr_sampler_name=hr_sampler_name,
            hr_scheduler=hr_scheduler,
            hr_prompt=hr_prompt,
            hr_negative_prompt=hr_negative_prompt,
            force_task_id=force_task_id,
            sampler_index=sampler_index,
            script_name=script_name,
            script_args=script_args,
            send_images=send_images,
            save_images=save_images,
            alwayson_scripts=alwayson_scripts,
            infotext=infotext,
        )

        stable_diffusion_processing_txt_2_img.additional_properties = d
        return stable_diffusion_processing_txt_2_img

    @property
    def additional_keys(self) -> List[str]:
        return list(self.additional_properties.keys())

    def __getitem__(self, key: str) -> Any:
        return self.additional_properties[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.additional_properties[key] = value

    def __delitem__(self, key: str) -> None:
        del self.additional_properties[key]

    def __contains__(self, key: str) -> bool:
        return key in self.additional_properties
