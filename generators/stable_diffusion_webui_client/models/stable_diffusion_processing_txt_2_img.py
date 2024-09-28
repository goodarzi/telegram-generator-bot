from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.stable_diffusion_processing_txt_2_img_alwayson_scripts import (
        StableDiffusionProcessingTxt2ImgAlwaysonScripts,
    )
    from ..models.stable_diffusion_processing_txt_2_img_comments import StableDiffusionProcessingTxt2ImgComments
    from ..models.stable_diffusion_processing_txt_2_img_override_settings import (
        StableDiffusionProcessingTxt2ImgOverrideSettings,
    )


T = TypeVar("T", bound="StableDiffusionProcessingTxt2Img")


@_attrs_define
class StableDiffusionProcessingTxt2Img:
    """
    Attributes:
        prompt (Union[Unset, str]):  Default: ''.
        negative_prompt (Union[Unset, str]):  Default: ''.
        styles (Union[Unset, List[str]]):
        seed (Union[Unset, int]):  Default: -1.
        subseed (Union[Unset, int]):  Default: -1.
        subseed_strength (Union[Unset, float]):  Default: 0.0.
        seed_resize_from_h (Union[Unset, int]):  Default: -1.
        seed_resize_from_w (Union[Unset, int]):  Default: -1.
        sampler_name (Union[Unset, str]):
        batch_size (Union[Unset, int]):  Default: 1.
        n_iter (Union[Unset, int]):  Default: 1.
        steps (Union[Unset, int]):  Default: 50.
        cfg_scale (Union[Unset, float]):  Default: 7.0.
        width (Union[Unset, int]):  Default: 512.
        height (Union[Unset, int]):  Default: 512.
        restore_faces (Union[Unset, bool]):
        tiling (Union[Unset, bool]):
        do_not_save_samples (Union[Unset, bool]):  Default: False.
        do_not_save_grid (Union[Unset, bool]):  Default: False.
        eta (Union[Unset, float]):
        denoising_strength (Union[Unset, float]):
        s_min_uncond (Union[Unset, float]):
        s_churn (Union[Unset, float]):
        s_tmax (Union[Unset, float]):
        s_tmin (Union[Unset, float]):
        s_noise (Union[Unset, float]):
        override_settings (Union[Unset, StableDiffusionProcessingTxt2ImgOverrideSettings]):
        override_settings_restore_afterwards (Union[Unset, bool]):  Default: True.
        refiner_checkpoint (Union[Unset, str]):
        refiner_switch_at (Union[Unset, float]):
        disable_extra_networks (Union[Unset, bool]):  Default: False.
        firstpass_image (Union[Unset, str]):
        comments (Union[Unset, StableDiffusionProcessingTxt2ImgComments]):
        enable_hr (Union[Unset, bool]):  Default: False.
        firstphase_width (Union[Unset, int]):  Default: 0.
        firstphase_height (Union[Unset, int]):  Default: 0.
        hr_scale (Union[Unset, float]):  Default: 2.0.
        hr_upscaler (Union[Unset, str]):
        hr_second_pass_steps (Union[Unset, int]):  Default: 0.
        hr_resize_x (Union[Unset, int]):  Default: 0.
        hr_resize_y (Union[Unset, int]):  Default: 0.
        hr_checkpoint_name (Union[Unset, str]):
        hr_sampler_name (Union[Unset, str]):
        hr_prompt (Union[Unset, str]):  Default: ''.
        hr_negative_prompt (Union[Unset, str]):  Default: ''.
        force_task_id (Union[Unset, str]):
        sampler_index (Union[Unset, str]):  Default: 'Euler'.
        script_name (Union[Unset, str]):
        script_args (Union[Unset, List[Any]]):
        send_images (Union[Unset, bool]):  Default: True.
        save_images (Union[Unset, bool]):  Default: False.
        alwayson_scripts (Union[Unset, StableDiffusionProcessingTxt2ImgAlwaysonScripts]):
        infotext (Union[Unset, str]):
    """

    prompt: Union[Unset, str] = ""
    negative_prompt: Union[Unset, str] = ""
    styles: Union[Unset, List[str]] = UNSET
    seed: Union[Unset, int] = -1
    subseed: Union[Unset, int] = -1
    subseed_strength: Union[Unset, float] = 0.0
    seed_resize_from_h: Union[Unset, int] = -1
    seed_resize_from_w: Union[Unset, int] = -1
    sampler_name: Union[Unset, str] = UNSET
    batch_size: Union[Unset, int] = 1
    n_iter: Union[Unset, int] = 1
    steps: Union[Unset, int] = 50
    cfg_scale: Union[Unset, float] = 7.0
    width: Union[Unset, int] = 512
    height: Union[Unset, int] = 512
    restore_faces: Union[Unset, bool] = UNSET
    tiling: Union[Unset, bool] = UNSET
    do_not_save_samples: Union[Unset, bool] = False
    do_not_save_grid: Union[Unset, bool] = False
    eta: Union[Unset, float] = UNSET
    denoising_strength: Union[Unset, float] = UNSET
    s_min_uncond: Union[Unset, float] = UNSET
    s_churn: Union[Unset, float] = UNSET
    s_tmax: Union[Unset, float] = UNSET
    s_tmin: Union[Unset, float] = UNSET
    s_noise: Union[Unset, float] = UNSET
    override_settings: Union[Unset, "StableDiffusionProcessingTxt2ImgOverrideSettings"] = UNSET
    override_settings_restore_afterwards: Union[Unset, bool] = True
    refiner_checkpoint: Union[Unset, str] = UNSET
    refiner_switch_at: Union[Unset, float] = UNSET
    disable_extra_networks: Union[Unset, bool] = False
    firstpass_image: Union[Unset, str] = UNSET
    comments: Union[Unset, "StableDiffusionProcessingTxt2ImgComments"] = UNSET
    enable_hr: Union[Unset, bool] = False
    firstphase_width: Union[Unset, int] = 0
    firstphase_height: Union[Unset, int] = 0
    hr_scale: Union[Unset, float] = 2.0
    hr_upscaler: Union[Unset, str] = UNSET
    hr_second_pass_steps: Union[Unset, int] = 0
    hr_resize_x: Union[Unset, int] = 0
    hr_resize_y: Union[Unset, int] = 0
    hr_checkpoint_name: Union[Unset, str] = UNSET
    hr_sampler_name: Union[Unset, str] = UNSET
    hr_prompt: Union[Unset, str] = ""
    hr_negative_prompt: Union[Unset, str] = ""
    force_task_id: Union[Unset, str] = UNSET
    sampler_index: Union[Unset, str] = "Euler"
    script_name: Union[Unset, str] = UNSET
    script_args: Union[Unset, List[Any]] = UNSET
    send_images: Union[Unset, bool] = True
    save_images: Union[Unset, bool] = False
    alwayson_scripts: Union[Unset, "StableDiffusionProcessingTxt2ImgAlwaysonScripts"] = UNSET
    infotext: Union[Unset, str] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        prompt = self.prompt

        negative_prompt = self.negative_prompt

        styles: Union[Unset, List[str]] = UNSET
        if not isinstance(self.styles, Unset):
            styles = self.styles

        seed = self.seed

        subseed = self.subseed

        subseed_strength = self.subseed_strength

        seed_resize_from_h = self.seed_resize_from_h

        seed_resize_from_w = self.seed_resize_from_w

        sampler_name = self.sampler_name

        batch_size = self.batch_size

        n_iter = self.n_iter

        steps = self.steps

        cfg_scale = self.cfg_scale

        width = self.width

        height = self.height

        restore_faces = self.restore_faces

        tiling = self.tiling

        do_not_save_samples = self.do_not_save_samples

        do_not_save_grid = self.do_not_save_grid

        eta = self.eta

        denoising_strength = self.denoising_strength

        s_min_uncond = self.s_min_uncond

        s_churn = self.s_churn

        s_tmax = self.s_tmax

        s_tmin = self.s_tmin

        s_noise = self.s_noise

        override_settings: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.override_settings, Unset):
            override_settings = self.override_settings.to_dict()

        override_settings_restore_afterwards = self.override_settings_restore_afterwards

        refiner_checkpoint = self.refiner_checkpoint

        refiner_switch_at = self.refiner_switch_at

        disable_extra_networks = self.disable_extra_networks

        firstpass_image = self.firstpass_image

        comments: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.comments, Unset):
            comments = self.comments.to_dict()

        enable_hr = self.enable_hr

        firstphase_width = self.firstphase_width

        firstphase_height = self.firstphase_height

        hr_scale = self.hr_scale

        hr_upscaler = self.hr_upscaler

        hr_second_pass_steps = self.hr_second_pass_steps

        hr_resize_x = self.hr_resize_x

        hr_resize_y = self.hr_resize_y

        hr_checkpoint_name = self.hr_checkpoint_name

        hr_sampler_name = self.hr_sampler_name

        hr_prompt = self.hr_prompt

        hr_negative_prompt = self.hr_negative_prompt

        force_task_id = self.force_task_id

        sampler_index = self.sampler_index

        script_name = self.script_name

        script_args: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.script_args, Unset):
            script_args = self.script_args

        send_images = self.send_images

        save_images = self.save_images

        alwayson_scripts: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.alwayson_scripts, Unset):
            alwayson_scripts = self.alwayson_scripts.to_dict()

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
        if batch_size is not UNSET:
            field_dict["batch_size"] = batch_size
        if n_iter is not UNSET:
            field_dict["n_iter"] = n_iter
        if steps is not UNSET:
            field_dict["steps"] = steps
        if cfg_scale is not UNSET:
            field_dict["cfg_scale"] = cfg_scale
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
        from ..models.stable_diffusion_processing_txt_2_img_comments import StableDiffusionProcessingTxt2ImgComments
        from ..models.stable_diffusion_processing_txt_2_img_override_settings import (
            StableDiffusionProcessingTxt2ImgOverrideSettings,
        )

        d = src_dict.copy()
        prompt = d.pop("prompt", UNSET)

        negative_prompt = d.pop("negative_prompt", UNSET)

        styles = cast(List[str], d.pop("styles", UNSET))

        seed = d.pop("seed", UNSET)

        subseed = d.pop("subseed", UNSET)

        subseed_strength = d.pop("subseed_strength", UNSET)

        seed_resize_from_h = d.pop("seed_resize_from_h", UNSET)

        seed_resize_from_w = d.pop("seed_resize_from_w", UNSET)

        sampler_name = d.pop("sampler_name", UNSET)

        batch_size = d.pop("batch_size", UNSET)

        n_iter = d.pop("n_iter", UNSET)

        steps = d.pop("steps", UNSET)

        cfg_scale = d.pop("cfg_scale", UNSET)

        width = d.pop("width", UNSET)

        height = d.pop("height", UNSET)

        restore_faces = d.pop("restore_faces", UNSET)

        tiling = d.pop("tiling", UNSET)

        do_not_save_samples = d.pop("do_not_save_samples", UNSET)

        do_not_save_grid = d.pop("do_not_save_grid", UNSET)

        eta = d.pop("eta", UNSET)

        denoising_strength = d.pop("denoising_strength", UNSET)

        s_min_uncond = d.pop("s_min_uncond", UNSET)

        s_churn = d.pop("s_churn", UNSET)

        s_tmax = d.pop("s_tmax", UNSET)

        s_tmin = d.pop("s_tmin", UNSET)

        s_noise = d.pop("s_noise", UNSET)

        _override_settings = d.pop("override_settings", UNSET)
        override_settings: Union[Unset, StableDiffusionProcessingTxt2ImgOverrideSettings]
        if isinstance(_override_settings, Unset):
            override_settings = UNSET
        else:
            override_settings = StableDiffusionProcessingTxt2ImgOverrideSettings.from_dict(_override_settings)

        override_settings_restore_afterwards = d.pop("override_settings_restore_afterwards", UNSET)

        refiner_checkpoint = d.pop("refiner_checkpoint", UNSET)

        refiner_switch_at = d.pop("refiner_switch_at", UNSET)

        disable_extra_networks = d.pop("disable_extra_networks", UNSET)

        firstpass_image = d.pop("firstpass_image", UNSET)

        _comments = d.pop("comments", UNSET)
        comments: Union[Unset, StableDiffusionProcessingTxt2ImgComments]
        if isinstance(_comments, Unset):
            comments = UNSET
        else:
            comments = StableDiffusionProcessingTxt2ImgComments.from_dict(_comments)

        enable_hr = d.pop("enable_hr", UNSET)

        firstphase_width = d.pop("firstphase_width", UNSET)

        firstphase_height = d.pop("firstphase_height", UNSET)

        hr_scale = d.pop("hr_scale", UNSET)

        hr_upscaler = d.pop("hr_upscaler", UNSET)

        hr_second_pass_steps = d.pop("hr_second_pass_steps", UNSET)

        hr_resize_x = d.pop("hr_resize_x", UNSET)

        hr_resize_y = d.pop("hr_resize_y", UNSET)

        hr_checkpoint_name = d.pop("hr_checkpoint_name", UNSET)

        hr_sampler_name = d.pop("hr_sampler_name", UNSET)

        hr_prompt = d.pop("hr_prompt", UNSET)

        hr_negative_prompt = d.pop("hr_negative_prompt", UNSET)

        force_task_id = d.pop("force_task_id", UNSET)

        sampler_index = d.pop("sampler_index", UNSET)

        script_name = d.pop("script_name", UNSET)

        script_args = cast(List[Any], d.pop("script_args", UNSET))

        send_images = d.pop("send_images", UNSET)

        save_images = d.pop("save_images", UNSET)

        _alwayson_scripts = d.pop("alwayson_scripts", UNSET)
        alwayson_scripts: Union[Unset, StableDiffusionProcessingTxt2ImgAlwaysonScripts]
        if isinstance(_alwayson_scripts, Unset):
            alwayson_scripts = UNSET
        else:
            alwayson_scripts = StableDiffusionProcessingTxt2ImgAlwaysonScripts.from_dict(_alwayson_scripts)

        infotext = d.pop("infotext", UNSET)

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
            batch_size=batch_size,
            n_iter=n_iter,
            steps=steps,
            cfg_scale=cfg_scale,
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
