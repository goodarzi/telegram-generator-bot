from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.stable_diffusion_processing_img_2_img_alwayson_scripts import (
        StableDiffusionProcessingImg2ImgAlwaysonScripts,
    )
    from ..models.stable_diffusion_processing_img_2_img_comments import StableDiffusionProcessingImg2ImgComments
    from ..models.stable_diffusion_processing_img_2_img_override_settings import (
        StableDiffusionProcessingImg2ImgOverrideSettings,
    )


T = TypeVar("T", bound="StableDiffusionProcessingImg2Img")


@_attrs_define
class StableDiffusionProcessingImg2Img:
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
        denoising_strength (Union[Unset, float]):  Default: 0.75.
        s_min_uncond (Union[Unset, float]):
        s_churn (Union[Unset, float]):
        s_tmax (Union[Unset, float]):
        s_tmin (Union[Unset, float]):
        s_noise (Union[Unset, float]):
        override_settings (Union[Unset, StableDiffusionProcessingImg2ImgOverrideSettings]):
        override_settings_restore_afterwards (Union[Unset, bool]):  Default: True.
        refiner_checkpoint (Union[Unset, str]):
        refiner_switch_at (Union[Unset, float]):
        disable_extra_networks (Union[Unset, bool]):  Default: False.
        firstpass_image (Union[Unset, str]):
        comments (Union[Unset, StableDiffusionProcessingImg2ImgComments]):
        init_images (Union[Unset, List[Any]]):
        resize_mode (Union[Unset, int]):  Default: 0.
        image_cfg_scale (Union[Unset, float]):
        mask (Union[Unset, str]):
        mask_blur_x (Union[Unset, int]):  Default: 4.
        mask_blur_y (Union[Unset, int]):  Default: 4.
        mask_blur (Union[Unset, int]):
        mask_round (Union[Unset, bool]):  Default: True.
        inpainting_fill (Union[Unset, int]):  Default: 0.
        inpaint_full_res (Union[Unset, bool]):  Default: True.
        inpaint_full_res_padding (Union[Unset, int]):  Default: 0.
        inpainting_mask_invert (Union[Unset, int]):  Default: 0.
        initial_noise_multiplier (Union[Unset, float]):
        latent_mask (Union[Unset, str]):
        force_task_id (Union[Unset, str]):
        sampler_index (Union[Unset, str]):  Default: 'Euler'.
        include_init_images (Union[Unset, bool]):  Default: False.
        script_name (Union[Unset, str]):
        script_args (Union[Unset, List[Any]]):
        send_images (Union[Unset, bool]):  Default: True.
        save_images (Union[Unset, bool]):  Default: False.
        alwayson_scripts (Union[Unset, StableDiffusionProcessingImg2ImgAlwaysonScripts]):
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
    denoising_strength: Union[Unset, float] = 0.75
    s_min_uncond: Union[Unset, float] = UNSET
    s_churn: Union[Unset, float] = UNSET
    s_tmax: Union[Unset, float] = UNSET
    s_tmin: Union[Unset, float] = UNSET
    s_noise: Union[Unset, float] = UNSET
    override_settings: Union[Unset, "StableDiffusionProcessingImg2ImgOverrideSettings"] = UNSET
    override_settings_restore_afterwards: Union[Unset, bool] = True
    refiner_checkpoint: Union[Unset, str] = UNSET
    refiner_switch_at: Union[Unset, float] = UNSET
    disable_extra_networks: Union[Unset, bool] = False
    firstpass_image: Union[Unset, str] = UNSET
    comments: Union[Unset, "StableDiffusionProcessingImg2ImgComments"] = UNSET
    init_images: Union[Unset, List[Any]] = UNSET
    resize_mode: Union[Unset, int] = 0
    image_cfg_scale: Union[Unset, float] = UNSET
    mask: Union[Unset, str] = UNSET
    mask_blur_x: Union[Unset, int] = 4
    mask_blur_y: Union[Unset, int] = 4
    mask_blur: Union[Unset, int] = UNSET
    mask_round: Union[Unset, bool] = True
    inpainting_fill: Union[Unset, int] = 0
    inpaint_full_res: Union[Unset, bool] = True
    inpaint_full_res_padding: Union[Unset, int] = 0
    inpainting_mask_invert: Union[Unset, int] = 0
    initial_noise_multiplier: Union[Unset, float] = UNSET
    latent_mask: Union[Unset, str] = UNSET
    force_task_id: Union[Unset, str] = UNSET
    sampler_index: Union[Unset, str] = "Euler"
    include_init_images: Union[Unset, bool] = False
    script_name: Union[Unset, str] = UNSET
    script_args: Union[Unset, List[Any]] = UNSET
    send_images: Union[Unset, bool] = True
    save_images: Union[Unset, bool] = False
    alwayson_scripts: Union[Unset, "StableDiffusionProcessingImg2ImgAlwaysonScripts"] = UNSET
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

        init_images: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.init_images, Unset):
            init_images = self.init_images

        resize_mode = self.resize_mode

        image_cfg_scale = self.image_cfg_scale

        mask = self.mask

        mask_blur_x = self.mask_blur_x

        mask_blur_y = self.mask_blur_y

        mask_blur = self.mask_blur

        mask_round = self.mask_round

        inpainting_fill = self.inpainting_fill

        inpaint_full_res = self.inpaint_full_res

        inpaint_full_res_padding = self.inpaint_full_res_padding

        inpainting_mask_invert = self.inpainting_mask_invert

        initial_noise_multiplier = self.initial_noise_multiplier

        latent_mask = self.latent_mask

        force_task_id = self.force_task_id

        sampler_index = self.sampler_index

        include_init_images = self.include_init_images

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
        if init_images is not UNSET:
            field_dict["init_images"] = init_images
        if resize_mode is not UNSET:
            field_dict["resize_mode"] = resize_mode
        if image_cfg_scale is not UNSET:
            field_dict["image_cfg_scale"] = image_cfg_scale
        if mask is not UNSET:
            field_dict["mask"] = mask
        if mask_blur_x is not UNSET:
            field_dict["mask_blur_x"] = mask_blur_x
        if mask_blur_y is not UNSET:
            field_dict["mask_blur_y"] = mask_blur_y
        if mask_blur is not UNSET:
            field_dict["mask_blur"] = mask_blur
        if mask_round is not UNSET:
            field_dict["mask_round"] = mask_round
        if inpainting_fill is not UNSET:
            field_dict["inpainting_fill"] = inpainting_fill
        if inpaint_full_res is not UNSET:
            field_dict["inpaint_full_res"] = inpaint_full_res
        if inpaint_full_res_padding is not UNSET:
            field_dict["inpaint_full_res_padding"] = inpaint_full_res_padding
        if inpainting_mask_invert is not UNSET:
            field_dict["inpainting_mask_invert"] = inpainting_mask_invert
        if initial_noise_multiplier is not UNSET:
            field_dict["initial_noise_multiplier"] = initial_noise_multiplier
        if latent_mask is not UNSET:
            field_dict["latent_mask"] = latent_mask
        if force_task_id is not UNSET:
            field_dict["force_task_id"] = force_task_id
        if sampler_index is not UNSET:
            field_dict["sampler_index"] = sampler_index
        if include_init_images is not UNSET:
            field_dict["include_init_images"] = include_init_images
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
        from ..models.stable_diffusion_processing_img_2_img_alwayson_scripts import (
            StableDiffusionProcessingImg2ImgAlwaysonScripts,
        )
        from ..models.stable_diffusion_processing_img_2_img_comments import StableDiffusionProcessingImg2ImgComments
        from ..models.stable_diffusion_processing_img_2_img_override_settings import (
            StableDiffusionProcessingImg2ImgOverrideSettings,
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
        override_settings: Union[Unset, StableDiffusionProcessingImg2ImgOverrideSettings]
        if isinstance(_override_settings, Unset):
            override_settings = UNSET
        else:
            override_settings = StableDiffusionProcessingImg2ImgOverrideSettings.from_dict(_override_settings)

        override_settings_restore_afterwards = d.pop("override_settings_restore_afterwards", UNSET)

        refiner_checkpoint = d.pop("refiner_checkpoint", UNSET)

        refiner_switch_at = d.pop("refiner_switch_at", UNSET)

        disable_extra_networks = d.pop("disable_extra_networks", UNSET)

        firstpass_image = d.pop("firstpass_image", UNSET)

        _comments = d.pop("comments", UNSET)
        comments: Union[Unset, StableDiffusionProcessingImg2ImgComments]
        if isinstance(_comments, Unset):
            comments = UNSET
        else:
            comments = StableDiffusionProcessingImg2ImgComments.from_dict(_comments)

        init_images = cast(List[Any], d.pop("init_images", UNSET))

        resize_mode = d.pop("resize_mode", UNSET)

        image_cfg_scale = d.pop("image_cfg_scale", UNSET)

        mask = d.pop("mask", UNSET)

        mask_blur_x = d.pop("mask_blur_x", UNSET)

        mask_blur_y = d.pop("mask_blur_y", UNSET)

        mask_blur = d.pop("mask_blur", UNSET)

        mask_round = d.pop("mask_round", UNSET)

        inpainting_fill = d.pop("inpainting_fill", UNSET)

        inpaint_full_res = d.pop("inpaint_full_res", UNSET)

        inpaint_full_res_padding = d.pop("inpaint_full_res_padding", UNSET)

        inpainting_mask_invert = d.pop("inpainting_mask_invert", UNSET)

        initial_noise_multiplier = d.pop("initial_noise_multiplier", UNSET)

        latent_mask = d.pop("latent_mask", UNSET)

        force_task_id = d.pop("force_task_id", UNSET)

        sampler_index = d.pop("sampler_index", UNSET)

        include_init_images = d.pop("include_init_images", UNSET)

        script_name = d.pop("script_name", UNSET)

        script_args = cast(List[Any], d.pop("script_args", UNSET))

        send_images = d.pop("send_images", UNSET)

        save_images = d.pop("save_images", UNSET)

        _alwayson_scripts = d.pop("alwayson_scripts", UNSET)
        alwayson_scripts: Union[Unset, StableDiffusionProcessingImg2ImgAlwaysonScripts]
        if isinstance(_alwayson_scripts, Unset):
            alwayson_scripts = UNSET
        else:
            alwayson_scripts = StableDiffusionProcessingImg2ImgAlwaysonScripts.from_dict(_alwayson_scripts)

        infotext = d.pop("infotext", UNSET)

        stable_diffusion_processing_img_2_img = cls(
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
            init_images=init_images,
            resize_mode=resize_mode,
            image_cfg_scale=image_cfg_scale,
            mask=mask,
            mask_blur_x=mask_blur_x,
            mask_blur_y=mask_blur_y,
            mask_blur=mask_blur,
            mask_round=mask_round,
            inpainting_fill=inpainting_fill,
            inpaint_full_res=inpaint_full_res,
            inpaint_full_res_padding=inpaint_full_res_padding,
            inpainting_mask_invert=inpainting_mask_invert,
            initial_noise_multiplier=initial_noise_multiplier,
            latent_mask=latent_mask,
            force_task_id=force_task_id,
            sampler_index=sampler_index,
            include_init_images=include_init_images,
            script_name=script_name,
            script_args=script_args,
            send_images=send_images,
            save_images=save_images,
            alwayson_scripts=alwayson_scripts,
            infotext=infotext,
        )

        stable_diffusion_processing_img_2_img.additional_properties = d
        return stable_diffusion_processing_img_2_img

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
