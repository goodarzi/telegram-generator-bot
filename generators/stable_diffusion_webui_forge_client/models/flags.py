from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

if TYPE_CHECKING:
    from ..models.flags_ngrok_options import FlagsNgrokOptions


T = TypeVar("T", bound="Flags")


@_attrs_define
class Flags:
    """
    Attributes:
        gpu_device_id (Union[None, Unset, str]):
        all_in_fp32 (Union[Unset, bool]):  Default: False.
        all_in_fp16 (Union[Unset, bool]):  Default: False.
        unet_in_bf16 (Union[Unset, bool]):  Default: False.
        unet_in_fp16 (Union[Unset, bool]):  Default: False.
        unet_in_fp8_e4m3fn (Union[Unset, bool]):  Default: False.
        unet_in_fp8_e5m2 (Union[Unset, bool]):  Default: False.
        vae_in_fp16 (Union[Unset, bool]):  Default: False.
        vae_in_fp32 (Union[Unset, bool]):  Default: False.
        vae_in_bf16 (Union[Unset, bool]):  Default: False.
        vae_in_cpu (Union[Unset, bool]):  Default: False.
        clip_in_fp8_e4m3fn (Union[Unset, bool]):  Default: False.
        clip_in_fp8_e5m2 (Union[Unset, bool]):  Default: False.
        clip_in_fp16 (Union[Unset, bool]):  Default: False.
        clip_in_fp32 (Union[Unset, bool]):  Default: False.
        attention_split (Union[Unset, bool]):  Default: False.
        attention_quad (Union[Unset, bool]):  Default: False.
        attention_pytorch (Union[Unset, bool]):  Default: False.
        force_upcast_attention (Union[Unset, bool]):  Default: False.
        disable_attention_upcast (Union[Unset, bool]):  Default: False.
        disable_xformers (Union[Unset, bool]):  Default: False.
        directml (Union[None, Unset, str]):
        disable_ipex_hijack (Union[Unset, bool]):  Default: False.
        always_gpu (Union[Unset, bool]):  Default: False.
        always_high_vram (Union[Unset, bool]):  Default: False.
        always_normal_vram (Union[Unset, bool]):  Default: False.
        always_low_vram (Union[Unset, bool]):  Default: False.
        always_no_vram (Union[Unset, bool]):  Default: False.
        always_cpu (Union[Unset, bool]):  Default: False.
        always_offload_from_vram (Union[Unset, bool]):  Default: False.
        pytorch_deterministic (Union[Unset, bool]):  Default: False.
        cuda_malloc (Union[Unset, bool]):  Default: False.
        cuda_stream (Union[Unset, bool]):  Default: False.
        pin_shared_memory (Union[Unset, bool]):  Default: False.
        disable_gpu_warning (Union[Unset, bool]):  Default: False.
        f (Union[Unset, bool]): ==SUPPRESS== Default: False.
        update_all_extensions (Union[Unset, bool]): launch.py argument: download updates for all extensions when
            starting the program Default: False.
        skip_python_version_check (Union[Unset, bool]): launch.py argument: do not check python version Default: False.
        skip_torch_cuda_test (Union[Unset, bool]): launch.py argument: do not check if CUDA is able to work properly
            Default: False.
        reinstall_xformers (Union[Unset, bool]): launch.py argument: install the appropriate version of xformers even if
            you have some version already installed Default: False.
        reinstall_torch (Union[Unset, bool]): launch.py argument: install the appropriate version of torch even if you
            have some version already installed Default: False.
        update_check (Union[Unset, bool]): launch.py argument: check for updates at startup Default: False.
        test_server (Union[Unset, bool]): launch.py argument: configure server for testing Default: False.
        log_startup (Union[Unset, bool]): launch.py argument: print a detailed log of what's happening at startup
            Default: False.
        skip_prepare_environment (Union[Unset, bool]): launch.py argument: skip all environment preparation Default:
            False.
        skip_install (Union[Unset, bool]): launch.py argument: skip installation of packages Default: False.
        dump_sysinfo (Union[Unset, bool]): launch.py argument: dump limited sysinfo file (without information about
            extensions, options) to disk and quit Default: False.
        loglevel (Union[None, Unset, str]): log level; one of: CRITICAL, ERROR, WARNING, INFO, DEBUG
        do_not_download_clip (Union[Unset, bool]): do not download CLIP model even if it's not included in the
            checkpoint Default: False.
        data_dir (Union[Unset, str]): base path where all user data is stored Default: '/mnt/sam860evo/forge/stable-
            diffusion-webui-forge'.
        models_dir (Union[None, Unset, str]): base path where models are stored; overrides --data-dir
        config (Union[Unset, str]): path to config which constructs model Default: '/mnt/sam860evo/forge/stable-
            diffusion-webui-forge/configs/v1-inference.yaml'.
        ckpt (Union[Unset, str]): path to checkpoint of stable diffusion model; if specified, this checkpoint will be
            added to the list of checkpoints and loaded Default: '/mnt/sam860evo/forge/stable-diffusion-webui-
            forge/model.ckpt'.
        ckpt_dir (Union[None, Unset, str]): Path to directory with stable diffusion checkpoints
        vae_dir (Union[None, Unset, str]): Path to directory with VAE files
        text_encoder_dir (Union[None, Unset, str]): Path to directory with text encoder models
        gfpgan_dir (Union[Unset, str]): GFPGAN directory Default: './GFPGAN'.
        gfpgan_model (Union[None, Unset, str]): GFPGAN model file name
        no_half (Union[Unset, bool]): do not switch the model to 16-bit floats Default: False.
        no_half_vae (Union[Unset, bool]): do not switch the VAE model to 16-bit floats Default: False.
        no_progressbar_hiding (Union[Unset, bool]): do not hide progressbar in gradio UI (we hide it because it slows
            down ML if you have hardware acceleration in browser) Default: False.
        max_batch_count (Union[Unset, int]): does not do anything Default: 16.
        embeddings_dir (Union[Unset, str]): embeddings directory for textual inversion (default: embeddings) Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/embeddings'.
        textual_inversion_templates_dir (Union[Unset, str]): directory with textual inversion templates Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/textual_inversion_templates'.
        hypernetwork_dir (Union[Unset, str]): hypernetwork directory Default: '/mnt/sam860evo/forge/stable-diffusion-
            webui-forge/models/hypernetworks'.
        localizations_dir (Union[Unset, str]): localizations directory Default: '/mnt/sam860evo/forge/stable-diffusion-
            webui-forge/localizations'.
        allow_code (Union[Unset, bool]): allow custom script execution from webui Default: False.
        medvram (Union[Unset, bool]): enable stable diffusion model optimizations for sacrificing a little speed for low
            VRM usage Default: False.
        medvram_sdxl (Union[Unset, bool]): enable --medvram optimization just for SDXL models Default: False.
        lowvram (Union[Unset, bool]): enable stable diffusion model optimizations for sacrificing a lot of speed for
            very low VRM usage Default: False.
        lowram (Union[Unset, bool]): load stable diffusion checkpoint weights to VRAM instead of RAM Default: False.
        always_batch_cond_uncond (Union[Unset, bool]): does not do anything Default: False.
        unload_gfpgan (Union[Unset, bool]): does not do anything. Default: False.
        precision (Union[Unset, str]): evaluate at this precision Default: 'autocast'.
        upcast_sampling (Union[Unset, bool]): upcast sampling. No effect with --no-half. Usually produces similar
            results to --no-half with better performance while using less memory. Default: False.
        share (Union[Unset, bool]): use share=True for gradio and make the UI accessible through their site Default:
            False.
        ngrok (Union[None, Unset, str]): ngrok authtoken, alternative to gradio --share
        ngrok_region (Union[Unset, str]): does not do anything. Default: ''.
        ngrok_options (Union[Unset, FlagsNgrokOptions]): The options to pass to ngrok in JSON format, e.g.:
            '{"authtoken_from_env":true, "basic_auth":"user:password", "oauth_provider":"google",
            "oauth_allow_emails":"user@asdf.com"}'
        enable_insecure_extension_access (Union[Unset, bool]): enable extensions tab regardless of other options
            Default: False.
        codeformer_models_path (Union[Unset, str]): Path to directory with codeformer model file(s). Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/Codeformer'.
        gfpgan_models_path (Union[Unset, str]): Path to directory with GFPGAN model file(s). Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/GFPGAN'.
        esrgan_models_path (Union[Unset, str]): Path to directory with ESRGAN model file(s). Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/ESRGAN'.
        bsrgan_models_path (Union[Unset, str]): Path to directory with BSRGAN model file(s). Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/BSRGAN'.
        realesrgan_models_path (Union[Unset, str]): Path to directory with RealESRGAN model file(s). Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/RealESRGAN'.
        dat_models_path (Union[Unset, str]): Path to directory with DAT model file(s). Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/DAT'.
        clip_models_path (Union[None, Unset, str]): Path to directory with CLIP model file(s), for Interrogate options.
        xformers (Union[Unset, bool]): enable xformers for cross attention layers Default: False.
        force_enable_xformers (Union[Unset, bool]): enable xformers for cross attention layers regardless of whether the
            checking code thinks you can run it; do not make bug reports if this fails to work Default: False.
        xformers_flash_attention (Union[Unset, bool]): enable xformers with Flash Attention to improve reproducibility
            (supported for SD2.x or variant only) Default: False.
        deepdanbooru (Union[Unset, bool]): does not do anything Default: False.
        opt_split_attention (Union[Unset, bool]): prefer Doggettx's cross-attention layer optimization for automatic
            choice of optimization Default: False.
        opt_sub_quad_attention (Union[Unset, bool]): prefer memory efficient sub-quadratic cross-attention layer
            optimization for automatic choice of optimization Default: False.
        sub_quad_q_chunk_size (Union[Unset, int]): query chunk size for the sub-quadratic cross-attention layer
            optimization to use Default: 1024.
        sub_quad_kv_chunk_size (Union[None, Unset, str]): kv chunk size for the sub-quadratic cross-attention layer
            optimization to use
        sub_quad_chunk_threshold (Union[None, Unset, str]): the percentage of VRAM threshold for the sub-quadratic
            cross-attention layer optimization to use chunking
        opt_split_attention_invokeai (Union[Unset, bool]): prefer InvokeAI's cross-attention layer optimization for
            automatic choice of optimization Default: False.
        opt_split_attention_v1 (Union[Unset, bool]): prefer older version of split attention optimization for automatic
            choice of optimization Default: False.
        opt_sdp_attention (Union[Unset, bool]): prefer scaled dot product cross-attention layer optimization for
            automatic choice of optimization; requires PyTorch 2.* Default: False.
        opt_sdp_no_mem_attention (Union[Unset, bool]): prefer scaled dot product cross-attention layer optimization
            without memory efficient attention for automatic choice of optimization, makes image generation deterministic;
            requires PyTorch 2.* Default: False.
        disable_opt_split_attention (Union[Unset, bool]): prefer no cross-attention layer optimization for automatic
            choice of optimization Default: False.
        disable_nan_check (Union[Unset, bool]): do not check if produced images/latent spaces have nans; useful for
            running without a checkpoint in CI Default: False.
        use_cpu (Union[Unset, List[Any]]): use CPU as torch device for specified modules
        use_ipex (Union[Unset, bool]): use Intel XPU as torch device Default: False.
        disable_model_loading_ram_optimization (Union[Unset, bool]): disable an optimization that reduces RAM use when
            loading a model Default: False.
        listen (Union[Unset, bool]): launch gradio with 0.0.0.0 as server name, allowing to respond to network requests
            Default: False.
        port (Union[None, Unset, str]): launch gradio with given server port, you need root/admin rights for ports <
            1024, defaults to 7860 if available
        show_negative_prompt (Union[Unset, bool]): does not do anything Default: False.
        ui_config_file (Union[Unset, str]): filename to use for ui configuration Default: '/mnt/sam860evo/forge/stable-
            diffusion-webui-forge/ui-config.json'.
        hide_ui_dir_config (Union[Unset, bool]): hide directory configuration from webui Default: False.
        freeze_settings (Union[Unset, bool]): disable editing of all settings globally Default: False.
        freeze_settings_in_sections (Union[None, Unset, str]): disable editing settings in specific sections of the
            settings page by specifying a comma-delimited list such like "saving-images,upscaling". The list of setting
            names can be found in the modules/shared_options.py file
        freeze_specific_settings (Union[None, Unset, str]): disable editing of individual settings by specifying a
            comma-delimited list like "samples_save,samples_format". The list of setting names can be found in the
            config.json file
        ui_settings_file (Union[Unset, str]): filename to use for ui settings Default: '/mnt/sam860evo/forge/stable-
            diffusion-webui-forge/config.json'.
        gradio_debug (Union[Unset, bool]): launch gradio with --debug option Default: False.
        gradio_auth (Union[None, Unset, str]): set gradio authentication like "username:password"; or comma-delimit
            multiple like "u1:p1,u2:p2,u3:p3"
        gradio_auth_path (Union[None, Unset, str]): set gradio authentication file path ex. "/path/to/auth/file" same
            auth format as --gradio-auth
        gradio_img2img_tool (Union[None, Unset, str]): does not do anything
        gradio_inpaint_tool (Union[None, Unset, str]): does not do anything
        gradio_allowed_path (Union[Unset, List[Any]]): add path to gradio's allowed_paths, make it possible to serve
            files from it
        opt_channelslast (Union[Unset, bool]): change memory type for stable diffusion to channels last Default: False.
        styles_file (Union[Unset, List[Any]]): path or wildcard path of styles files, allow multiple entries.
        autolaunch (Union[Unset, bool]): open the webui URL in the system's default browser upon launch Default: False.
        theme (Union[None, Unset, str]): launches the UI with light or dark theme
        use_textbox_seed (Union[Unset, bool]): use textbox for seeds in UI (no up/down, but possible to input long
            seeds) Default: False.
        disable_console_progressbars (Union[Unset, bool]): do not output progressbars to console Default: False.
        enable_console_prompts (Union[Unset, bool]): does not do anything Default: False.
        vae_path (Union[None, Unset, str]): Checkpoint to use as VAE; setting this argument disables all settings
            related to VAE
        disable_safe_unpickle (Union[Unset, bool]): disable checking pytorch models for malicious code Default: False.
        api (Union[Unset, bool]): use api=True to launch the API together with the webui (use --nowebui instead for only
            the API) Default: False.
        api_auth (Union[None, Unset, str]): Set authentication for API like "username:password"; or comma-delimit
            multiple like "u1:p1,u2:p2,u3:p3"
        api_log (Union[Unset, bool]): use api-log=True to enable logging of all API requests Default: False.
        nowebui (Union[Unset, bool]): use api=True to launch the API instead of the webui Default: False.
        ui_debug_mode (Union[Unset, bool]): Don't load model to quickly launch UI Default: False.
        device_id (Union[None, Unset, str]): Select the default CUDA device to use (export CUDA_VISIBLE_DEVICES=0,1,etc
            might be needed before)
        administrator (Union[Unset, bool]): Administrator rights Default: False.
        cors_allow_origins (Union[None, Unset, str]): Allowed CORS origin(s) in the form of a comma-separated list (no
            spaces)
        cors_allow_origins_regex (Union[None, Unset, str]): Allowed CORS origin(s) in the form of a single regular
            expression
        tls_keyfile (Union[None, Unset, str]): Partially enables TLS, requires --tls-certfile to fully function
        tls_certfile (Union[None, Unset, str]): Partially enables TLS, requires --tls-keyfile to fully function
        disable_tls_verify (Union[None, Unset, str]): When passed, enables the use of self-signed certificates.
        server_name (Union[None, Unset, str]): Sets hostname of server
        gradio_queue (Union[Unset, bool]): does not do anything Default: True.
        no_gradio_queue (Union[Unset, bool]): Disables gradio queue; causes the webpage to use http requests instead of
            websockets; was the default in earlier versions Default: False.
        skip_version_check (Union[Unset, bool]): Do not check versions of torch and xformers Default: False.
        no_hashing (Union[Unset, bool]): disable sha256 hashing of checkpoints to help loading performance Default:
            False.
        no_download_sd_model (Union[Unset, bool]): don't download SD1.5 model even if no model is found in --ckpt-dir
            Default: False.
        subpath (Union[None, Unset, str]): customize the subpath for gradio, use with reverse proxy
        add_stop_route (Union[Unset, bool]): does not do anything Default: False.
        api_server_stop (Union[Unset, bool]): enable server stop/restart/kill via api Default: False.
        timeout_keep_alive (Union[Unset, int]): set timeout_keep_alive for uvicorn Default: 30.
        disable_all_extensions (Union[Unset, bool]): prevent all extensions from running regardless of any other
            settings Default: False.
        disable_extra_extensions (Union[Unset, bool]): prevent all extensions except built-in from running regardless of
            any other settings Default: False.
        skip_load_model_at_start (Union[Unset, bool]): if load a model at web start, only take effect when --nowebui
            Default: False.
        unix_filenames_sanitization (Union[Unset, bool]): allow any symbols except '/' in filenames. May conflict with
            your browser and file system Default: False.
        filenames_max_length (Union[Unset, int]): maximal length of filenames of saved images. If you override it, it
            can conflict with your file system Default: 128.
        no_prompt_history (Union[Unset, bool]): disable read prompt from last generation feature; settings this argument
            will not create '--data_path/params.txt' file Default: False.
        forge_ref_a1111_home (Union[None, Unset, str]): Look for models in an existing A1111 checkout's path
        controlnet_dir (Union[None, Unset, str]): Path to directory with ControlNet models
        controlnet_preprocessor_models_dir (Union[None, Unset, str]): Path to directory with annotator model directories
        scunet_models_path (Union[Unset, str]): Path to directory with ScuNET model file(s). Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/ScuNET'.
        swinir_models_path (Union[Unset, str]): Path to directory with SwinIR model file(s). Default:
            '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/SwinIR'.
        controlnet_loglevel (Union[Unset, str]): Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) Default:
            'INFO'.
        controlnet_tracemalloc (Union[None, Unset, str]): Enable memory tracing.
        lora_dir (Union[Unset, str]): Path to directory with Lora networks. Default: '/mnt/sam860evo/forge/stable-
            diffusion-webui-forge/models/Lora'.
        lyco_dir_backcompat (Union[Unset, str]): Path to directory with LyCORIS networks (for backawards compatibility;
            can also use --lyco-dir). Default: '/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/LyCORIS'.
    """

    gpu_device_id: Union[None, Unset, str] = UNSET
    all_in_fp32: Union[Unset, bool] = False
    all_in_fp16: Union[Unset, bool] = False
    unet_in_bf16: Union[Unset, bool] = False
    unet_in_fp16: Union[Unset, bool] = False
    unet_in_fp8_e4m3fn: Union[Unset, bool] = False
    unet_in_fp8_e5m2: Union[Unset, bool] = False
    vae_in_fp16: Union[Unset, bool] = False
    vae_in_fp32: Union[Unset, bool] = False
    vae_in_bf16: Union[Unset, bool] = False
    vae_in_cpu: Union[Unset, bool] = False
    clip_in_fp8_e4m3fn: Union[Unset, bool] = False
    clip_in_fp8_e5m2: Union[Unset, bool] = False
    clip_in_fp16: Union[Unset, bool] = False
    clip_in_fp32: Union[Unset, bool] = False
    attention_split: Union[Unset, bool] = False
    attention_quad: Union[Unset, bool] = False
    attention_pytorch: Union[Unset, bool] = False
    force_upcast_attention: Union[Unset, bool] = False
    disable_attention_upcast: Union[Unset, bool] = False
    disable_xformers: Union[Unset, bool] = False
    directml: Union[None, Unset, str] = UNSET
    disable_ipex_hijack: Union[Unset, bool] = False
    always_gpu: Union[Unset, bool] = False
    always_high_vram: Union[Unset, bool] = False
    always_normal_vram: Union[Unset, bool] = False
    always_low_vram: Union[Unset, bool] = False
    always_no_vram: Union[Unset, bool] = False
    always_cpu: Union[Unset, bool] = False
    always_offload_from_vram: Union[Unset, bool] = False
    pytorch_deterministic: Union[Unset, bool] = False
    cuda_malloc: Union[Unset, bool] = False
    cuda_stream: Union[Unset, bool] = False
    pin_shared_memory: Union[Unset, bool] = False
    disable_gpu_warning: Union[Unset, bool] = False
    f: Union[Unset, bool] = False
    update_all_extensions: Union[Unset, bool] = False
    skip_python_version_check: Union[Unset, bool] = False
    skip_torch_cuda_test: Union[Unset, bool] = False
    reinstall_xformers: Union[Unset, bool] = False
    reinstall_torch: Union[Unset, bool] = False
    update_check: Union[Unset, bool] = False
    test_server: Union[Unset, bool] = False
    log_startup: Union[Unset, bool] = False
    skip_prepare_environment: Union[Unset, bool] = False
    skip_install: Union[Unset, bool] = False
    dump_sysinfo: Union[Unset, bool] = False
    loglevel: Union[None, Unset, str] = UNSET
    do_not_download_clip: Union[Unset, bool] = False
    data_dir: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge"
    models_dir: Union[None, Unset, str] = UNSET
    config: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/configs/v1-inference.yaml"
    ckpt: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/model.ckpt"
    ckpt_dir: Union[None, Unset, str] = UNSET
    vae_dir: Union[None, Unset, str] = UNSET
    text_encoder_dir: Union[None, Unset, str] = UNSET
    gfpgan_dir: Union[Unset, str] = "./GFPGAN"
    gfpgan_model: Union[None, Unset, str] = UNSET
    no_half: Union[Unset, bool] = False
    no_half_vae: Union[Unset, bool] = False
    no_progressbar_hiding: Union[Unset, bool] = False
    max_batch_count: Union[Unset, int] = 16
    embeddings_dir: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/embeddings"
    textual_inversion_templates_dir: Union[Unset, str] = (
        "/mnt/sam860evo/forge/stable-diffusion-webui-forge/textual_inversion_templates"
    )
    hypernetwork_dir: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/hypernetworks"
    localizations_dir: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/localizations"
    allow_code: Union[Unset, bool] = False
    medvram: Union[Unset, bool] = False
    medvram_sdxl: Union[Unset, bool] = False
    lowvram: Union[Unset, bool] = False
    lowram: Union[Unset, bool] = False
    always_batch_cond_uncond: Union[Unset, bool] = False
    unload_gfpgan: Union[Unset, bool] = False
    precision: Union[Unset, str] = "autocast"
    upcast_sampling: Union[Unset, bool] = False
    share: Union[Unset, bool] = False
    ngrok: Union[None, Unset, str] = UNSET
    ngrok_region: Union[Unset, str] = ""
    ngrok_options: Union[Unset, "FlagsNgrokOptions"] = UNSET
    enable_insecure_extension_access: Union[Unset, bool] = False
    codeformer_models_path: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/Codeformer"
    gfpgan_models_path: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/GFPGAN"
    esrgan_models_path: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/ESRGAN"
    bsrgan_models_path: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/BSRGAN"
    realesrgan_models_path: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/RealESRGAN"
    dat_models_path: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/DAT"
    clip_models_path: Union[None, Unset, str] = UNSET
    xformers: Union[Unset, bool] = False
    force_enable_xformers: Union[Unset, bool] = False
    xformers_flash_attention: Union[Unset, bool] = False
    deepdanbooru: Union[Unset, bool] = False
    opt_split_attention: Union[Unset, bool] = False
    opt_sub_quad_attention: Union[Unset, bool] = False
    sub_quad_q_chunk_size: Union[Unset, int] = 1024
    sub_quad_kv_chunk_size: Union[None, Unset, str] = UNSET
    sub_quad_chunk_threshold: Union[None, Unset, str] = UNSET
    opt_split_attention_invokeai: Union[Unset, bool] = False
    opt_split_attention_v1: Union[Unset, bool] = False
    opt_sdp_attention: Union[Unset, bool] = False
    opt_sdp_no_mem_attention: Union[Unset, bool] = False
    disable_opt_split_attention: Union[Unset, bool] = False
    disable_nan_check: Union[Unset, bool] = False
    use_cpu: Union[Unset, List[Any]] = UNSET
    use_ipex: Union[Unset, bool] = False
    disable_model_loading_ram_optimization: Union[Unset, bool] = False
    listen: Union[Unset, bool] = False
    port: Union[None, Unset, str] = UNSET
    show_negative_prompt: Union[Unset, bool] = False
    ui_config_file: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/ui-config.json"
    hide_ui_dir_config: Union[Unset, bool] = False
    freeze_settings: Union[Unset, bool] = False
    freeze_settings_in_sections: Union[None, Unset, str] = UNSET
    freeze_specific_settings: Union[None, Unset, str] = UNSET
    ui_settings_file: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/config.json"
    gradio_debug: Union[Unset, bool] = False
    gradio_auth: Union[None, Unset, str] = UNSET
    gradio_auth_path: Union[None, Unset, str] = UNSET
    gradio_img2img_tool: Union[None, Unset, str] = UNSET
    gradio_inpaint_tool: Union[None, Unset, str] = UNSET
    gradio_allowed_path: Union[Unset, List[Any]] = UNSET
    opt_channelslast: Union[Unset, bool] = False
    styles_file: Union[Unset, List[Any]] = UNSET
    autolaunch: Union[Unset, bool] = False
    theme: Union[None, Unset, str] = UNSET
    use_textbox_seed: Union[Unset, bool] = False
    disable_console_progressbars: Union[Unset, bool] = False
    enable_console_prompts: Union[Unset, bool] = False
    vae_path: Union[None, Unset, str] = UNSET
    disable_safe_unpickle: Union[Unset, bool] = False
    api: Union[Unset, bool] = False
    api_auth: Union[None, Unset, str] = UNSET
    api_log: Union[Unset, bool] = False
    nowebui: Union[Unset, bool] = False
    ui_debug_mode: Union[Unset, bool] = False
    device_id: Union[None, Unset, str] = UNSET
    administrator: Union[Unset, bool] = False
    cors_allow_origins: Union[None, Unset, str] = UNSET
    cors_allow_origins_regex: Union[None, Unset, str] = UNSET
    tls_keyfile: Union[None, Unset, str] = UNSET
    tls_certfile: Union[None, Unset, str] = UNSET
    disable_tls_verify: Union[None, Unset, str] = UNSET
    server_name: Union[None, Unset, str] = UNSET
    gradio_queue: Union[Unset, bool] = True
    no_gradio_queue: Union[Unset, bool] = False
    skip_version_check: Union[Unset, bool] = False
    no_hashing: Union[Unset, bool] = False
    no_download_sd_model: Union[Unset, bool] = False
    subpath: Union[None, Unset, str] = UNSET
    add_stop_route: Union[Unset, bool] = False
    api_server_stop: Union[Unset, bool] = False
    timeout_keep_alive: Union[Unset, int] = 30
    disable_all_extensions: Union[Unset, bool] = False
    disable_extra_extensions: Union[Unset, bool] = False
    skip_load_model_at_start: Union[Unset, bool] = False
    unix_filenames_sanitization: Union[Unset, bool] = False
    filenames_max_length: Union[Unset, int] = 128
    no_prompt_history: Union[Unset, bool] = False
    forge_ref_a1111_home: Union[None, Unset, str] = UNSET
    controlnet_dir: Union[None, Unset, str] = UNSET
    controlnet_preprocessor_models_dir: Union[None, Unset, str] = UNSET
    scunet_models_path: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/ScuNET"
    swinir_models_path: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/SwinIR"
    controlnet_loglevel: Union[Unset, str] = "INFO"
    controlnet_tracemalloc: Union[None, Unset, str] = UNSET
    lora_dir: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/Lora"
    lyco_dir_backcompat: Union[Unset, str] = "/mnt/sam860evo/forge/stable-diffusion-webui-forge/models/LyCORIS"
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        gpu_device_id: Union[None, Unset, str]
        if isinstance(self.gpu_device_id, Unset):
            gpu_device_id = UNSET
        else:
            gpu_device_id = self.gpu_device_id

        all_in_fp32 = self.all_in_fp32

        all_in_fp16 = self.all_in_fp16

        unet_in_bf16 = self.unet_in_bf16

        unet_in_fp16 = self.unet_in_fp16

        unet_in_fp8_e4m3fn = self.unet_in_fp8_e4m3fn

        unet_in_fp8_e5m2 = self.unet_in_fp8_e5m2

        vae_in_fp16 = self.vae_in_fp16

        vae_in_fp32 = self.vae_in_fp32

        vae_in_bf16 = self.vae_in_bf16

        vae_in_cpu = self.vae_in_cpu

        clip_in_fp8_e4m3fn = self.clip_in_fp8_e4m3fn

        clip_in_fp8_e5m2 = self.clip_in_fp8_e5m2

        clip_in_fp16 = self.clip_in_fp16

        clip_in_fp32 = self.clip_in_fp32

        attention_split = self.attention_split

        attention_quad = self.attention_quad

        attention_pytorch = self.attention_pytorch

        force_upcast_attention = self.force_upcast_attention

        disable_attention_upcast = self.disable_attention_upcast

        disable_xformers = self.disable_xformers

        directml: Union[None, Unset, str]
        if isinstance(self.directml, Unset):
            directml = UNSET
        else:
            directml = self.directml

        disable_ipex_hijack = self.disable_ipex_hijack

        always_gpu = self.always_gpu

        always_high_vram = self.always_high_vram

        always_normal_vram = self.always_normal_vram

        always_low_vram = self.always_low_vram

        always_no_vram = self.always_no_vram

        always_cpu = self.always_cpu

        always_offload_from_vram = self.always_offload_from_vram

        pytorch_deterministic = self.pytorch_deterministic

        cuda_malloc = self.cuda_malloc

        cuda_stream = self.cuda_stream

        pin_shared_memory = self.pin_shared_memory

        disable_gpu_warning = self.disable_gpu_warning

        f = self.f

        update_all_extensions = self.update_all_extensions

        skip_python_version_check = self.skip_python_version_check

        skip_torch_cuda_test = self.skip_torch_cuda_test

        reinstall_xformers = self.reinstall_xformers

        reinstall_torch = self.reinstall_torch

        update_check = self.update_check

        test_server = self.test_server

        log_startup = self.log_startup

        skip_prepare_environment = self.skip_prepare_environment

        skip_install = self.skip_install

        dump_sysinfo = self.dump_sysinfo

        loglevel: Union[None, Unset, str]
        if isinstance(self.loglevel, Unset):
            loglevel = UNSET
        else:
            loglevel = self.loglevel

        do_not_download_clip = self.do_not_download_clip

        data_dir = self.data_dir

        models_dir: Union[None, Unset, str]
        if isinstance(self.models_dir, Unset):
            models_dir = UNSET
        else:
            models_dir = self.models_dir

        config = self.config

        ckpt = self.ckpt

        ckpt_dir: Union[None, Unset, str]
        if isinstance(self.ckpt_dir, Unset):
            ckpt_dir = UNSET
        else:
            ckpt_dir = self.ckpt_dir

        vae_dir: Union[None, Unset, str]
        if isinstance(self.vae_dir, Unset):
            vae_dir = UNSET
        else:
            vae_dir = self.vae_dir

        text_encoder_dir: Union[None, Unset, str]
        if isinstance(self.text_encoder_dir, Unset):
            text_encoder_dir = UNSET
        else:
            text_encoder_dir = self.text_encoder_dir

        gfpgan_dir = self.gfpgan_dir

        gfpgan_model: Union[None, Unset, str]
        if isinstance(self.gfpgan_model, Unset):
            gfpgan_model = UNSET
        else:
            gfpgan_model = self.gfpgan_model

        no_half = self.no_half

        no_half_vae = self.no_half_vae

        no_progressbar_hiding = self.no_progressbar_hiding

        max_batch_count = self.max_batch_count

        embeddings_dir = self.embeddings_dir

        textual_inversion_templates_dir = self.textual_inversion_templates_dir

        hypernetwork_dir = self.hypernetwork_dir

        localizations_dir = self.localizations_dir

        allow_code = self.allow_code

        medvram = self.medvram

        medvram_sdxl = self.medvram_sdxl

        lowvram = self.lowvram

        lowram = self.lowram

        always_batch_cond_uncond = self.always_batch_cond_uncond

        unload_gfpgan = self.unload_gfpgan

        precision = self.precision

        upcast_sampling = self.upcast_sampling

        share = self.share

        ngrok: Union[None, Unset, str]
        if isinstance(self.ngrok, Unset):
            ngrok = UNSET
        else:
            ngrok = self.ngrok

        ngrok_region = self.ngrok_region

        ngrok_options: Union[Unset, Dict[str, Any]] = UNSET
        if not isinstance(self.ngrok_options, Unset):
            ngrok_options = self.ngrok_options.to_dict()

        enable_insecure_extension_access = self.enable_insecure_extension_access

        codeformer_models_path = self.codeformer_models_path

        gfpgan_models_path = self.gfpgan_models_path

        esrgan_models_path = self.esrgan_models_path

        bsrgan_models_path = self.bsrgan_models_path

        realesrgan_models_path = self.realesrgan_models_path

        dat_models_path = self.dat_models_path

        clip_models_path: Union[None, Unset, str]
        if isinstance(self.clip_models_path, Unset):
            clip_models_path = UNSET
        else:
            clip_models_path = self.clip_models_path

        xformers = self.xformers

        force_enable_xformers = self.force_enable_xformers

        xformers_flash_attention = self.xformers_flash_attention

        deepdanbooru = self.deepdanbooru

        opt_split_attention = self.opt_split_attention

        opt_sub_quad_attention = self.opt_sub_quad_attention

        sub_quad_q_chunk_size = self.sub_quad_q_chunk_size

        sub_quad_kv_chunk_size: Union[None, Unset, str]
        if isinstance(self.sub_quad_kv_chunk_size, Unset):
            sub_quad_kv_chunk_size = UNSET
        else:
            sub_quad_kv_chunk_size = self.sub_quad_kv_chunk_size

        sub_quad_chunk_threshold: Union[None, Unset, str]
        if isinstance(self.sub_quad_chunk_threshold, Unset):
            sub_quad_chunk_threshold = UNSET
        else:
            sub_quad_chunk_threshold = self.sub_quad_chunk_threshold

        opt_split_attention_invokeai = self.opt_split_attention_invokeai

        opt_split_attention_v1 = self.opt_split_attention_v1

        opt_sdp_attention = self.opt_sdp_attention

        opt_sdp_no_mem_attention = self.opt_sdp_no_mem_attention

        disable_opt_split_attention = self.disable_opt_split_attention

        disable_nan_check = self.disable_nan_check

        use_cpu: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.use_cpu, Unset):
            use_cpu = self.use_cpu

        use_ipex = self.use_ipex

        disable_model_loading_ram_optimization = self.disable_model_loading_ram_optimization

        listen = self.listen

        port: Union[None, Unset, str]
        if isinstance(self.port, Unset):
            port = UNSET
        else:
            port = self.port

        show_negative_prompt = self.show_negative_prompt

        ui_config_file = self.ui_config_file

        hide_ui_dir_config = self.hide_ui_dir_config

        freeze_settings = self.freeze_settings

        freeze_settings_in_sections: Union[None, Unset, str]
        if isinstance(self.freeze_settings_in_sections, Unset):
            freeze_settings_in_sections = UNSET
        else:
            freeze_settings_in_sections = self.freeze_settings_in_sections

        freeze_specific_settings: Union[None, Unset, str]
        if isinstance(self.freeze_specific_settings, Unset):
            freeze_specific_settings = UNSET
        else:
            freeze_specific_settings = self.freeze_specific_settings

        ui_settings_file = self.ui_settings_file

        gradio_debug = self.gradio_debug

        gradio_auth: Union[None, Unset, str]
        if isinstance(self.gradio_auth, Unset):
            gradio_auth = UNSET
        else:
            gradio_auth = self.gradio_auth

        gradio_auth_path: Union[None, Unset, str]
        if isinstance(self.gradio_auth_path, Unset):
            gradio_auth_path = UNSET
        else:
            gradio_auth_path = self.gradio_auth_path

        gradio_img2img_tool: Union[None, Unset, str]
        if isinstance(self.gradio_img2img_tool, Unset):
            gradio_img2img_tool = UNSET
        else:
            gradio_img2img_tool = self.gradio_img2img_tool

        gradio_inpaint_tool: Union[None, Unset, str]
        if isinstance(self.gradio_inpaint_tool, Unset):
            gradio_inpaint_tool = UNSET
        else:
            gradio_inpaint_tool = self.gradio_inpaint_tool

        gradio_allowed_path: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.gradio_allowed_path, Unset):
            gradio_allowed_path = self.gradio_allowed_path

        opt_channelslast = self.opt_channelslast

        styles_file: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.styles_file, Unset):
            styles_file = self.styles_file

        autolaunch = self.autolaunch

        theme: Union[None, Unset, str]
        if isinstance(self.theme, Unset):
            theme = UNSET
        else:
            theme = self.theme

        use_textbox_seed = self.use_textbox_seed

        disable_console_progressbars = self.disable_console_progressbars

        enable_console_prompts = self.enable_console_prompts

        vae_path: Union[None, Unset, str]
        if isinstance(self.vae_path, Unset):
            vae_path = UNSET
        else:
            vae_path = self.vae_path

        disable_safe_unpickle = self.disable_safe_unpickle

        api = self.api

        api_auth: Union[None, Unset, str]
        if isinstance(self.api_auth, Unset):
            api_auth = UNSET
        else:
            api_auth = self.api_auth

        api_log = self.api_log

        nowebui = self.nowebui

        ui_debug_mode = self.ui_debug_mode

        device_id: Union[None, Unset, str]
        if isinstance(self.device_id, Unset):
            device_id = UNSET
        else:
            device_id = self.device_id

        administrator = self.administrator

        cors_allow_origins: Union[None, Unset, str]
        if isinstance(self.cors_allow_origins, Unset):
            cors_allow_origins = UNSET
        else:
            cors_allow_origins = self.cors_allow_origins

        cors_allow_origins_regex: Union[None, Unset, str]
        if isinstance(self.cors_allow_origins_regex, Unset):
            cors_allow_origins_regex = UNSET
        else:
            cors_allow_origins_regex = self.cors_allow_origins_regex

        tls_keyfile: Union[None, Unset, str]
        if isinstance(self.tls_keyfile, Unset):
            tls_keyfile = UNSET
        else:
            tls_keyfile = self.tls_keyfile

        tls_certfile: Union[None, Unset, str]
        if isinstance(self.tls_certfile, Unset):
            tls_certfile = UNSET
        else:
            tls_certfile = self.tls_certfile

        disable_tls_verify: Union[None, Unset, str]
        if isinstance(self.disable_tls_verify, Unset):
            disable_tls_verify = UNSET
        else:
            disable_tls_verify = self.disable_tls_verify

        server_name: Union[None, Unset, str]
        if isinstance(self.server_name, Unset):
            server_name = UNSET
        else:
            server_name = self.server_name

        gradio_queue = self.gradio_queue

        no_gradio_queue = self.no_gradio_queue

        skip_version_check = self.skip_version_check

        no_hashing = self.no_hashing

        no_download_sd_model = self.no_download_sd_model

        subpath: Union[None, Unset, str]
        if isinstance(self.subpath, Unset):
            subpath = UNSET
        else:
            subpath = self.subpath

        add_stop_route = self.add_stop_route

        api_server_stop = self.api_server_stop

        timeout_keep_alive = self.timeout_keep_alive

        disable_all_extensions = self.disable_all_extensions

        disable_extra_extensions = self.disable_extra_extensions

        skip_load_model_at_start = self.skip_load_model_at_start

        unix_filenames_sanitization = self.unix_filenames_sanitization

        filenames_max_length = self.filenames_max_length

        no_prompt_history = self.no_prompt_history

        forge_ref_a1111_home: Union[None, Unset, str]
        if isinstance(self.forge_ref_a1111_home, Unset):
            forge_ref_a1111_home = UNSET
        else:
            forge_ref_a1111_home = self.forge_ref_a1111_home

        controlnet_dir: Union[None, Unset, str]
        if isinstance(self.controlnet_dir, Unset):
            controlnet_dir = UNSET
        else:
            controlnet_dir = self.controlnet_dir

        controlnet_preprocessor_models_dir: Union[None, Unset, str]
        if isinstance(self.controlnet_preprocessor_models_dir, Unset):
            controlnet_preprocessor_models_dir = UNSET
        else:
            controlnet_preprocessor_models_dir = self.controlnet_preprocessor_models_dir

        scunet_models_path = self.scunet_models_path

        swinir_models_path = self.swinir_models_path

        controlnet_loglevel = self.controlnet_loglevel

        controlnet_tracemalloc: Union[None, Unset, str]
        if isinstance(self.controlnet_tracemalloc, Unset):
            controlnet_tracemalloc = UNSET
        else:
            controlnet_tracemalloc = self.controlnet_tracemalloc

        lora_dir = self.lora_dir

        lyco_dir_backcompat = self.lyco_dir_backcompat

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if gpu_device_id is not UNSET:
            field_dict["gpu_device_id"] = gpu_device_id
        if all_in_fp32 is not UNSET:
            field_dict["all_in_fp32"] = all_in_fp32
        if all_in_fp16 is not UNSET:
            field_dict["all_in_fp16"] = all_in_fp16
        if unet_in_bf16 is not UNSET:
            field_dict["unet_in_bf16"] = unet_in_bf16
        if unet_in_fp16 is not UNSET:
            field_dict["unet_in_fp16"] = unet_in_fp16
        if unet_in_fp8_e4m3fn is not UNSET:
            field_dict["unet_in_fp8_e4m3fn"] = unet_in_fp8_e4m3fn
        if unet_in_fp8_e5m2 is not UNSET:
            field_dict["unet_in_fp8_e5m2"] = unet_in_fp8_e5m2
        if vae_in_fp16 is not UNSET:
            field_dict["vae_in_fp16"] = vae_in_fp16
        if vae_in_fp32 is not UNSET:
            field_dict["vae_in_fp32"] = vae_in_fp32
        if vae_in_bf16 is not UNSET:
            field_dict["vae_in_bf16"] = vae_in_bf16
        if vae_in_cpu is not UNSET:
            field_dict["vae_in_cpu"] = vae_in_cpu
        if clip_in_fp8_e4m3fn is not UNSET:
            field_dict["clip_in_fp8_e4m3fn"] = clip_in_fp8_e4m3fn
        if clip_in_fp8_e5m2 is not UNSET:
            field_dict["clip_in_fp8_e5m2"] = clip_in_fp8_e5m2
        if clip_in_fp16 is not UNSET:
            field_dict["clip_in_fp16"] = clip_in_fp16
        if clip_in_fp32 is not UNSET:
            field_dict["clip_in_fp32"] = clip_in_fp32
        if attention_split is not UNSET:
            field_dict["attention_split"] = attention_split
        if attention_quad is not UNSET:
            field_dict["attention_quad"] = attention_quad
        if attention_pytorch is not UNSET:
            field_dict["attention_pytorch"] = attention_pytorch
        if force_upcast_attention is not UNSET:
            field_dict["force_upcast_attention"] = force_upcast_attention
        if disable_attention_upcast is not UNSET:
            field_dict["disable_attention_upcast"] = disable_attention_upcast
        if disable_xformers is not UNSET:
            field_dict["disable_xformers"] = disable_xformers
        if directml is not UNSET:
            field_dict["directml"] = directml
        if disable_ipex_hijack is not UNSET:
            field_dict["disable_ipex_hijack"] = disable_ipex_hijack
        if always_gpu is not UNSET:
            field_dict["always_gpu"] = always_gpu
        if always_high_vram is not UNSET:
            field_dict["always_high_vram"] = always_high_vram
        if always_normal_vram is not UNSET:
            field_dict["always_normal_vram"] = always_normal_vram
        if always_low_vram is not UNSET:
            field_dict["always_low_vram"] = always_low_vram
        if always_no_vram is not UNSET:
            field_dict["always_no_vram"] = always_no_vram
        if always_cpu is not UNSET:
            field_dict["always_cpu"] = always_cpu
        if always_offload_from_vram is not UNSET:
            field_dict["always_offload_from_vram"] = always_offload_from_vram
        if pytorch_deterministic is not UNSET:
            field_dict["pytorch_deterministic"] = pytorch_deterministic
        if cuda_malloc is not UNSET:
            field_dict["cuda_malloc"] = cuda_malloc
        if cuda_stream is not UNSET:
            field_dict["cuda_stream"] = cuda_stream
        if pin_shared_memory is not UNSET:
            field_dict["pin_shared_memory"] = pin_shared_memory
        if disable_gpu_warning is not UNSET:
            field_dict["disable_gpu_warning"] = disable_gpu_warning
        if f is not UNSET:
            field_dict["f"] = f
        if update_all_extensions is not UNSET:
            field_dict["update_all_extensions"] = update_all_extensions
        if skip_python_version_check is not UNSET:
            field_dict["skip_python_version_check"] = skip_python_version_check
        if skip_torch_cuda_test is not UNSET:
            field_dict["skip_torch_cuda_test"] = skip_torch_cuda_test
        if reinstall_xformers is not UNSET:
            field_dict["reinstall_xformers"] = reinstall_xformers
        if reinstall_torch is not UNSET:
            field_dict["reinstall_torch"] = reinstall_torch
        if update_check is not UNSET:
            field_dict["update_check"] = update_check
        if test_server is not UNSET:
            field_dict["test_server"] = test_server
        if log_startup is not UNSET:
            field_dict["log_startup"] = log_startup
        if skip_prepare_environment is not UNSET:
            field_dict["skip_prepare_environment"] = skip_prepare_environment
        if skip_install is not UNSET:
            field_dict["skip_install"] = skip_install
        if dump_sysinfo is not UNSET:
            field_dict["dump_sysinfo"] = dump_sysinfo
        if loglevel is not UNSET:
            field_dict["loglevel"] = loglevel
        if do_not_download_clip is not UNSET:
            field_dict["do_not_download_clip"] = do_not_download_clip
        if data_dir is not UNSET:
            field_dict["data_dir"] = data_dir
        if models_dir is not UNSET:
            field_dict["models_dir"] = models_dir
        if config is not UNSET:
            field_dict["config"] = config
        if ckpt is not UNSET:
            field_dict["ckpt"] = ckpt
        if ckpt_dir is not UNSET:
            field_dict["ckpt_dir"] = ckpt_dir
        if vae_dir is not UNSET:
            field_dict["vae_dir"] = vae_dir
        if text_encoder_dir is not UNSET:
            field_dict["text_encoder_dir"] = text_encoder_dir
        if gfpgan_dir is not UNSET:
            field_dict["gfpgan_dir"] = gfpgan_dir
        if gfpgan_model is not UNSET:
            field_dict["gfpgan_model"] = gfpgan_model
        if no_half is not UNSET:
            field_dict["no_half"] = no_half
        if no_half_vae is not UNSET:
            field_dict["no_half_vae"] = no_half_vae
        if no_progressbar_hiding is not UNSET:
            field_dict["no_progressbar_hiding"] = no_progressbar_hiding
        if max_batch_count is not UNSET:
            field_dict["max_batch_count"] = max_batch_count
        if embeddings_dir is not UNSET:
            field_dict["embeddings_dir"] = embeddings_dir
        if textual_inversion_templates_dir is not UNSET:
            field_dict["textual_inversion_templates_dir"] = textual_inversion_templates_dir
        if hypernetwork_dir is not UNSET:
            field_dict["hypernetwork_dir"] = hypernetwork_dir
        if localizations_dir is not UNSET:
            field_dict["localizations_dir"] = localizations_dir
        if allow_code is not UNSET:
            field_dict["allow_code"] = allow_code
        if medvram is not UNSET:
            field_dict["medvram"] = medvram
        if medvram_sdxl is not UNSET:
            field_dict["medvram_sdxl"] = medvram_sdxl
        if lowvram is not UNSET:
            field_dict["lowvram"] = lowvram
        if lowram is not UNSET:
            field_dict["lowram"] = lowram
        if always_batch_cond_uncond is not UNSET:
            field_dict["always_batch_cond_uncond"] = always_batch_cond_uncond
        if unload_gfpgan is not UNSET:
            field_dict["unload_gfpgan"] = unload_gfpgan
        if precision is not UNSET:
            field_dict["precision"] = precision
        if upcast_sampling is not UNSET:
            field_dict["upcast_sampling"] = upcast_sampling
        if share is not UNSET:
            field_dict["share"] = share
        if ngrok is not UNSET:
            field_dict["ngrok"] = ngrok
        if ngrok_region is not UNSET:
            field_dict["ngrok_region"] = ngrok_region
        if ngrok_options is not UNSET:
            field_dict["ngrok_options"] = ngrok_options
        if enable_insecure_extension_access is not UNSET:
            field_dict["enable_insecure_extension_access"] = enable_insecure_extension_access
        if codeformer_models_path is not UNSET:
            field_dict["codeformer_models_path"] = codeformer_models_path
        if gfpgan_models_path is not UNSET:
            field_dict["gfpgan_models_path"] = gfpgan_models_path
        if esrgan_models_path is not UNSET:
            field_dict["esrgan_models_path"] = esrgan_models_path
        if bsrgan_models_path is not UNSET:
            field_dict["bsrgan_models_path"] = bsrgan_models_path
        if realesrgan_models_path is not UNSET:
            field_dict["realesrgan_models_path"] = realesrgan_models_path
        if dat_models_path is not UNSET:
            field_dict["dat_models_path"] = dat_models_path
        if clip_models_path is not UNSET:
            field_dict["clip_models_path"] = clip_models_path
        if xformers is not UNSET:
            field_dict["xformers"] = xformers
        if force_enable_xformers is not UNSET:
            field_dict["force_enable_xformers"] = force_enable_xformers
        if xformers_flash_attention is not UNSET:
            field_dict["xformers_flash_attention"] = xformers_flash_attention
        if deepdanbooru is not UNSET:
            field_dict["deepdanbooru"] = deepdanbooru
        if opt_split_attention is not UNSET:
            field_dict["opt_split_attention"] = opt_split_attention
        if opt_sub_quad_attention is not UNSET:
            field_dict["opt_sub_quad_attention"] = opt_sub_quad_attention
        if sub_quad_q_chunk_size is not UNSET:
            field_dict["sub_quad_q_chunk_size"] = sub_quad_q_chunk_size
        if sub_quad_kv_chunk_size is not UNSET:
            field_dict["sub_quad_kv_chunk_size"] = sub_quad_kv_chunk_size
        if sub_quad_chunk_threshold is not UNSET:
            field_dict["sub_quad_chunk_threshold"] = sub_quad_chunk_threshold
        if opt_split_attention_invokeai is not UNSET:
            field_dict["opt_split_attention_invokeai"] = opt_split_attention_invokeai
        if opt_split_attention_v1 is not UNSET:
            field_dict["opt_split_attention_v1"] = opt_split_attention_v1
        if opt_sdp_attention is not UNSET:
            field_dict["opt_sdp_attention"] = opt_sdp_attention
        if opt_sdp_no_mem_attention is not UNSET:
            field_dict["opt_sdp_no_mem_attention"] = opt_sdp_no_mem_attention
        if disable_opt_split_attention is not UNSET:
            field_dict["disable_opt_split_attention"] = disable_opt_split_attention
        if disable_nan_check is not UNSET:
            field_dict["disable_nan_check"] = disable_nan_check
        if use_cpu is not UNSET:
            field_dict["use_cpu"] = use_cpu
        if use_ipex is not UNSET:
            field_dict["use_ipex"] = use_ipex
        if disable_model_loading_ram_optimization is not UNSET:
            field_dict["disable_model_loading_ram_optimization"] = disable_model_loading_ram_optimization
        if listen is not UNSET:
            field_dict["listen"] = listen
        if port is not UNSET:
            field_dict["port"] = port
        if show_negative_prompt is not UNSET:
            field_dict["show_negative_prompt"] = show_negative_prompt
        if ui_config_file is not UNSET:
            field_dict["ui_config_file"] = ui_config_file
        if hide_ui_dir_config is not UNSET:
            field_dict["hide_ui_dir_config"] = hide_ui_dir_config
        if freeze_settings is not UNSET:
            field_dict["freeze_settings"] = freeze_settings
        if freeze_settings_in_sections is not UNSET:
            field_dict["freeze_settings_in_sections"] = freeze_settings_in_sections
        if freeze_specific_settings is not UNSET:
            field_dict["freeze_specific_settings"] = freeze_specific_settings
        if ui_settings_file is not UNSET:
            field_dict["ui_settings_file"] = ui_settings_file
        if gradio_debug is not UNSET:
            field_dict["gradio_debug"] = gradio_debug
        if gradio_auth is not UNSET:
            field_dict["gradio_auth"] = gradio_auth
        if gradio_auth_path is not UNSET:
            field_dict["gradio_auth_path"] = gradio_auth_path
        if gradio_img2img_tool is not UNSET:
            field_dict["gradio_img2img_tool"] = gradio_img2img_tool
        if gradio_inpaint_tool is not UNSET:
            field_dict["gradio_inpaint_tool"] = gradio_inpaint_tool
        if gradio_allowed_path is not UNSET:
            field_dict["gradio_allowed_path"] = gradio_allowed_path
        if opt_channelslast is not UNSET:
            field_dict["opt_channelslast"] = opt_channelslast
        if styles_file is not UNSET:
            field_dict["styles_file"] = styles_file
        if autolaunch is not UNSET:
            field_dict["autolaunch"] = autolaunch
        if theme is not UNSET:
            field_dict["theme"] = theme
        if use_textbox_seed is not UNSET:
            field_dict["use_textbox_seed"] = use_textbox_seed
        if disable_console_progressbars is not UNSET:
            field_dict["disable_console_progressbars"] = disable_console_progressbars
        if enable_console_prompts is not UNSET:
            field_dict["enable_console_prompts"] = enable_console_prompts
        if vae_path is not UNSET:
            field_dict["vae_path"] = vae_path
        if disable_safe_unpickle is not UNSET:
            field_dict["disable_safe_unpickle"] = disable_safe_unpickle
        if api is not UNSET:
            field_dict["api"] = api
        if api_auth is not UNSET:
            field_dict["api_auth"] = api_auth
        if api_log is not UNSET:
            field_dict["api_log"] = api_log
        if nowebui is not UNSET:
            field_dict["nowebui"] = nowebui
        if ui_debug_mode is not UNSET:
            field_dict["ui_debug_mode"] = ui_debug_mode
        if device_id is not UNSET:
            field_dict["device_id"] = device_id
        if administrator is not UNSET:
            field_dict["administrator"] = administrator
        if cors_allow_origins is not UNSET:
            field_dict["cors_allow_origins"] = cors_allow_origins
        if cors_allow_origins_regex is not UNSET:
            field_dict["cors_allow_origins_regex"] = cors_allow_origins_regex
        if tls_keyfile is not UNSET:
            field_dict["tls_keyfile"] = tls_keyfile
        if tls_certfile is not UNSET:
            field_dict["tls_certfile"] = tls_certfile
        if disable_tls_verify is not UNSET:
            field_dict["disable_tls_verify"] = disable_tls_verify
        if server_name is not UNSET:
            field_dict["server_name"] = server_name
        if gradio_queue is not UNSET:
            field_dict["gradio_queue"] = gradio_queue
        if no_gradio_queue is not UNSET:
            field_dict["no_gradio_queue"] = no_gradio_queue
        if skip_version_check is not UNSET:
            field_dict["skip_version_check"] = skip_version_check
        if no_hashing is not UNSET:
            field_dict["no_hashing"] = no_hashing
        if no_download_sd_model is not UNSET:
            field_dict["no_download_sd_model"] = no_download_sd_model
        if subpath is not UNSET:
            field_dict["subpath"] = subpath
        if add_stop_route is not UNSET:
            field_dict["add_stop_route"] = add_stop_route
        if api_server_stop is not UNSET:
            field_dict["api_server_stop"] = api_server_stop
        if timeout_keep_alive is not UNSET:
            field_dict["timeout_keep_alive"] = timeout_keep_alive
        if disable_all_extensions is not UNSET:
            field_dict["disable_all_extensions"] = disable_all_extensions
        if disable_extra_extensions is not UNSET:
            field_dict["disable_extra_extensions"] = disable_extra_extensions
        if skip_load_model_at_start is not UNSET:
            field_dict["skip_load_model_at_start"] = skip_load_model_at_start
        if unix_filenames_sanitization is not UNSET:
            field_dict["unix_filenames_sanitization"] = unix_filenames_sanitization
        if filenames_max_length is not UNSET:
            field_dict["filenames_max_length"] = filenames_max_length
        if no_prompt_history is not UNSET:
            field_dict["no_prompt_history"] = no_prompt_history
        if forge_ref_a1111_home is not UNSET:
            field_dict["forge_ref_a1111_home"] = forge_ref_a1111_home
        if controlnet_dir is not UNSET:
            field_dict["controlnet_dir"] = controlnet_dir
        if controlnet_preprocessor_models_dir is not UNSET:
            field_dict["controlnet_preprocessor_models_dir"] = controlnet_preprocessor_models_dir
        if scunet_models_path is not UNSET:
            field_dict["scunet_models_path"] = scunet_models_path
        if swinir_models_path is not UNSET:
            field_dict["swinir_models_path"] = swinir_models_path
        if controlnet_loglevel is not UNSET:
            field_dict["controlnet_loglevel"] = controlnet_loglevel
        if controlnet_tracemalloc is not UNSET:
            field_dict["controlnet_tracemalloc"] = controlnet_tracemalloc
        if lora_dir is not UNSET:
            field_dict["lora_dir"] = lora_dir
        if lyco_dir_backcompat is not UNSET:
            field_dict["lyco_dir_backcompat"] = lyco_dir_backcompat

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        from ..models.flags_ngrok_options import FlagsNgrokOptions

        d = src_dict.copy()

        def _parse_gpu_device_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        gpu_device_id = _parse_gpu_device_id(d.pop("gpu_device_id", UNSET))

        all_in_fp32 = d.pop("all_in_fp32", UNSET)

        all_in_fp16 = d.pop("all_in_fp16", UNSET)

        unet_in_bf16 = d.pop("unet_in_bf16", UNSET)

        unet_in_fp16 = d.pop("unet_in_fp16", UNSET)

        unet_in_fp8_e4m3fn = d.pop("unet_in_fp8_e4m3fn", UNSET)

        unet_in_fp8_e5m2 = d.pop("unet_in_fp8_e5m2", UNSET)

        vae_in_fp16 = d.pop("vae_in_fp16", UNSET)

        vae_in_fp32 = d.pop("vae_in_fp32", UNSET)

        vae_in_bf16 = d.pop("vae_in_bf16", UNSET)

        vae_in_cpu = d.pop("vae_in_cpu", UNSET)

        clip_in_fp8_e4m3fn = d.pop("clip_in_fp8_e4m3fn", UNSET)

        clip_in_fp8_e5m2 = d.pop("clip_in_fp8_e5m2", UNSET)

        clip_in_fp16 = d.pop("clip_in_fp16", UNSET)

        clip_in_fp32 = d.pop("clip_in_fp32", UNSET)

        attention_split = d.pop("attention_split", UNSET)

        attention_quad = d.pop("attention_quad", UNSET)

        attention_pytorch = d.pop("attention_pytorch", UNSET)

        force_upcast_attention = d.pop("force_upcast_attention", UNSET)

        disable_attention_upcast = d.pop("disable_attention_upcast", UNSET)

        disable_xformers = d.pop("disable_xformers", UNSET)

        def _parse_directml(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        directml = _parse_directml(d.pop("directml", UNSET))

        disable_ipex_hijack = d.pop("disable_ipex_hijack", UNSET)

        always_gpu = d.pop("always_gpu", UNSET)

        always_high_vram = d.pop("always_high_vram", UNSET)

        always_normal_vram = d.pop("always_normal_vram", UNSET)

        always_low_vram = d.pop("always_low_vram", UNSET)

        always_no_vram = d.pop("always_no_vram", UNSET)

        always_cpu = d.pop("always_cpu", UNSET)

        always_offload_from_vram = d.pop("always_offload_from_vram", UNSET)

        pytorch_deterministic = d.pop("pytorch_deterministic", UNSET)

        cuda_malloc = d.pop("cuda_malloc", UNSET)

        cuda_stream = d.pop("cuda_stream", UNSET)

        pin_shared_memory = d.pop("pin_shared_memory", UNSET)

        disable_gpu_warning = d.pop("disable_gpu_warning", UNSET)

        f = d.pop("f", UNSET)

        update_all_extensions = d.pop("update_all_extensions", UNSET)

        skip_python_version_check = d.pop("skip_python_version_check", UNSET)

        skip_torch_cuda_test = d.pop("skip_torch_cuda_test", UNSET)

        reinstall_xformers = d.pop("reinstall_xformers", UNSET)

        reinstall_torch = d.pop("reinstall_torch", UNSET)

        update_check = d.pop("update_check", UNSET)

        test_server = d.pop("test_server", UNSET)

        log_startup = d.pop("log_startup", UNSET)

        skip_prepare_environment = d.pop("skip_prepare_environment", UNSET)

        skip_install = d.pop("skip_install", UNSET)

        dump_sysinfo = d.pop("dump_sysinfo", UNSET)

        def _parse_loglevel(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        loglevel = _parse_loglevel(d.pop("loglevel", UNSET))

        do_not_download_clip = d.pop("do_not_download_clip", UNSET)

        data_dir = d.pop("data_dir", UNSET)

        def _parse_models_dir(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        models_dir = _parse_models_dir(d.pop("models_dir", UNSET))

        config = d.pop("config", UNSET)

        ckpt = d.pop("ckpt", UNSET)

        def _parse_ckpt_dir(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        ckpt_dir = _parse_ckpt_dir(d.pop("ckpt_dir", UNSET))

        def _parse_vae_dir(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        vae_dir = _parse_vae_dir(d.pop("vae_dir", UNSET))

        def _parse_text_encoder_dir(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        text_encoder_dir = _parse_text_encoder_dir(d.pop("text_encoder_dir", UNSET))

        gfpgan_dir = d.pop("gfpgan_dir", UNSET)

        def _parse_gfpgan_model(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        gfpgan_model = _parse_gfpgan_model(d.pop("gfpgan_model", UNSET))

        no_half = d.pop("no_half", UNSET)

        no_half_vae = d.pop("no_half_vae", UNSET)

        no_progressbar_hiding = d.pop("no_progressbar_hiding", UNSET)

        max_batch_count = d.pop("max_batch_count", UNSET)

        embeddings_dir = d.pop("embeddings_dir", UNSET)

        textual_inversion_templates_dir = d.pop("textual_inversion_templates_dir", UNSET)

        hypernetwork_dir = d.pop("hypernetwork_dir", UNSET)

        localizations_dir = d.pop("localizations_dir", UNSET)

        allow_code = d.pop("allow_code", UNSET)

        medvram = d.pop("medvram", UNSET)

        medvram_sdxl = d.pop("medvram_sdxl", UNSET)

        lowvram = d.pop("lowvram", UNSET)

        lowram = d.pop("lowram", UNSET)

        always_batch_cond_uncond = d.pop("always_batch_cond_uncond", UNSET)

        unload_gfpgan = d.pop("unload_gfpgan", UNSET)

        precision = d.pop("precision", UNSET)

        upcast_sampling = d.pop("upcast_sampling", UNSET)

        share = d.pop("share", UNSET)

        def _parse_ngrok(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        ngrok = _parse_ngrok(d.pop("ngrok", UNSET))

        ngrok_region = d.pop("ngrok_region", UNSET)

        _ngrok_options = d.pop("ngrok_options", UNSET)
        ngrok_options: Union[Unset, FlagsNgrokOptions]
        if isinstance(_ngrok_options, Unset):
            ngrok_options = UNSET
        else:
            ngrok_options = FlagsNgrokOptions.from_dict(_ngrok_options)

        enable_insecure_extension_access = d.pop("enable_insecure_extension_access", UNSET)

        codeformer_models_path = d.pop("codeformer_models_path", UNSET)

        gfpgan_models_path = d.pop("gfpgan_models_path", UNSET)

        esrgan_models_path = d.pop("esrgan_models_path", UNSET)

        bsrgan_models_path = d.pop("bsrgan_models_path", UNSET)

        realesrgan_models_path = d.pop("realesrgan_models_path", UNSET)

        dat_models_path = d.pop("dat_models_path", UNSET)

        def _parse_clip_models_path(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        clip_models_path = _parse_clip_models_path(d.pop("clip_models_path", UNSET))

        xformers = d.pop("xformers", UNSET)

        force_enable_xformers = d.pop("force_enable_xformers", UNSET)

        xformers_flash_attention = d.pop("xformers_flash_attention", UNSET)

        deepdanbooru = d.pop("deepdanbooru", UNSET)

        opt_split_attention = d.pop("opt_split_attention", UNSET)

        opt_sub_quad_attention = d.pop("opt_sub_quad_attention", UNSET)

        sub_quad_q_chunk_size = d.pop("sub_quad_q_chunk_size", UNSET)

        def _parse_sub_quad_kv_chunk_size(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sub_quad_kv_chunk_size = _parse_sub_quad_kv_chunk_size(d.pop("sub_quad_kv_chunk_size", UNSET))

        def _parse_sub_quad_chunk_threshold(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sub_quad_chunk_threshold = _parse_sub_quad_chunk_threshold(d.pop("sub_quad_chunk_threshold", UNSET))

        opt_split_attention_invokeai = d.pop("opt_split_attention_invokeai", UNSET)

        opt_split_attention_v1 = d.pop("opt_split_attention_v1", UNSET)

        opt_sdp_attention = d.pop("opt_sdp_attention", UNSET)

        opt_sdp_no_mem_attention = d.pop("opt_sdp_no_mem_attention", UNSET)

        disable_opt_split_attention = d.pop("disable_opt_split_attention", UNSET)

        disable_nan_check = d.pop("disable_nan_check", UNSET)

        use_cpu = cast(List[Any], d.pop("use_cpu", UNSET))

        use_ipex = d.pop("use_ipex", UNSET)

        disable_model_loading_ram_optimization = d.pop("disable_model_loading_ram_optimization", UNSET)

        listen = d.pop("listen", UNSET)

        def _parse_port(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        port = _parse_port(d.pop("port", UNSET))

        show_negative_prompt = d.pop("show_negative_prompt", UNSET)

        ui_config_file = d.pop("ui_config_file", UNSET)

        hide_ui_dir_config = d.pop("hide_ui_dir_config", UNSET)

        freeze_settings = d.pop("freeze_settings", UNSET)

        def _parse_freeze_settings_in_sections(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        freeze_settings_in_sections = _parse_freeze_settings_in_sections(d.pop("freeze_settings_in_sections", UNSET))

        def _parse_freeze_specific_settings(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        freeze_specific_settings = _parse_freeze_specific_settings(d.pop("freeze_specific_settings", UNSET))

        ui_settings_file = d.pop("ui_settings_file", UNSET)

        gradio_debug = d.pop("gradio_debug", UNSET)

        def _parse_gradio_auth(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        gradio_auth = _parse_gradio_auth(d.pop("gradio_auth", UNSET))

        def _parse_gradio_auth_path(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        gradio_auth_path = _parse_gradio_auth_path(d.pop("gradio_auth_path", UNSET))

        def _parse_gradio_img2img_tool(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        gradio_img2img_tool = _parse_gradio_img2img_tool(d.pop("gradio_img2img_tool", UNSET))

        def _parse_gradio_inpaint_tool(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        gradio_inpaint_tool = _parse_gradio_inpaint_tool(d.pop("gradio_inpaint_tool", UNSET))

        gradio_allowed_path = cast(List[Any], d.pop("gradio_allowed_path", UNSET))

        opt_channelslast = d.pop("opt_channelslast", UNSET)

        styles_file = cast(List[Any], d.pop("styles_file", UNSET))

        autolaunch = d.pop("autolaunch", UNSET)

        def _parse_theme(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        theme = _parse_theme(d.pop("theme", UNSET))

        use_textbox_seed = d.pop("use_textbox_seed", UNSET)

        disable_console_progressbars = d.pop("disable_console_progressbars", UNSET)

        enable_console_prompts = d.pop("enable_console_prompts", UNSET)

        def _parse_vae_path(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        vae_path = _parse_vae_path(d.pop("vae_path", UNSET))

        disable_safe_unpickle = d.pop("disable_safe_unpickle", UNSET)

        api = d.pop("api", UNSET)

        def _parse_api_auth(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        api_auth = _parse_api_auth(d.pop("api_auth", UNSET))

        api_log = d.pop("api_log", UNSET)

        nowebui = d.pop("nowebui", UNSET)

        ui_debug_mode = d.pop("ui_debug_mode", UNSET)

        def _parse_device_id(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        device_id = _parse_device_id(d.pop("device_id", UNSET))

        administrator = d.pop("administrator", UNSET)

        def _parse_cors_allow_origins(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        cors_allow_origins = _parse_cors_allow_origins(d.pop("cors_allow_origins", UNSET))

        def _parse_cors_allow_origins_regex(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        cors_allow_origins_regex = _parse_cors_allow_origins_regex(d.pop("cors_allow_origins_regex", UNSET))

        def _parse_tls_keyfile(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        tls_keyfile = _parse_tls_keyfile(d.pop("tls_keyfile", UNSET))

        def _parse_tls_certfile(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        tls_certfile = _parse_tls_certfile(d.pop("tls_certfile", UNSET))

        def _parse_disable_tls_verify(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        disable_tls_verify = _parse_disable_tls_verify(d.pop("disable_tls_verify", UNSET))

        def _parse_server_name(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        server_name = _parse_server_name(d.pop("server_name", UNSET))

        gradio_queue = d.pop("gradio_queue", UNSET)

        no_gradio_queue = d.pop("no_gradio_queue", UNSET)

        skip_version_check = d.pop("skip_version_check", UNSET)

        no_hashing = d.pop("no_hashing", UNSET)

        no_download_sd_model = d.pop("no_download_sd_model", UNSET)

        def _parse_subpath(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        subpath = _parse_subpath(d.pop("subpath", UNSET))

        add_stop_route = d.pop("add_stop_route", UNSET)

        api_server_stop = d.pop("api_server_stop", UNSET)

        timeout_keep_alive = d.pop("timeout_keep_alive", UNSET)

        disable_all_extensions = d.pop("disable_all_extensions", UNSET)

        disable_extra_extensions = d.pop("disable_extra_extensions", UNSET)

        skip_load_model_at_start = d.pop("skip_load_model_at_start", UNSET)

        unix_filenames_sanitization = d.pop("unix_filenames_sanitization", UNSET)

        filenames_max_length = d.pop("filenames_max_length", UNSET)

        no_prompt_history = d.pop("no_prompt_history", UNSET)

        def _parse_forge_ref_a1111_home(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        forge_ref_a1111_home = _parse_forge_ref_a1111_home(d.pop("forge_ref_a1111_home", UNSET))

        def _parse_controlnet_dir(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        controlnet_dir = _parse_controlnet_dir(d.pop("controlnet_dir", UNSET))

        def _parse_controlnet_preprocessor_models_dir(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        controlnet_preprocessor_models_dir = _parse_controlnet_preprocessor_models_dir(
            d.pop("controlnet_preprocessor_models_dir", UNSET)
        )

        scunet_models_path = d.pop("scunet_models_path", UNSET)

        swinir_models_path = d.pop("swinir_models_path", UNSET)

        controlnet_loglevel = d.pop("controlnet_loglevel", UNSET)

        def _parse_controlnet_tracemalloc(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        controlnet_tracemalloc = _parse_controlnet_tracemalloc(d.pop("controlnet_tracemalloc", UNSET))

        lora_dir = d.pop("lora_dir", UNSET)

        lyco_dir_backcompat = d.pop("lyco_dir_backcompat", UNSET)

        flags = cls(
            gpu_device_id=gpu_device_id,
            all_in_fp32=all_in_fp32,
            all_in_fp16=all_in_fp16,
            unet_in_bf16=unet_in_bf16,
            unet_in_fp16=unet_in_fp16,
            unet_in_fp8_e4m3fn=unet_in_fp8_e4m3fn,
            unet_in_fp8_e5m2=unet_in_fp8_e5m2,
            vae_in_fp16=vae_in_fp16,
            vae_in_fp32=vae_in_fp32,
            vae_in_bf16=vae_in_bf16,
            vae_in_cpu=vae_in_cpu,
            clip_in_fp8_e4m3fn=clip_in_fp8_e4m3fn,
            clip_in_fp8_e5m2=clip_in_fp8_e5m2,
            clip_in_fp16=clip_in_fp16,
            clip_in_fp32=clip_in_fp32,
            attention_split=attention_split,
            attention_quad=attention_quad,
            attention_pytorch=attention_pytorch,
            force_upcast_attention=force_upcast_attention,
            disable_attention_upcast=disable_attention_upcast,
            disable_xformers=disable_xformers,
            directml=directml,
            disable_ipex_hijack=disable_ipex_hijack,
            always_gpu=always_gpu,
            always_high_vram=always_high_vram,
            always_normal_vram=always_normal_vram,
            always_low_vram=always_low_vram,
            always_no_vram=always_no_vram,
            always_cpu=always_cpu,
            always_offload_from_vram=always_offload_from_vram,
            pytorch_deterministic=pytorch_deterministic,
            cuda_malloc=cuda_malloc,
            cuda_stream=cuda_stream,
            pin_shared_memory=pin_shared_memory,
            disable_gpu_warning=disable_gpu_warning,
            f=f,
            update_all_extensions=update_all_extensions,
            skip_python_version_check=skip_python_version_check,
            skip_torch_cuda_test=skip_torch_cuda_test,
            reinstall_xformers=reinstall_xformers,
            reinstall_torch=reinstall_torch,
            update_check=update_check,
            test_server=test_server,
            log_startup=log_startup,
            skip_prepare_environment=skip_prepare_environment,
            skip_install=skip_install,
            dump_sysinfo=dump_sysinfo,
            loglevel=loglevel,
            do_not_download_clip=do_not_download_clip,
            data_dir=data_dir,
            models_dir=models_dir,
            config=config,
            ckpt=ckpt,
            ckpt_dir=ckpt_dir,
            vae_dir=vae_dir,
            text_encoder_dir=text_encoder_dir,
            gfpgan_dir=gfpgan_dir,
            gfpgan_model=gfpgan_model,
            no_half=no_half,
            no_half_vae=no_half_vae,
            no_progressbar_hiding=no_progressbar_hiding,
            max_batch_count=max_batch_count,
            embeddings_dir=embeddings_dir,
            textual_inversion_templates_dir=textual_inversion_templates_dir,
            hypernetwork_dir=hypernetwork_dir,
            localizations_dir=localizations_dir,
            allow_code=allow_code,
            medvram=medvram,
            medvram_sdxl=medvram_sdxl,
            lowvram=lowvram,
            lowram=lowram,
            always_batch_cond_uncond=always_batch_cond_uncond,
            unload_gfpgan=unload_gfpgan,
            precision=precision,
            upcast_sampling=upcast_sampling,
            share=share,
            ngrok=ngrok,
            ngrok_region=ngrok_region,
            ngrok_options=ngrok_options,
            enable_insecure_extension_access=enable_insecure_extension_access,
            codeformer_models_path=codeformer_models_path,
            gfpgan_models_path=gfpgan_models_path,
            esrgan_models_path=esrgan_models_path,
            bsrgan_models_path=bsrgan_models_path,
            realesrgan_models_path=realesrgan_models_path,
            dat_models_path=dat_models_path,
            clip_models_path=clip_models_path,
            xformers=xformers,
            force_enable_xformers=force_enable_xformers,
            xformers_flash_attention=xformers_flash_attention,
            deepdanbooru=deepdanbooru,
            opt_split_attention=opt_split_attention,
            opt_sub_quad_attention=opt_sub_quad_attention,
            sub_quad_q_chunk_size=sub_quad_q_chunk_size,
            sub_quad_kv_chunk_size=sub_quad_kv_chunk_size,
            sub_quad_chunk_threshold=sub_quad_chunk_threshold,
            opt_split_attention_invokeai=opt_split_attention_invokeai,
            opt_split_attention_v1=opt_split_attention_v1,
            opt_sdp_attention=opt_sdp_attention,
            opt_sdp_no_mem_attention=opt_sdp_no_mem_attention,
            disable_opt_split_attention=disable_opt_split_attention,
            disable_nan_check=disable_nan_check,
            use_cpu=use_cpu,
            use_ipex=use_ipex,
            disable_model_loading_ram_optimization=disable_model_loading_ram_optimization,
            listen=listen,
            port=port,
            show_negative_prompt=show_negative_prompt,
            ui_config_file=ui_config_file,
            hide_ui_dir_config=hide_ui_dir_config,
            freeze_settings=freeze_settings,
            freeze_settings_in_sections=freeze_settings_in_sections,
            freeze_specific_settings=freeze_specific_settings,
            ui_settings_file=ui_settings_file,
            gradio_debug=gradio_debug,
            gradio_auth=gradio_auth,
            gradio_auth_path=gradio_auth_path,
            gradio_img2img_tool=gradio_img2img_tool,
            gradio_inpaint_tool=gradio_inpaint_tool,
            gradio_allowed_path=gradio_allowed_path,
            opt_channelslast=opt_channelslast,
            styles_file=styles_file,
            autolaunch=autolaunch,
            theme=theme,
            use_textbox_seed=use_textbox_seed,
            disable_console_progressbars=disable_console_progressbars,
            enable_console_prompts=enable_console_prompts,
            vae_path=vae_path,
            disable_safe_unpickle=disable_safe_unpickle,
            api=api,
            api_auth=api_auth,
            api_log=api_log,
            nowebui=nowebui,
            ui_debug_mode=ui_debug_mode,
            device_id=device_id,
            administrator=administrator,
            cors_allow_origins=cors_allow_origins,
            cors_allow_origins_regex=cors_allow_origins_regex,
            tls_keyfile=tls_keyfile,
            tls_certfile=tls_certfile,
            disable_tls_verify=disable_tls_verify,
            server_name=server_name,
            gradio_queue=gradio_queue,
            no_gradio_queue=no_gradio_queue,
            skip_version_check=skip_version_check,
            no_hashing=no_hashing,
            no_download_sd_model=no_download_sd_model,
            subpath=subpath,
            add_stop_route=add_stop_route,
            api_server_stop=api_server_stop,
            timeout_keep_alive=timeout_keep_alive,
            disable_all_extensions=disable_all_extensions,
            disable_extra_extensions=disable_extra_extensions,
            skip_load_model_at_start=skip_load_model_at_start,
            unix_filenames_sanitization=unix_filenames_sanitization,
            filenames_max_length=filenames_max_length,
            no_prompt_history=no_prompt_history,
            forge_ref_a1111_home=forge_ref_a1111_home,
            controlnet_dir=controlnet_dir,
            controlnet_preprocessor_models_dir=controlnet_preprocessor_models_dir,
            scunet_models_path=scunet_models_path,
            swinir_models_path=swinir_models_path,
            controlnet_loglevel=controlnet_loglevel,
            controlnet_tracemalloc=controlnet_tracemalloc,
            lora_dir=lora_dir,
            lyco_dir_backcompat=lyco_dir_backcompat,
        )

        flags.additional_properties = d
        return flags

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
