from typing import Any, Dict, List, Type, TypeVar, Union, cast

from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

T = TypeVar("T", bound="Options")


@_attrs_define
class Options:
    r"""
    Attributes:
        samples_save (Union[Unset, bool]): Always save all generated images Default: True.
        samples_format (Union[Unset, str]): File format for images Default: 'png'.
        samples_filename_pattern (Union[Unset, Any]): Images filename pattern
        save_images_add_number (Union[Unset, bool]): Add number to filename when saving Default: True.
        save_images_replace_action (Union[Unset, str]): Saving the image to an existing file Default: 'Replace'.
        grid_save (Union[Unset, bool]): Always save all generated image grids Default: True.
        grid_format (Union[Unset, str]): File format for grids Default: 'png'.
        grid_extended_filename (Union[Unset, Any]): Add extended info (seed, prompt) to filename when saving grid
        grid_only_if_multiple (Union[Unset, bool]): Do not save grids consisting of one picture Default: True.
        grid_prevent_empty_spots (Union[Unset, Any]): Prevent empty spots in grid (when set to autodetect)
        grid_zip_filename_pattern (Union[Unset, Any]): Archive filename pattern
        n_rows (Union[Unset, float]): Grid row count; use -1 for autodetect and 0 for it to be same as batch size
            Default: -1.0.
        font (Union[Unset, Any]): Font for image grids that have text
        grid_text_active_color (Union[Unset, str]): Text color for image grids Default: '#000000'.
        grid_text_inactive_color (Union[Unset, str]): Inactive text color for image grids Default: '#999999'.
        grid_background_color (Union[Unset, str]): Background color for image grids Default: '#ffffff'.
        save_images_before_face_restoration (Union[Unset, Any]): Save a copy of image before doing face restoration.
        save_images_before_highres_fix (Union[Unset, Any]): Save a copy of image before applying highres fix.
        save_images_before_color_correction (Union[Unset, Any]): Save a copy of image before applying color correction
            to img2img results
        save_mask (Union[Unset, Any]): For inpainting, save a copy of the greyscale mask
        save_mask_composite (Union[Unset, Any]): For inpainting, save a masked composite
        jpeg_quality (Union[Unset, float]): Quality for saved jpeg images Default: 80.0.
        webp_lossless (Union[Unset, Any]): Use lossless compression for webp images
        export_for_4chan (Union[Unset, bool]): Save copy of large images as JPG Default: True.
        img_downscale_threshold (Union[Unset, float]): File size limit for the above option, MB Default: 4.0.
        target_side_length (Union[Unset, float]): Width/height limit for the above option, in pixels Default: 4000.0.
        img_max_size_mp (Union[Unset, float]): Maximum image size Default: 200.0.
        use_original_name_batch (Union[Unset, bool]): Use original name for output filename during batch process in
            extras tab Default: True.
        use_upscaler_name_as_suffix (Union[Unset, Any]): Use upscaler name as filename suffix in the extras tab
        save_selected_only (Union[Unset, bool]): When using 'Save' button, only save a single selected image Default:
            True.
        save_init_img (Union[Unset, Any]): Save init images when using img2img
        temp_dir (Union[Unset, Any]): Directory for temporary images; leave empty for default
        clean_temp_dir_at_start (Union[Unset, Any]): Cleanup non-default temporary directory when starting webui
        save_incomplete_images (Union[Unset, Any]): Save incomplete images
        notification_audio (Union[Unset, bool]): Play notification sound after image generation Default: True.
        notification_volume (Union[Unset, float]): Notification sound volume Default: 100.0.
        outdir_samples (Union[Unset, Any]): Output directory for images; if empty, defaults to three directories below
        outdir_txt2img_samples (Union[Unset, str]): Output directory for txt2img images Default: 'output/txt2img-
            images'.
        outdir_img2img_samples (Union[Unset, str]): Output directory for img2img images Default: 'output/img2img-
            images'.
        outdir_extras_samples (Union[Unset, str]): Output directory for images from extras tab Default: 'output/extras-
            images'.
        outdir_grids (Union[Unset, Any]): Output directory for grids; if empty, defaults to two directories below
        outdir_txt2img_grids (Union[Unset, str]): Output directory for txt2img grids Default: 'output/txt2img-grids'.
        outdir_img2img_grids (Union[Unset, str]): Output directory for img2img grids Default: 'output/img2img-grids'.
        outdir_save (Union[Unset, str]): Directory for saving images using the Save button Default: 'log/images'.
        outdir_init_images (Union[Unset, str]): Directory for saving init images when using img2img Default:
            'output/init-images'.
        save_to_dirs (Union[Unset, bool]): Save images to a subdirectory Default: True.
        grid_save_to_dirs (Union[Unset, bool]): Save grids to a subdirectory Default: True.
        use_save_to_dirs_for_ui (Union[Unset, Any]): When using "Save" button, save images to a subdirectory
        directories_filename_pattern (Union[Unset, str]): Directory name pattern Default: '[date]'.
        directories_max_prompt_words (Union[Unset, float]): Max prompt words for [prompt_words] pattern Default: 8.0.
        esrgan_tile (Union[Unset, float]): Tile size for ESRGAN upscalers. Default: 192.0.
        esrgan_tile_overlap (Union[Unset, float]): Tile overlap for ESRGAN upscalers. Default: 8.0.
        realesrgan_enabled_models (Union[Unset, List[Any]]): Select which Real-ESRGAN models to show in the web UI.
        dat_enabled_models (Union[Unset, List[Any]]): Select which DAT models to show in the web UI.
        dat_tile (Union[Unset, float]): Tile size for DAT upscalers. Default: 192.0.
        dat_tile_overlap (Union[Unset, float]): Tile overlap for DAT upscalers. Default: 8.0.
        upscaler_for_img2img (Union[Unset, Any]): Upscaler for img2img
        face_restoration (Union[Unset, Any]): Restore faces
        face_restoration_model (Union[Unset, str]): Face restoration model Default: 'CodeFormer'.
        code_former_weight (Union[Unset, float]): CodeFormer weight Default: 0.5.
        face_restoration_unload (Union[Unset, Any]): Move face restoration model from VRAM into RAM after processing
        auto_launch_browser (Union[Unset, str]): Automatically open webui in browser on startup Default: 'Local'.
        enable_console_prompts (Union[Unset, Any]): Print prompts to console when generating with txt2img and img2img.
        show_warnings (Union[Unset, Any]): Show warnings in console.
        show_gradio_deprecation_warnings (Union[Unset, bool]): Show gradio deprecation warnings in console. Default:
            True.
        memmon_poll_rate (Union[Unset, float]): VRAM usage polls per second during generation. Default: 8.0.
        samples_log_stdout (Union[Unset, Any]): Always print all generation info to standard output
        multiple_tqdm (Union[Unset, bool]): Add a second progress bar to the console that shows progress for an entire
            job. Default: True.
        enable_upscale_progressbar (Union[Unset, bool]): Show a progress bar in the console for tiled upscaling.
            Default: True.
        print_hypernet_extra (Union[Unset, Any]): Print extra hypernetwork information to console.
        list_hidden_files (Union[Unset, bool]): Load models/files in hidden directories Default: True.
        disable_mmap_load_safetensors (Union[Unset, Any]): Disable memmapping for loading .safetensors files.
        hide_ldm_prints (Union[Unset, bool]): Prevent Stability-AI's ldm/sgm modules from printing noise to console.
            Default: True.
        dump_stacks_on_signal (Union[Unset, Any]): Print stack traces before exiting the program with ctrl+c.
        api_enable_requests (Union[Unset, bool]): Allow http:// and https:// URLs for input images in API Default: True.
        api_forbid_local_requests (Union[Unset, bool]): Forbid URLs to local resources Default: True.
        api_useragent (Union[Unset, Any]): User agent for requests
        unload_models_when_training (Union[Unset, Any]): Move VAE and CLIP to RAM when training if possible. Saves VRAM.
        pin_memory (Union[Unset, Any]): Turn on pin_memory for DataLoader. Makes training slightly faster but can
            increase memory usage.
        save_optimizer_state (Union[Unset, Any]): Saves Optimizer state as separate *.optim file. Training of embedding
            or HN can be resumed with the matching optim file.
        save_training_settings_to_txt (Union[Unset, bool]): Save textual inversion and hypernet settings to a text file
            whenever training starts. Default: True.
        dataset_filename_word_regex (Union[Unset, Any]): Filename word regex
        dataset_filename_join_string (Union[Unset, str]): Filename join string Default: ' '.
        training_image_repeats_per_epoch (Union[Unset, float]): Number of repeats for a single input image per epoch;
            used only for displaying epoch number Default: 1.0.
        training_write_csv_every (Union[Unset, float]): Save an csv containing the loss to log directory every N steps,
            0 to disable Default: 500.0.
        training_xattention_optimizations (Union[Unset, Any]): Use cross attention optimizations while training
        training_enable_tensorboard (Union[Unset, Any]): Enable tensorboard logging.
        training_tensorboard_save_images (Union[Unset, Any]): Save generated images within tensorboard.
        training_tensorboard_flush_every (Union[Unset, float]): How often, in seconds, to flush the pending tensorboard
            events and summaries to disk. Default: 120.0.
        sd_model_checkpoint (Union[Unset, Any]): Stable Diffusion checkpoint
        sd_checkpoints_limit (Union[Unset, float]): Maximum number of checkpoints loaded at the same time Default: 1.0.
        sd_checkpoints_keep_in_cpu (Union[Unset, bool]): Only keep one model on device Default: True.
        sd_checkpoint_cache (Union[Unset, Any]): Checkpoints to cache in RAM
        sd_unet (Union[Unset, str]): SD Unet Default: 'Automatic'.
        enable_quantization (Union[Unset, Any]): Enable quantization in K samplers for sharper and cleaner results. This
            may change existing seeds
        emphasis (Union[Unset, str]): Emphasis mode Default: 'Original'.
        enable_batch_seeds (Union[Unset, bool]): Make K-diffusion samplers produce same images in a batch as when making
            a single image Default: True.
        comma_padding_backtrack (Union[Unset, float]): Prompt word wrap length limit Default: 20.0.
        clip_stop_at_last_layers (Union[Unset, float]): Clip skip Default: 1.0.
        upcast_attn (Union[Unset, Any]): Upcast cross attention layer to float32
        randn_source (Union[Unset, str]): Random number generator source. Default: 'GPU'.
        tiling (Union[Unset, Any]): Tiling
        hires_fix_refiner_pass (Union[Unset, str]): Hires fix: which pass to enable refiner for Default: 'second pass'.
        sdxl_crop_top (Union[Unset, Any]): crop top coordinate
        sdxl_crop_left (Union[Unset, Any]): crop left coordinate
        sdxl_refiner_low_aesthetic_score (Union[Unset, float]): SDXL low aesthetic score Default: 2.5.
        sdxl_refiner_high_aesthetic_score (Union[Unset, float]): SDXL high aesthetic score Default: 6.0.
        sd_vae_explanation (Union[Unset, str]):  Default: "<abbr title='Variational autoencoder'>VAE</abbr> is a neural
            network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>\nimage into latent space
            representation and back. Latent space representation is what stable diffusion is working on during
            sampling\n(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting
            image after the sampling is finished.\nFor img2img, VAE is used to process user's input image before the
            sampling, and to create an image after sampling.".
        sd_vae_checkpoint_cache (Union[Unset, Any]): VAE Checkpoints to cache in RAM
        sd_vae (Union[Unset, str]): SD VAE Default: 'Automatic'.
        sd_vae_overrides_per_model_preferences (Union[Unset, bool]): Selected VAE overrides per-model preferences
            Default: True.
        auto_vae_precision_bfloat16 (Union[Unset, Any]): Automatically convert VAE to bfloat16
        auto_vae_precision (Union[Unset, bool]): Automatically revert VAE to 32-bit floats Default: True.
        sd_vae_encode_method (Union[Unset, str]): VAE type for encode Default: 'Full'.
        sd_vae_decode_method (Union[Unset, str]): VAE type for decode Default: 'Full'.
        inpainting_mask_weight (Union[Unset, float]): Inpainting conditioning mask strength Default: 1.0.
        initial_noise_multiplier (Union[Unset, float]): Noise multiplier for img2img Default: 1.0.
        img2img_extra_noise (Union[Unset, Any]): Extra noise multiplier for img2img and hires fix
        img2img_color_correction (Union[Unset, Any]): Apply color correction to img2img results to match original
            colors.
        img2img_fix_steps (Union[Unset, Any]): With img2img, do exactly the amount of steps the slider specifies.
        img2img_background_color (Union[Unset, str]): With img2img, fill transparent parts of the input image with this
            color. Default: '#ffffff'.
        img2img_editor_height (Union[Unset, float]): Height of the image editor Default: 720.0.
        img2img_sketch_default_brush_color (Union[Unset, str]): Sketch initial brush color Default: '#ffffff'.
        img2img_inpaint_mask_brush_color (Union[Unset, str]): Inpaint mask brush color Default: '#ffffff'.
        img2img_inpaint_sketch_default_brush_color (Union[Unset, str]): Inpaint sketch initial brush color Default:
            '#ffffff'.
        return_mask (Union[Unset, Any]): For inpainting, include the greyscale mask in results for web
        return_mask_composite (Union[Unset, Any]): For inpainting, include masked composite in results for web
        img2img_batch_show_results_limit (Union[Unset, float]): Show the first N batch img2img results in UI Default:
            32.0.
        overlay_inpaint (Union[Unset, bool]): Overlay original for inpaint Default: True.
        cross_attention_optimization (Union[Unset, str]): Cross attention optimization Default: 'Automatic'.
        s_min_uncond (Union[Unset, Any]): Negative Guidance minimum sigma
        token_merging_ratio (Union[Unset, Any]): Token merging ratio
        token_merging_ratio_img2img (Union[Unset, Any]): Token merging ratio for img2img
        token_merging_ratio_hr (Union[Unset, Any]): Token merging ratio for high-res pass
        pad_cond_uncond (Union[Unset, Any]): Pad prompt/negative prompt
        pad_cond_uncond_v0 (Union[Unset, Any]): Pad prompt/negative prompt (v0)
        persistent_cond_cache (Union[Unset, bool]): Persistent cond cache Default: True.
        batch_cond_uncond (Union[Unset, bool]): Batch cond/uncond Default: True.
        fp8_storage (Union[Unset, str]): FP8 weight Default: 'Disable'.
        cache_fp16_weight (Union[Unset, Any]): Cache FP16 weight for LoRA
        auto_backcompat (Union[Unset, bool]): Automatic backward compatibility Default: True.
        use_old_emphasis_implementation (Union[Unset, Any]): Use old emphasis implementation. Can be useful to reproduce
            old seeds.
        use_old_karras_scheduler_sigmas (Union[Unset, Any]): Use old karras scheduler sigmas (0.1 to 10).
        no_dpmpp_sde_batch_determinism (Union[Unset, Any]): Do not make DPM++ SDE deterministic across different batch
            sizes.
        use_old_hires_fix_width_height (Union[Unset, Any]): For hires fix, use width/height sliders to set final
            resolution rather than first pass (disables Upscale by, Resize width/height to).
        dont_fix_second_order_samplers_schedule (Union[Unset, Any]): Do not fix prompt schedule for second order
            samplers.
        hires_fix_use_firstpass_conds (Union[Unset, Any]): For hires fix, calculate conds of second pass using extra
            networks of first pass.
        use_old_scheduling (Union[Unset, Any]): Use old prompt editing timelines.
        use_downcasted_alpha_bar (Union[Unset, Any]): Downcast model alphas_cumprod to fp16 before sampling. For
            reproducing old seeds.
        interrogate_keep_models_in_memory (Union[Unset, Any]): Keep models in VRAM
        interrogate_return_ranks (Union[Unset, Any]): Include ranks of model tags matches in results.
        interrogate_clip_num_beams (Union[Unset, float]): BLIP: num_beams Default: 1.0.
        interrogate_clip_min_length (Union[Unset, float]): BLIP: minimum description length Default: 24.0.
        interrogate_clip_max_length (Union[Unset, float]): BLIP: maximum description length Default: 48.0.
        interrogate_clip_dict_limit (Union[Unset, float]): CLIP: maximum number of lines in text file Default: 1500.0.
        interrogate_clip_skip_categories (Union[Unset, Any]): CLIP: skip inquire categories
        interrogate_deepbooru_score_threshold (Union[Unset, float]): deepbooru: score threshold Default: 0.5.
        deepbooru_sort_alpha (Union[Unset, bool]): deepbooru: sort tags alphabetically Default: True.
        deepbooru_use_spaces (Union[Unset, bool]): deepbooru: use spaces in tags Default: True.
        deepbooru_escape (Union[Unset, bool]): deepbooru: escape (\) brackets Default: True.
        deepbooru_filter_tags (Union[Unset, Any]): deepbooru: filter out those tags
        extra_networks_show_hidden_directories (Union[Unset, bool]): Show hidden directories Default: True.
        extra_networks_dir_button_function (Union[Unset, Any]): Add a '/' to the beginning of directory buttons
        extra_networks_hidden_models (Union[Unset, str]): Show cards for models in hidden directories Default: 'When
            searched'.
        extra_networks_default_multiplier (Union[Unset, float]): Default multiplier for extra networks Default: 1.0.
        extra_networks_card_width (Union[Unset, Any]): Card width for Extra Networks
        extra_networks_card_height (Union[Unset, Any]): Card height for Extra Networks
        extra_networks_card_text_scale (Union[Unset, float]): Card text scale Default: 1.0.
        extra_networks_card_show_desc (Union[Unset, bool]): Show description on card Default: True.
        extra_networks_card_description_is_html (Union[Unset, Any]): Treat card description as HTML
        extra_networks_card_order_field (Union[Unset, str]): Default order field for Extra Networks cards Default:
            'Path'.
        extra_networks_card_order (Union[Unset, str]): Default order for Extra Networks cards Default: 'Ascending'.
        extra_networks_tree_view_default_enabled (Union[Unset, Any]): Enables the Extra Networks directory tree view by
            default
        extra_networks_add_text_separator (Union[Unset, str]): Extra networks separator Default: ' '.
        ui_extra_networks_tab_reorder (Union[Unset, Any]): Extra networks tab order
        textual_inversion_print_at_load (Union[Unset, Any]): Print a list of Textual Inversion embeddings when loading
            model
        textual_inversion_add_hashes_to_infotext (Union[Unset, bool]): Add Textual Inversion hashes to infotext Default:
            True.
        sd_hypernetwork (Union[Unset, str]): Add hypernetwork to prompt Default: 'None'.
        keyedit_precision_attention (Union[Unset, float]): Precision for (attention:1.1) when editing the prompt with
            Ctrl+up/down Default: 0.1.
        keyedit_precision_extra (Union[Unset, float]): Precision for <extra networks:0.9> when editing the prompt with
            Ctrl+up/down Default: 0.05.
        keyedit_delimiters (Union[Unset, str]): Word delimiters when editing the prompt with Ctrl+up/down Default:
            '.,\\/!?%^*;:{}=`~() '.
        keyedit_delimiters_whitespace (Union[Unset, List[Any]]): Ctrl+up/down whitespace delimiters
        keyedit_move (Union[Unset, bool]): Alt+left/right moves prompt elements Default: True.
        disable_token_counters (Union[Unset, Any]): Disable prompt token counters
        include_styles_into_token_counters (Union[Unset, bool]): Count tokens of enabled styles Default: True.
        return_grid (Union[Unset, bool]): Show grid in gallery Default: True.
        do_not_show_images (Union[Unset, Any]): Do not show any images in gallery
        js_modal_lightbox (Union[Unset, bool]): Full page image viewer: enable Default: True.
        js_modal_lightbox_initially_zoomed (Union[Unset, bool]): Full page image viewer: show images zoomed in by
            default Default: True.
        js_modal_lightbox_gamepad (Union[Unset, Any]): Full page image viewer: navigate with gamepad
        js_modal_lightbox_gamepad_repeat (Union[Unset, float]): Full page image viewer: gamepad repeat period Default:
            250.0.
        sd_webui_modal_lightbox_icon_opacity (Union[Unset, float]): Full page image viewer: control icon unfocused
            opacity Default: 1.0.
        sd_webui_modal_lightbox_toolbar_opacity (Union[Unset, float]): Full page image viewer: tool bar opacity Default:
            0.9.
        gallery_height (Union[Unset, Any]): Gallery height
        open_dir_button_choice (Union[Unset, str]): What directory the [📂] button opens Default: 'Subdirectory'.
        compact_prompt_box (Union[Unset, Any]): Compact prompt layout
        samplers_in_dropdown (Union[Unset, bool]): Use dropdown for sampler selection instead of radio group Default:
            True.
        dimensions_and_batch_together (Union[Unset, bool]): Show Width/Height and Batch sliders in same row Default:
            True.
        sd_checkpoint_dropdown_use_short (Union[Unset, Any]): Checkpoint dropdown: use filenames without paths
        hires_fix_show_sampler (Union[Unset, Any]): Hires fix: show hires checkpoint and sampler selection
        hires_fix_show_prompts (Union[Unset, Any]): Hires fix: show hires prompt and negative prompt
        txt2img_settings_accordion (Union[Unset, Any]): Settings in txt2img hidden under Accordion
        img2img_settings_accordion (Union[Unset, Any]): Settings in img2img hidden under Accordion
        interrupt_after_current (Union[Unset, bool]): Don't Interrupt in the middle Default: True.
        localization (Union[Unset, str]): Localization Default: 'None'.
        quicksettings_list (Union[Unset, List[Any]]): Quicksettings list
        ui_tab_order (Union[Unset, Any]): UI tab order
        hidden_tabs (Union[Unset, Any]): Hidden UI tabs
        ui_reorder_list (Union[Unset, Any]): UI item order for txt2img/img2img tabs
        gradio_theme (Union[Unset, str]): Gradio theme Default: 'Default'.
        gradio_themes_cache (Union[Unset, bool]): Cache gradio themes locally Default: True.
        show_progress_in_title (Union[Unset, bool]): Show generation progress in window title. Default: True.
        send_seed (Union[Unset, bool]): Send seed when sending prompt or image to other interface Default: True.
        send_size (Union[Unset, bool]): Send size when sending prompt or image to another interface Default: True.
        infotext_explanation (Union[Unset, str]):  Default: 'Infotext is what this software calls the text that contains
            generation parameters and can be used to generate the same picture again.\nIt is displayed in UI below the
            image. To use infotext, paste it into the prompt and click the ↙️ paste button.'.
        enable_pnginfo (Union[Unset, bool]): Write infotext to metadata of the generated image Default: True.
        save_txt (Union[Unset, Any]): Create a text file with infotext next to every generated image
        add_model_name_to_info (Union[Unset, bool]): Add model name to infotext Default: True.
        add_model_hash_to_info (Union[Unset, bool]): Add model hash to infotext Default: True.
        add_vae_name_to_info (Union[Unset, bool]): Add VAE name to infotext Default: True.
        add_vae_hash_to_info (Union[Unset, bool]): Add VAE hash to infotext Default: True.
        add_user_name_to_info (Union[Unset, Any]): Add user name to infotext when authenticated
        add_version_to_infotext (Union[Unset, bool]): Add program version to infotext Default: True.
        disable_weights_auto_swap (Union[Unset, bool]): Disregard checkpoint information from pasted infotext Default:
            True.
        infotext_skip_pasting (Union[Unset, Any]): Disregard fields from pasted infotext
        infotext_styles (Union[Unset, str]): Infer styles from prompts of pasted infotext Default: 'Apply if any'.
        show_progressbar (Union[Unset, bool]): Show progressbar Default: True.
        live_previews_enable (Union[Unset, bool]): Show live previews of the created image Default: True.
        live_previews_image_format (Union[Unset, str]): Live preview file format Default: 'png'.
        show_progress_grid (Union[Unset, bool]): Show previews of all images generated in a batch as a grid Default:
            True.
        show_progress_every_n_steps (Union[Unset, float]): Live preview display period Default: 10.0.
        show_progress_type (Union[Unset, str]): Live preview method Default: 'Approx NN'.
        live_preview_allow_lowvram_full (Union[Unset, Any]): Allow Full live preview method with lowvram/medvram
        live_preview_content (Union[Unset, str]): Live preview subject Default: 'Prompt'.
        live_preview_refresh_period (Union[Unset, float]): Progressbar and preview update period Default: 1000.0.
        live_preview_fast_interrupt (Union[Unset, Any]): Return image with chosen live preview method on interrupt
        js_live_preview_in_modal_lightbox (Union[Unset, Any]): Show Live preview in full page image viewer
        hide_samplers (Union[Unset, Any]): Hide samplers in user interface
        eta_ddim (Union[Unset, Any]): Eta for DDIM
        eta_ancestral (Union[Unset, float]): Eta for k-diffusion samplers Default: 1.0.
        ddim_discretize (Union[Unset, str]): img2img DDIM discretize Default: 'uniform'.
        s_churn (Union[Unset, Any]): sigma churn
        s_tmin (Union[Unset, Any]): sigma tmin
        s_tmax (Union[Unset, Any]): sigma tmax
        s_noise (Union[Unset, float]): sigma noise Default: 1.0.
        k_sched_type (Union[Unset, str]): Scheduler type Default: 'Automatic'.
        sigma_min (Union[Unset, Any]): sigma min
        sigma_max (Union[Unset, Any]): sigma max
        rho (Union[Unset, Any]): rho
        eta_noise_seed_delta (Union[Unset, Any]): Eta noise seed delta
        always_discard_next_to_last_sigma (Union[Unset, Any]): Always discard next-to-last sigma
        sgm_noise_multiplier (Union[Unset, Any]): SGM noise multiplier
        uni_pc_variant (Union[Unset, str]): UniPC variant Default: 'bh1'.
        uni_pc_skip_type (Union[Unset, str]): UniPC skip type Default: 'time_uniform'.
        uni_pc_order (Union[Unset, float]): UniPC order Default: 3.0.
        uni_pc_lower_order_final (Union[Unset, bool]): UniPC lower order final Default: True.
        sd_noise_schedule (Union[Unset, str]): Noise schedule for sampling Default: 'Default'.
        postprocessing_enable_in_main_ui (Union[Unset, Any]): Enable postprocessing operations in txt2img and img2img
            tabs
        postprocessing_operation_order (Union[Unset, Any]): Postprocessing operation order
        upscaling_max_images_in_cache (Union[Unset, float]): Maximum number of images in upscaling cache Default: 5.0.
        postprocessing_existing_caption_action (Union[Unset, str]): Action for existing captions Default: 'Ignore'.
        disabled_extensions (Union[Unset, Any]): Disable these extensions
        disable_all_extensions (Union[Unset, str]): Disable all extensions (preserves the list of disabled extensions)
            Default: 'none'.
        restore_config_state_file (Union[Unset, Any]): Config state file to restore from, under 'config-states/' folder
        sd_checkpoint_hash (Union[Unset, Any]): SHA256 hash of the current checkpoint
        sd_lora (Union[Unset, str]): Add network to prompt Default: 'None'.
        lora_preferred_name (Union[Unset, str]): When adding to prompt, refer to Lora by Default: 'Alias from file'.
        lora_add_hashes_to_infotext (Union[Unset, bool]): Add Lora hashes to infotext Default: True.
        lora_show_all (Union[Unset, Any]): Always show all networks on the Lora page
        lora_hide_unknown_for_versions (Union[Unset, Any]): Hide networks of unknown versions for model versions
        lora_in_memory_limit (Union[Unset, Any]): Number of Lora networks to keep cached in memory
        lora_not_found_warning_console (Union[Unset, Any]): Lora not found warning in console
        lora_not_found_gradio_warning (Union[Unset, Any]): Lora not found warning popup in webui
        lora_functional (Union[Unset, Any]): Lora/Networks: use old method that takes longer when you have multiple
            Loras active and produces same results as kohya-ss/sd-webui-additional-networks extension
        canvas_hotkey_zoom (Union[Unset, str]): Zoom canvas Default: 'Alt'.
        canvas_hotkey_adjust (Union[Unset, str]): Adjust brush size Default: 'Ctrl'.
        canvas_hotkey_shrink_brush (Union[Unset, str]): Shrink the brush size Default: 'Q'.
        canvas_hotkey_grow_brush (Union[Unset, str]): Enlarge the brush size Default: 'W'.
        canvas_hotkey_move (Union[Unset, str]): Moving the canvas Default: 'F'.
        canvas_hotkey_fullscreen (Union[Unset, str]): Fullscreen Mode, maximizes the picture so that it fits into the
            screen and stretches it to its full width  Default: 'S'.
        canvas_hotkey_reset (Union[Unset, str]): Reset zoom and canvas positon Default: 'R'.
        canvas_hotkey_overlap (Union[Unset, str]): Toggle overlap Default: 'O'.
        canvas_show_tooltip (Union[Unset, bool]): Enable tooltip on the canvas Default: True.
        canvas_auto_expand (Union[Unset, bool]): Automatically expands an image that does not fit completely in the
            canvas area, similar to manually pressing the S and R buttons Default: True.
        canvas_blur_prompt (Union[Unset, Any]): Take the focus off the prompt when working with a canvas
        canvas_disabled_functions (Union[Unset, List[Any]]): Disable function that you don't use
        settings_in_ui (Union[Unset, str]):  Default: 'This page allows you to add some settings to the main interface
            of txt2img and img2img tabs.'.
        extra_options_txt2img (Union[Unset, Any]): Settings for txt2img
        extra_options_img2img (Union[Unset, Any]): Settings for img2img
        extra_options_cols (Union[Unset, float]): Number of columns for added settings Default: 1.0.
        extra_options_accordion (Union[Unset, Any]): Place added settings into an accordion
    """

    samples_save: Union[Unset, bool] = True
    samples_format: Union[Unset, str] = "png"
    samples_filename_pattern: Union[Unset, Any] = UNSET
    save_images_add_number: Union[Unset, bool] = True
    save_images_replace_action: Union[Unset, str] = "Replace"
    grid_save: Union[Unset, bool] = True
    grid_format: Union[Unset, str] = "png"
    grid_extended_filename: Union[Unset, Any] = UNSET
    grid_only_if_multiple: Union[Unset, bool] = True
    grid_prevent_empty_spots: Union[Unset, Any] = UNSET
    grid_zip_filename_pattern: Union[Unset, Any] = UNSET
    n_rows: Union[Unset, float] = -1.0
    font: Union[Unset, Any] = UNSET
    grid_text_active_color: Union[Unset, str] = "#000000"
    grid_text_inactive_color: Union[Unset, str] = "#999999"
    grid_background_color: Union[Unset, str] = "#ffffff"
    save_images_before_face_restoration: Union[Unset, Any] = UNSET
    save_images_before_highres_fix: Union[Unset, Any] = UNSET
    save_images_before_color_correction: Union[Unset, Any] = UNSET
    save_mask: Union[Unset, Any] = UNSET
    save_mask_composite: Union[Unset, Any] = UNSET
    jpeg_quality: Union[Unset, float] = 80.0
    webp_lossless: Union[Unset, Any] = UNSET
    export_for_4chan: Union[Unset, bool] = True
    img_downscale_threshold: Union[Unset, float] = 4.0
    target_side_length: Union[Unset, float] = 4000.0
    img_max_size_mp: Union[Unset, float] = 200.0
    use_original_name_batch: Union[Unset, bool] = True
    use_upscaler_name_as_suffix: Union[Unset, Any] = UNSET
    save_selected_only: Union[Unset, bool] = True
    save_init_img: Union[Unset, Any] = UNSET
    temp_dir: Union[Unset, Any] = UNSET
    clean_temp_dir_at_start: Union[Unset, Any] = UNSET
    save_incomplete_images: Union[Unset, Any] = UNSET
    notification_audio: Union[Unset, bool] = True
    notification_volume: Union[Unset, float] = 100.0
    outdir_samples: Union[Unset, Any] = UNSET
    outdir_txt2img_samples: Union[Unset, str] = "output/txt2img-images"
    outdir_img2img_samples: Union[Unset, str] = "output/img2img-images"
    outdir_extras_samples: Union[Unset, str] = "output/extras-images"
    outdir_grids: Union[Unset, Any] = UNSET
    outdir_txt2img_grids: Union[Unset, str] = "output/txt2img-grids"
    outdir_img2img_grids: Union[Unset, str] = "output/img2img-grids"
    outdir_save: Union[Unset, str] = "log/images"
    outdir_init_images: Union[Unset, str] = "output/init-images"
    save_to_dirs: Union[Unset, bool] = True
    grid_save_to_dirs: Union[Unset, bool] = True
    use_save_to_dirs_for_ui: Union[Unset, Any] = UNSET
    directories_filename_pattern: Union[Unset, str] = "[date]"
    directories_max_prompt_words: Union[Unset, float] = 8.0
    esrgan_tile: Union[Unset, float] = 192.0
    esrgan_tile_overlap: Union[Unset, float] = 8.0
    realesrgan_enabled_models: Union[Unset, List[Any]] = UNSET
    dat_enabled_models: Union[Unset, List[Any]] = UNSET
    dat_tile: Union[Unset, float] = 192.0
    dat_tile_overlap: Union[Unset, float] = 8.0
    upscaler_for_img2img: Union[Unset, Any] = UNSET
    face_restoration: Union[Unset, Any] = UNSET
    face_restoration_model: Union[Unset, str] = "CodeFormer"
    code_former_weight: Union[Unset, float] = 0.5
    face_restoration_unload: Union[Unset, Any] = UNSET
    auto_launch_browser: Union[Unset, str] = "Local"
    enable_console_prompts: Union[Unset, Any] = UNSET
    show_warnings: Union[Unset, Any] = UNSET
    show_gradio_deprecation_warnings: Union[Unset, bool] = True
    memmon_poll_rate: Union[Unset, float] = 8.0
    samples_log_stdout: Union[Unset, Any] = UNSET
    multiple_tqdm: Union[Unset, bool] = True
    enable_upscale_progressbar: Union[Unset, bool] = True
    print_hypernet_extra: Union[Unset, Any] = UNSET
    list_hidden_files: Union[Unset, bool] = True
    disable_mmap_load_safetensors: Union[Unset, Any] = UNSET
    hide_ldm_prints: Union[Unset, bool] = True
    dump_stacks_on_signal: Union[Unset, Any] = UNSET
    api_enable_requests: Union[Unset, bool] = True
    api_forbid_local_requests: Union[Unset, bool] = True
    api_useragent: Union[Unset, Any] = UNSET
    unload_models_when_training: Union[Unset, Any] = UNSET
    pin_memory: Union[Unset, Any] = UNSET
    save_optimizer_state: Union[Unset, Any] = UNSET
    save_training_settings_to_txt: Union[Unset, bool] = True
    dataset_filename_word_regex: Union[Unset, Any] = UNSET
    dataset_filename_join_string: Union[Unset, str] = " "
    training_image_repeats_per_epoch: Union[Unset, float] = 1.0
    training_write_csv_every: Union[Unset, float] = 500.0
    training_xattention_optimizations: Union[Unset, Any] = UNSET
    training_enable_tensorboard: Union[Unset, Any] = UNSET
    training_tensorboard_save_images: Union[Unset, Any] = UNSET
    training_tensorboard_flush_every: Union[Unset, float] = 120.0
    sd_model_checkpoint: Union[Unset, Any] = UNSET
    sd_checkpoints_limit: Union[Unset, float] = 1.0
    sd_checkpoints_keep_in_cpu: Union[Unset, bool] = True
    sd_checkpoint_cache: Union[Unset, Any] = UNSET
    sd_unet: Union[Unset, str] = "Automatic"
    enable_quantization: Union[Unset, Any] = UNSET
    emphasis: Union[Unset, str] = "Original"
    enable_batch_seeds: Union[Unset, bool] = True
    comma_padding_backtrack: Union[Unset, float] = 20.0
    clip_stop_at_last_layers: Union[Unset, float] = 1.0
    upcast_attn: Union[Unset, Any] = UNSET
    randn_source: Union[Unset, str] = "GPU"
    tiling: Union[Unset, Any] = UNSET
    hires_fix_refiner_pass: Union[Unset, str] = "second pass"
    sdxl_crop_top: Union[Unset, Any] = UNSET
    sdxl_crop_left: Union[Unset, Any] = UNSET
    sdxl_refiner_low_aesthetic_score: Union[Unset, float] = 2.5
    sdxl_refiner_high_aesthetic_score: Union[Unset, float] = 6.0
    sd_vae_explanation: Union[Unset, str] = (
        "<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>\nimage into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling\n(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.\nFor img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling."
    )
    sd_vae_checkpoint_cache: Union[Unset, Any] = UNSET
    sd_vae: Union[Unset, str] = "Automatic"
    sd_vae_overrides_per_model_preferences: Union[Unset, bool] = True
    auto_vae_precision_bfloat16: Union[Unset, Any] = UNSET
    auto_vae_precision: Union[Unset, bool] = True
    sd_vae_encode_method: Union[Unset, str] = "Full"
    sd_vae_decode_method: Union[Unset, str] = "Full"
    inpainting_mask_weight: Union[Unset, float] = 1.0
    initial_noise_multiplier: Union[Unset, float] = 1.0
    img2img_extra_noise: Union[Unset, Any] = UNSET
    img2img_color_correction: Union[Unset, Any] = UNSET
    img2img_fix_steps: Union[Unset, Any] = UNSET
    img2img_background_color: Union[Unset, str] = "#ffffff"
    img2img_editor_height: Union[Unset, float] = 720.0
    img2img_sketch_default_brush_color: Union[Unset, str] = "#ffffff"
    img2img_inpaint_mask_brush_color: Union[Unset, str] = "#ffffff"
    img2img_inpaint_sketch_default_brush_color: Union[Unset, str] = "#ffffff"
    return_mask: Union[Unset, Any] = UNSET
    return_mask_composite: Union[Unset, Any] = UNSET
    img2img_batch_show_results_limit: Union[Unset, float] = 32.0
    overlay_inpaint: Union[Unset, bool] = True
    cross_attention_optimization: Union[Unset, str] = "Automatic"
    s_min_uncond: Union[Unset, Any] = UNSET
    token_merging_ratio: Union[Unset, Any] = UNSET
    token_merging_ratio_img2img: Union[Unset, Any] = UNSET
    token_merging_ratio_hr: Union[Unset, Any] = UNSET
    pad_cond_uncond: Union[Unset, Any] = UNSET
    pad_cond_uncond_v0: Union[Unset, Any] = UNSET
    persistent_cond_cache: Union[Unset, bool] = True
    batch_cond_uncond: Union[Unset, bool] = True
    fp8_storage: Union[Unset, str] = "Disable"
    cache_fp16_weight: Union[Unset, Any] = UNSET
    auto_backcompat: Union[Unset, bool] = True
    use_old_emphasis_implementation: Union[Unset, Any] = UNSET
    use_old_karras_scheduler_sigmas: Union[Unset, Any] = UNSET
    no_dpmpp_sde_batch_determinism: Union[Unset, Any] = UNSET
    use_old_hires_fix_width_height: Union[Unset, Any] = UNSET
    dont_fix_second_order_samplers_schedule: Union[Unset, Any] = UNSET
    hires_fix_use_firstpass_conds: Union[Unset, Any] = UNSET
    use_old_scheduling: Union[Unset, Any] = UNSET
    use_downcasted_alpha_bar: Union[Unset, Any] = UNSET
    interrogate_keep_models_in_memory: Union[Unset, Any] = UNSET
    interrogate_return_ranks: Union[Unset, Any] = UNSET
    interrogate_clip_num_beams: Union[Unset, float] = 1.0
    interrogate_clip_min_length: Union[Unset, float] = 24.0
    interrogate_clip_max_length: Union[Unset, float] = 48.0
    interrogate_clip_dict_limit: Union[Unset, float] = 1500.0
    interrogate_clip_skip_categories: Union[Unset, Any] = UNSET
    interrogate_deepbooru_score_threshold: Union[Unset, float] = 0.5
    deepbooru_sort_alpha: Union[Unset, bool] = True
    deepbooru_use_spaces: Union[Unset, bool] = True
    deepbooru_escape: Union[Unset, bool] = True
    deepbooru_filter_tags: Union[Unset, Any] = UNSET
    extra_networks_show_hidden_directories: Union[Unset, bool] = True
    extra_networks_dir_button_function: Union[Unset, Any] = UNSET
    extra_networks_hidden_models: Union[Unset, str] = "When searched"
    extra_networks_default_multiplier: Union[Unset, float] = 1.0
    extra_networks_card_width: Union[Unset, Any] = UNSET
    extra_networks_card_height: Union[Unset, Any] = UNSET
    extra_networks_card_text_scale: Union[Unset, float] = 1.0
    extra_networks_card_show_desc: Union[Unset, bool] = True
    extra_networks_card_description_is_html: Union[Unset, Any] = UNSET
    extra_networks_card_order_field: Union[Unset, str] = "Path"
    extra_networks_card_order: Union[Unset, str] = "Ascending"
    extra_networks_tree_view_default_enabled: Union[Unset, Any] = UNSET
    extra_networks_add_text_separator: Union[Unset, str] = " "
    ui_extra_networks_tab_reorder: Union[Unset, Any] = UNSET
    textual_inversion_print_at_load: Union[Unset, Any] = UNSET
    textual_inversion_add_hashes_to_infotext: Union[Unset, bool] = True
    sd_hypernetwork: Union[Unset, str] = "None"
    keyedit_precision_attention: Union[Unset, float] = 0.1
    keyedit_precision_extra: Union[Unset, float] = 0.05
    keyedit_delimiters: Union[Unset, str] = ".,\\/!?%^*;:{}=`~() "
    keyedit_delimiters_whitespace: Union[Unset, List[Any]] = UNSET
    keyedit_move: Union[Unset, bool] = True
    disable_token_counters: Union[Unset, Any] = UNSET
    include_styles_into_token_counters: Union[Unset, bool] = True
    return_grid: Union[Unset, bool] = True
    do_not_show_images: Union[Unset, Any] = UNSET
    js_modal_lightbox: Union[Unset, bool] = True
    js_modal_lightbox_initially_zoomed: Union[Unset, bool] = True
    js_modal_lightbox_gamepad: Union[Unset, Any] = UNSET
    js_modal_lightbox_gamepad_repeat: Union[Unset, float] = 250.0
    sd_webui_modal_lightbox_icon_opacity: Union[Unset, float] = 1.0
    sd_webui_modal_lightbox_toolbar_opacity: Union[Unset, float] = 0.9
    gallery_height: Union[Unset, Any] = UNSET
    open_dir_button_choice: Union[Unset, str] = "Subdirectory"
    compact_prompt_box: Union[Unset, Any] = UNSET
    samplers_in_dropdown: Union[Unset, bool] = True
    dimensions_and_batch_together: Union[Unset, bool] = True
    sd_checkpoint_dropdown_use_short: Union[Unset, Any] = UNSET
    hires_fix_show_sampler: Union[Unset, Any] = UNSET
    hires_fix_show_prompts: Union[Unset, Any] = UNSET
    txt2img_settings_accordion: Union[Unset, Any] = UNSET
    img2img_settings_accordion: Union[Unset, Any] = UNSET
    interrupt_after_current: Union[Unset, bool] = True
    localization: Union[Unset, str] = "None"
    quicksettings_list: Union[Unset, List[Any]] = UNSET
    ui_tab_order: Union[Unset, Any] = UNSET
    hidden_tabs: Union[Unset, Any] = UNSET
    ui_reorder_list: Union[Unset, Any] = UNSET
    gradio_theme: Union[Unset, str] = "Default"
    gradio_themes_cache: Union[Unset, bool] = True
    show_progress_in_title: Union[Unset, bool] = True
    send_seed: Union[Unset, bool] = True
    send_size: Union[Unset, bool] = True
    infotext_explanation: Union[Unset, str] = (
        "Infotext is what this software calls the text that contains generation parameters and can be used to generate the same picture again.\nIt is displayed in UI below the image. To use infotext, paste it into the prompt and click the ↙️ paste button."
    )
    enable_pnginfo: Union[Unset, bool] = True
    save_txt: Union[Unset, Any] = UNSET
    add_model_name_to_info: Union[Unset, bool] = True
    add_model_hash_to_info: Union[Unset, bool] = True
    add_vae_name_to_info: Union[Unset, bool] = True
    add_vae_hash_to_info: Union[Unset, bool] = True
    add_user_name_to_info: Union[Unset, Any] = UNSET
    add_version_to_infotext: Union[Unset, bool] = True
    disable_weights_auto_swap: Union[Unset, bool] = True
    infotext_skip_pasting: Union[Unset, Any] = UNSET
    infotext_styles: Union[Unset, str] = "Apply if any"
    show_progressbar: Union[Unset, bool] = True
    live_previews_enable: Union[Unset, bool] = True
    live_previews_image_format: Union[Unset, str] = "png"
    show_progress_grid: Union[Unset, bool] = True
    show_progress_every_n_steps: Union[Unset, float] = 10.0
    show_progress_type: Union[Unset, str] = "Approx NN"
    live_preview_allow_lowvram_full: Union[Unset, Any] = UNSET
    live_preview_content: Union[Unset, str] = "Prompt"
    live_preview_refresh_period: Union[Unset, float] = 1000.0
    live_preview_fast_interrupt: Union[Unset, Any] = UNSET
    js_live_preview_in_modal_lightbox: Union[Unset, Any] = UNSET
    hide_samplers: Union[Unset, Any] = UNSET
    eta_ddim: Union[Unset, Any] = UNSET
    eta_ancestral: Union[Unset, float] = 1.0
    ddim_discretize: Union[Unset, str] = "uniform"
    s_churn: Union[Unset, Any] = UNSET
    s_tmin: Union[Unset, Any] = UNSET
    s_tmax: Union[Unset, Any] = UNSET
    s_noise: Union[Unset, float] = 1.0
    k_sched_type: Union[Unset, str] = "Automatic"
    sigma_min: Union[Unset, Any] = UNSET
    sigma_max: Union[Unset, Any] = UNSET
    rho: Union[Unset, Any] = UNSET
    eta_noise_seed_delta: Union[Unset, Any] = UNSET
    always_discard_next_to_last_sigma: Union[Unset, Any] = UNSET
    sgm_noise_multiplier: Union[Unset, Any] = UNSET
    uni_pc_variant: Union[Unset, str] = "bh1"
    uni_pc_skip_type: Union[Unset, str] = "time_uniform"
    uni_pc_order: Union[Unset, float] = 3.0
    uni_pc_lower_order_final: Union[Unset, bool] = True
    sd_noise_schedule: Union[Unset, str] = "Default"
    postprocessing_enable_in_main_ui: Union[Unset, Any] = UNSET
    postprocessing_operation_order: Union[Unset, Any] = UNSET
    upscaling_max_images_in_cache: Union[Unset, float] = 5.0
    postprocessing_existing_caption_action: Union[Unset, str] = "Ignore"
    disabled_extensions: Union[Unset, Any] = UNSET
    disable_all_extensions: Union[Unset, str] = "none"
    restore_config_state_file: Union[Unset, Any] = UNSET
    sd_checkpoint_hash: Union[Unset, Any] = UNSET
    sd_lora: Union[Unset, str] = "None"
    lora_preferred_name: Union[Unset, str] = "Alias from file"
    lora_add_hashes_to_infotext: Union[Unset, bool] = True
    lora_show_all: Union[Unset, Any] = UNSET
    lora_hide_unknown_for_versions: Union[Unset, Any] = UNSET
    lora_in_memory_limit: Union[Unset, Any] = UNSET
    lora_not_found_warning_console: Union[Unset, Any] = UNSET
    lora_not_found_gradio_warning: Union[Unset, Any] = UNSET
    lora_functional: Union[Unset, Any] = UNSET
    canvas_hotkey_zoom: Union[Unset, str] = "Alt"
    canvas_hotkey_adjust: Union[Unset, str] = "Ctrl"
    canvas_hotkey_shrink_brush: Union[Unset, str] = "Q"
    canvas_hotkey_grow_brush: Union[Unset, str] = "W"
    canvas_hotkey_move: Union[Unset, str] = "F"
    canvas_hotkey_fullscreen: Union[Unset, str] = "S"
    canvas_hotkey_reset: Union[Unset, str] = "R"
    canvas_hotkey_overlap: Union[Unset, str] = "O"
    canvas_show_tooltip: Union[Unset, bool] = True
    canvas_auto_expand: Union[Unset, bool] = True
    canvas_blur_prompt: Union[Unset, Any] = UNSET
    canvas_disabled_functions: Union[Unset, List[Any]] = UNSET
    settings_in_ui: Union[Unset, str] = (
        "This page allows you to add some settings to the main interface of txt2img and img2img tabs."
    )
    extra_options_txt2img: Union[Unset, Any] = UNSET
    extra_options_img2img: Union[Unset, Any] = UNSET
    extra_options_cols: Union[Unset, float] = 1.0
    extra_options_accordion: Union[Unset, Any] = UNSET
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        samples_save = self.samples_save

        samples_format = self.samples_format

        samples_filename_pattern = self.samples_filename_pattern

        save_images_add_number = self.save_images_add_number

        save_images_replace_action = self.save_images_replace_action

        grid_save = self.grid_save

        grid_format = self.grid_format

        grid_extended_filename = self.grid_extended_filename

        grid_only_if_multiple = self.grid_only_if_multiple

        grid_prevent_empty_spots = self.grid_prevent_empty_spots

        grid_zip_filename_pattern = self.grid_zip_filename_pattern

        n_rows = self.n_rows

        font = self.font

        grid_text_active_color = self.grid_text_active_color

        grid_text_inactive_color = self.grid_text_inactive_color

        grid_background_color = self.grid_background_color

        save_images_before_face_restoration = self.save_images_before_face_restoration

        save_images_before_highres_fix = self.save_images_before_highres_fix

        save_images_before_color_correction = self.save_images_before_color_correction

        save_mask = self.save_mask

        save_mask_composite = self.save_mask_composite

        jpeg_quality = self.jpeg_quality

        webp_lossless = self.webp_lossless

        export_for_4chan = self.export_for_4chan

        img_downscale_threshold = self.img_downscale_threshold

        target_side_length = self.target_side_length

        img_max_size_mp = self.img_max_size_mp

        use_original_name_batch = self.use_original_name_batch

        use_upscaler_name_as_suffix = self.use_upscaler_name_as_suffix

        save_selected_only = self.save_selected_only

        save_init_img = self.save_init_img

        temp_dir = self.temp_dir

        clean_temp_dir_at_start = self.clean_temp_dir_at_start

        save_incomplete_images = self.save_incomplete_images

        notification_audio = self.notification_audio

        notification_volume = self.notification_volume

        outdir_samples = self.outdir_samples

        outdir_txt2img_samples = self.outdir_txt2img_samples

        outdir_img2img_samples = self.outdir_img2img_samples

        outdir_extras_samples = self.outdir_extras_samples

        outdir_grids = self.outdir_grids

        outdir_txt2img_grids = self.outdir_txt2img_grids

        outdir_img2img_grids = self.outdir_img2img_grids

        outdir_save = self.outdir_save

        outdir_init_images = self.outdir_init_images

        save_to_dirs = self.save_to_dirs

        grid_save_to_dirs = self.grid_save_to_dirs

        use_save_to_dirs_for_ui = self.use_save_to_dirs_for_ui

        directories_filename_pattern = self.directories_filename_pattern

        directories_max_prompt_words = self.directories_max_prompt_words

        esrgan_tile = self.esrgan_tile

        esrgan_tile_overlap = self.esrgan_tile_overlap

        realesrgan_enabled_models: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.realesrgan_enabled_models, Unset):
            realesrgan_enabled_models = self.realesrgan_enabled_models

        dat_enabled_models: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.dat_enabled_models, Unset):
            dat_enabled_models = self.dat_enabled_models

        dat_tile = self.dat_tile

        dat_tile_overlap = self.dat_tile_overlap

        upscaler_for_img2img = self.upscaler_for_img2img

        face_restoration = self.face_restoration

        face_restoration_model = self.face_restoration_model

        code_former_weight = self.code_former_weight

        face_restoration_unload = self.face_restoration_unload

        auto_launch_browser = self.auto_launch_browser

        enable_console_prompts = self.enable_console_prompts

        show_warnings = self.show_warnings

        show_gradio_deprecation_warnings = self.show_gradio_deprecation_warnings

        memmon_poll_rate = self.memmon_poll_rate

        samples_log_stdout = self.samples_log_stdout

        multiple_tqdm = self.multiple_tqdm

        enable_upscale_progressbar = self.enable_upscale_progressbar

        print_hypernet_extra = self.print_hypernet_extra

        list_hidden_files = self.list_hidden_files

        disable_mmap_load_safetensors = self.disable_mmap_load_safetensors

        hide_ldm_prints = self.hide_ldm_prints

        dump_stacks_on_signal = self.dump_stacks_on_signal

        api_enable_requests = self.api_enable_requests

        api_forbid_local_requests = self.api_forbid_local_requests

        api_useragent = self.api_useragent

        unload_models_when_training = self.unload_models_when_training

        pin_memory = self.pin_memory

        save_optimizer_state = self.save_optimizer_state

        save_training_settings_to_txt = self.save_training_settings_to_txt

        dataset_filename_word_regex = self.dataset_filename_word_regex

        dataset_filename_join_string = self.dataset_filename_join_string

        training_image_repeats_per_epoch = self.training_image_repeats_per_epoch

        training_write_csv_every = self.training_write_csv_every

        training_xattention_optimizations = self.training_xattention_optimizations

        training_enable_tensorboard = self.training_enable_tensorboard

        training_tensorboard_save_images = self.training_tensorboard_save_images

        training_tensorboard_flush_every = self.training_tensorboard_flush_every

        sd_model_checkpoint = self.sd_model_checkpoint

        sd_checkpoints_limit = self.sd_checkpoints_limit

        sd_checkpoints_keep_in_cpu = self.sd_checkpoints_keep_in_cpu

        sd_checkpoint_cache = self.sd_checkpoint_cache

        sd_unet = self.sd_unet

        enable_quantization = self.enable_quantization

        emphasis = self.emphasis

        enable_batch_seeds = self.enable_batch_seeds

        comma_padding_backtrack = self.comma_padding_backtrack

        clip_stop_at_last_layers = self.clip_stop_at_last_layers

        upcast_attn = self.upcast_attn

        randn_source = self.randn_source

        tiling = self.tiling

        hires_fix_refiner_pass = self.hires_fix_refiner_pass

        sdxl_crop_top = self.sdxl_crop_top

        sdxl_crop_left = self.sdxl_crop_left

        sdxl_refiner_low_aesthetic_score = self.sdxl_refiner_low_aesthetic_score

        sdxl_refiner_high_aesthetic_score = self.sdxl_refiner_high_aesthetic_score

        sd_vae_explanation = self.sd_vae_explanation

        sd_vae_checkpoint_cache = self.sd_vae_checkpoint_cache

        sd_vae = self.sd_vae

        sd_vae_overrides_per_model_preferences = self.sd_vae_overrides_per_model_preferences

        auto_vae_precision_bfloat16 = self.auto_vae_precision_bfloat16

        auto_vae_precision = self.auto_vae_precision

        sd_vae_encode_method = self.sd_vae_encode_method

        sd_vae_decode_method = self.sd_vae_decode_method

        inpainting_mask_weight = self.inpainting_mask_weight

        initial_noise_multiplier = self.initial_noise_multiplier

        img2img_extra_noise = self.img2img_extra_noise

        img2img_color_correction = self.img2img_color_correction

        img2img_fix_steps = self.img2img_fix_steps

        img2img_background_color = self.img2img_background_color

        img2img_editor_height = self.img2img_editor_height

        img2img_sketch_default_brush_color = self.img2img_sketch_default_brush_color

        img2img_inpaint_mask_brush_color = self.img2img_inpaint_mask_brush_color

        img2img_inpaint_sketch_default_brush_color = self.img2img_inpaint_sketch_default_brush_color

        return_mask = self.return_mask

        return_mask_composite = self.return_mask_composite

        img2img_batch_show_results_limit = self.img2img_batch_show_results_limit

        overlay_inpaint = self.overlay_inpaint

        cross_attention_optimization = self.cross_attention_optimization

        s_min_uncond = self.s_min_uncond

        token_merging_ratio = self.token_merging_ratio

        token_merging_ratio_img2img = self.token_merging_ratio_img2img

        token_merging_ratio_hr = self.token_merging_ratio_hr

        pad_cond_uncond = self.pad_cond_uncond

        pad_cond_uncond_v0 = self.pad_cond_uncond_v0

        persistent_cond_cache = self.persistent_cond_cache

        batch_cond_uncond = self.batch_cond_uncond

        fp8_storage = self.fp8_storage

        cache_fp16_weight = self.cache_fp16_weight

        auto_backcompat = self.auto_backcompat

        use_old_emphasis_implementation = self.use_old_emphasis_implementation

        use_old_karras_scheduler_sigmas = self.use_old_karras_scheduler_sigmas

        no_dpmpp_sde_batch_determinism = self.no_dpmpp_sde_batch_determinism

        use_old_hires_fix_width_height = self.use_old_hires_fix_width_height

        dont_fix_second_order_samplers_schedule = self.dont_fix_second_order_samplers_schedule

        hires_fix_use_firstpass_conds = self.hires_fix_use_firstpass_conds

        use_old_scheduling = self.use_old_scheduling

        use_downcasted_alpha_bar = self.use_downcasted_alpha_bar

        interrogate_keep_models_in_memory = self.interrogate_keep_models_in_memory

        interrogate_return_ranks = self.interrogate_return_ranks

        interrogate_clip_num_beams = self.interrogate_clip_num_beams

        interrogate_clip_min_length = self.interrogate_clip_min_length

        interrogate_clip_max_length = self.interrogate_clip_max_length

        interrogate_clip_dict_limit = self.interrogate_clip_dict_limit

        interrogate_clip_skip_categories = self.interrogate_clip_skip_categories

        interrogate_deepbooru_score_threshold = self.interrogate_deepbooru_score_threshold

        deepbooru_sort_alpha = self.deepbooru_sort_alpha

        deepbooru_use_spaces = self.deepbooru_use_spaces

        deepbooru_escape = self.deepbooru_escape

        deepbooru_filter_tags = self.deepbooru_filter_tags

        extra_networks_show_hidden_directories = self.extra_networks_show_hidden_directories

        extra_networks_dir_button_function = self.extra_networks_dir_button_function

        extra_networks_hidden_models = self.extra_networks_hidden_models

        extra_networks_default_multiplier = self.extra_networks_default_multiplier

        extra_networks_card_width = self.extra_networks_card_width

        extra_networks_card_height = self.extra_networks_card_height

        extra_networks_card_text_scale = self.extra_networks_card_text_scale

        extra_networks_card_show_desc = self.extra_networks_card_show_desc

        extra_networks_card_description_is_html = self.extra_networks_card_description_is_html

        extra_networks_card_order_field = self.extra_networks_card_order_field

        extra_networks_card_order = self.extra_networks_card_order

        extra_networks_tree_view_default_enabled = self.extra_networks_tree_view_default_enabled

        extra_networks_add_text_separator = self.extra_networks_add_text_separator

        ui_extra_networks_tab_reorder = self.ui_extra_networks_tab_reorder

        textual_inversion_print_at_load = self.textual_inversion_print_at_load

        textual_inversion_add_hashes_to_infotext = self.textual_inversion_add_hashes_to_infotext

        sd_hypernetwork = self.sd_hypernetwork

        keyedit_precision_attention = self.keyedit_precision_attention

        keyedit_precision_extra = self.keyedit_precision_extra

        keyedit_delimiters = self.keyedit_delimiters

        keyedit_delimiters_whitespace: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.keyedit_delimiters_whitespace, Unset):
            keyedit_delimiters_whitespace = self.keyedit_delimiters_whitespace

        keyedit_move = self.keyedit_move

        disable_token_counters = self.disable_token_counters

        include_styles_into_token_counters = self.include_styles_into_token_counters

        return_grid = self.return_grid

        do_not_show_images = self.do_not_show_images

        js_modal_lightbox = self.js_modal_lightbox

        js_modal_lightbox_initially_zoomed = self.js_modal_lightbox_initially_zoomed

        js_modal_lightbox_gamepad = self.js_modal_lightbox_gamepad

        js_modal_lightbox_gamepad_repeat = self.js_modal_lightbox_gamepad_repeat

        sd_webui_modal_lightbox_icon_opacity = self.sd_webui_modal_lightbox_icon_opacity

        sd_webui_modal_lightbox_toolbar_opacity = self.sd_webui_modal_lightbox_toolbar_opacity

        gallery_height = self.gallery_height

        open_dir_button_choice = self.open_dir_button_choice

        compact_prompt_box = self.compact_prompt_box

        samplers_in_dropdown = self.samplers_in_dropdown

        dimensions_and_batch_together = self.dimensions_and_batch_together

        sd_checkpoint_dropdown_use_short = self.sd_checkpoint_dropdown_use_short

        hires_fix_show_sampler = self.hires_fix_show_sampler

        hires_fix_show_prompts = self.hires_fix_show_prompts

        txt2img_settings_accordion = self.txt2img_settings_accordion

        img2img_settings_accordion = self.img2img_settings_accordion

        interrupt_after_current = self.interrupt_after_current

        localization = self.localization

        quicksettings_list: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.quicksettings_list, Unset):
            quicksettings_list = self.quicksettings_list

        ui_tab_order = self.ui_tab_order

        hidden_tabs = self.hidden_tabs

        ui_reorder_list = self.ui_reorder_list

        gradio_theme = self.gradio_theme

        gradio_themes_cache = self.gradio_themes_cache

        show_progress_in_title = self.show_progress_in_title

        send_seed = self.send_seed

        send_size = self.send_size

        infotext_explanation = self.infotext_explanation

        enable_pnginfo = self.enable_pnginfo

        save_txt = self.save_txt

        add_model_name_to_info = self.add_model_name_to_info

        add_model_hash_to_info = self.add_model_hash_to_info

        add_vae_name_to_info = self.add_vae_name_to_info

        add_vae_hash_to_info = self.add_vae_hash_to_info

        add_user_name_to_info = self.add_user_name_to_info

        add_version_to_infotext = self.add_version_to_infotext

        disable_weights_auto_swap = self.disable_weights_auto_swap

        infotext_skip_pasting = self.infotext_skip_pasting

        infotext_styles = self.infotext_styles

        show_progressbar = self.show_progressbar

        live_previews_enable = self.live_previews_enable

        live_previews_image_format = self.live_previews_image_format

        show_progress_grid = self.show_progress_grid

        show_progress_every_n_steps = self.show_progress_every_n_steps

        show_progress_type = self.show_progress_type

        live_preview_allow_lowvram_full = self.live_preview_allow_lowvram_full

        live_preview_content = self.live_preview_content

        live_preview_refresh_period = self.live_preview_refresh_period

        live_preview_fast_interrupt = self.live_preview_fast_interrupt

        js_live_preview_in_modal_lightbox = self.js_live_preview_in_modal_lightbox

        hide_samplers = self.hide_samplers

        eta_ddim = self.eta_ddim

        eta_ancestral = self.eta_ancestral

        ddim_discretize = self.ddim_discretize

        s_churn = self.s_churn

        s_tmin = self.s_tmin

        s_tmax = self.s_tmax

        s_noise = self.s_noise

        k_sched_type = self.k_sched_type

        sigma_min = self.sigma_min

        sigma_max = self.sigma_max

        rho = self.rho

        eta_noise_seed_delta = self.eta_noise_seed_delta

        always_discard_next_to_last_sigma = self.always_discard_next_to_last_sigma

        sgm_noise_multiplier = self.sgm_noise_multiplier

        uni_pc_variant = self.uni_pc_variant

        uni_pc_skip_type = self.uni_pc_skip_type

        uni_pc_order = self.uni_pc_order

        uni_pc_lower_order_final = self.uni_pc_lower_order_final

        sd_noise_schedule = self.sd_noise_schedule

        postprocessing_enable_in_main_ui = self.postprocessing_enable_in_main_ui

        postprocessing_operation_order = self.postprocessing_operation_order

        upscaling_max_images_in_cache = self.upscaling_max_images_in_cache

        postprocessing_existing_caption_action = self.postprocessing_existing_caption_action

        disabled_extensions = self.disabled_extensions

        disable_all_extensions = self.disable_all_extensions

        restore_config_state_file = self.restore_config_state_file

        sd_checkpoint_hash = self.sd_checkpoint_hash

        sd_lora = self.sd_lora

        lora_preferred_name = self.lora_preferred_name

        lora_add_hashes_to_infotext = self.lora_add_hashes_to_infotext

        lora_show_all = self.lora_show_all

        lora_hide_unknown_for_versions = self.lora_hide_unknown_for_versions

        lora_in_memory_limit = self.lora_in_memory_limit

        lora_not_found_warning_console = self.lora_not_found_warning_console

        lora_not_found_gradio_warning = self.lora_not_found_gradio_warning

        lora_functional = self.lora_functional

        canvas_hotkey_zoom = self.canvas_hotkey_zoom

        canvas_hotkey_adjust = self.canvas_hotkey_adjust

        canvas_hotkey_shrink_brush = self.canvas_hotkey_shrink_brush

        canvas_hotkey_grow_brush = self.canvas_hotkey_grow_brush

        canvas_hotkey_move = self.canvas_hotkey_move

        canvas_hotkey_fullscreen = self.canvas_hotkey_fullscreen

        canvas_hotkey_reset = self.canvas_hotkey_reset

        canvas_hotkey_overlap = self.canvas_hotkey_overlap

        canvas_show_tooltip = self.canvas_show_tooltip

        canvas_auto_expand = self.canvas_auto_expand

        canvas_blur_prompt = self.canvas_blur_prompt

        canvas_disabled_functions: Union[Unset, List[Any]] = UNSET
        if not isinstance(self.canvas_disabled_functions, Unset):
            canvas_disabled_functions = self.canvas_disabled_functions

        settings_in_ui = self.settings_in_ui

        extra_options_txt2img = self.extra_options_txt2img

        extra_options_img2img = self.extra_options_img2img

        extra_options_cols = self.extra_options_cols

        extra_options_accordion = self.extra_options_accordion

        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({})
        if samples_save is not UNSET:
            field_dict["samples_save"] = samples_save
        if samples_format is not UNSET:
            field_dict["samples_format"] = samples_format
        if samples_filename_pattern is not UNSET:
            field_dict["samples_filename_pattern"] = samples_filename_pattern
        if save_images_add_number is not UNSET:
            field_dict["save_images_add_number"] = save_images_add_number
        if save_images_replace_action is not UNSET:
            field_dict["save_images_replace_action"] = save_images_replace_action
        if grid_save is not UNSET:
            field_dict["grid_save"] = grid_save
        if grid_format is not UNSET:
            field_dict["grid_format"] = grid_format
        if grid_extended_filename is not UNSET:
            field_dict["grid_extended_filename"] = grid_extended_filename
        if grid_only_if_multiple is not UNSET:
            field_dict["grid_only_if_multiple"] = grid_only_if_multiple
        if grid_prevent_empty_spots is not UNSET:
            field_dict["grid_prevent_empty_spots"] = grid_prevent_empty_spots
        if grid_zip_filename_pattern is not UNSET:
            field_dict["grid_zip_filename_pattern"] = grid_zip_filename_pattern
        if n_rows is not UNSET:
            field_dict["n_rows"] = n_rows
        if font is not UNSET:
            field_dict["font"] = font
        if grid_text_active_color is not UNSET:
            field_dict["grid_text_active_color"] = grid_text_active_color
        if grid_text_inactive_color is not UNSET:
            field_dict["grid_text_inactive_color"] = grid_text_inactive_color
        if grid_background_color is not UNSET:
            field_dict["grid_background_color"] = grid_background_color
        if save_images_before_face_restoration is not UNSET:
            field_dict["save_images_before_face_restoration"] = save_images_before_face_restoration
        if save_images_before_highres_fix is not UNSET:
            field_dict["save_images_before_highres_fix"] = save_images_before_highres_fix
        if save_images_before_color_correction is not UNSET:
            field_dict["save_images_before_color_correction"] = save_images_before_color_correction
        if save_mask is not UNSET:
            field_dict["save_mask"] = save_mask
        if save_mask_composite is not UNSET:
            field_dict["save_mask_composite"] = save_mask_composite
        if jpeg_quality is not UNSET:
            field_dict["jpeg_quality"] = jpeg_quality
        if webp_lossless is not UNSET:
            field_dict["webp_lossless"] = webp_lossless
        if export_for_4chan is not UNSET:
            field_dict["export_for_4chan"] = export_for_4chan
        if img_downscale_threshold is not UNSET:
            field_dict["img_downscale_threshold"] = img_downscale_threshold
        if target_side_length is not UNSET:
            field_dict["target_side_length"] = target_side_length
        if img_max_size_mp is not UNSET:
            field_dict["img_max_size_mp"] = img_max_size_mp
        if use_original_name_batch is not UNSET:
            field_dict["use_original_name_batch"] = use_original_name_batch
        if use_upscaler_name_as_suffix is not UNSET:
            field_dict["use_upscaler_name_as_suffix"] = use_upscaler_name_as_suffix
        if save_selected_only is not UNSET:
            field_dict["save_selected_only"] = save_selected_only
        if save_init_img is not UNSET:
            field_dict["save_init_img"] = save_init_img
        if temp_dir is not UNSET:
            field_dict["temp_dir"] = temp_dir
        if clean_temp_dir_at_start is not UNSET:
            field_dict["clean_temp_dir_at_start"] = clean_temp_dir_at_start
        if save_incomplete_images is not UNSET:
            field_dict["save_incomplete_images"] = save_incomplete_images
        if notification_audio is not UNSET:
            field_dict["notification_audio"] = notification_audio
        if notification_volume is not UNSET:
            field_dict["notification_volume"] = notification_volume
        if outdir_samples is not UNSET:
            field_dict["outdir_samples"] = outdir_samples
        if outdir_txt2img_samples is not UNSET:
            field_dict["outdir_txt2img_samples"] = outdir_txt2img_samples
        if outdir_img2img_samples is not UNSET:
            field_dict["outdir_img2img_samples"] = outdir_img2img_samples
        if outdir_extras_samples is not UNSET:
            field_dict["outdir_extras_samples"] = outdir_extras_samples
        if outdir_grids is not UNSET:
            field_dict["outdir_grids"] = outdir_grids
        if outdir_txt2img_grids is not UNSET:
            field_dict["outdir_txt2img_grids"] = outdir_txt2img_grids
        if outdir_img2img_grids is not UNSET:
            field_dict["outdir_img2img_grids"] = outdir_img2img_grids
        if outdir_save is not UNSET:
            field_dict["outdir_save"] = outdir_save
        if outdir_init_images is not UNSET:
            field_dict["outdir_init_images"] = outdir_init_images
        if save_to_dirs is not UNSET:
            field_dict["save_to_dirs"] = save_to_dirs
        if grid_save_to_dirs is not UNSET:
            field_dict["grid_save_to_dirs"] = grid_save_to_dirs
        if use_save_to_dirs_for_ui is not UNSET:
            field_dict["use_save_to_dirs_for_ui"] = use_save_to_dirs_for_ui
        if directories_filename_pattern is not UNSET:
            field_dict["directories_filename_pattern"] = directories_filename_pattern
        if directories_max_prompt_words is not UNSET:
            field_dict["directories_max_prompt_words"] = directories_max_prompt_words
        if esrgan_tile is not UNSET:
            field_dict["ESRGAN_tile"] = esrgan_tile
        if esrgan_tile_overlap is not UNSET:
            field_dict["ESRGAN_tile_overlap"] = esrgan_tile_overlap
        if realesrgan_enabled_models is not UNSET:
            field_dict["realesrgan_enabled_models"] = realesrgan_enabled_models
        if dat_enabled_models is not UNSET:
            field_dict["dat_enabled_models"] = dat_enabled_models
        if dat_tile is not UNSET:
            field_dict["DAT_tile"] = dat_tile
        if dat_tile_overlap is not UNSET:
            field_dict["DAT_tile_overlap"] = dat_tile_overlap
        if upscaler_for_img2img is not UNSET:
            field_dict["upscaler_for_img2img"] = upscaler_for_img2img
        if face_restoration is not UNSET:
            field_dict["face_restoration"] = face_restoration
        if face_restoration_model is not UNSET:
            field_dict["face_restoration_model"] = face_restoration_model
        if code_former_weight is not UNSET:
            field_dict["code_former_weight"] = code_former_weight
        if face_restoration_unload is not UNSET:
            field_dict["face_restoration_unload"] = face_restoration_unload
        if auto_launch_browser is not UNSET:
            field_dict["auto_launch_browser"] = auto_launch_browser
        if enable_console_prompts is not UNSET:
            field_dict["enable_console_prompts"] = enable_console_prompts
        if show_warnings is not UNSET:
            field_dict["show_warnings"] = show_warnings
        if show_gradio_deprecation_warnings is not UNSET:
            field_dict["show_gradio_deprecation_warnings"] = show_gradio_deprecation_warnings
        if memmon_poll_rate is not UNSET:
            field_dict["memmon_poll_rate"] = memmon_poll_rate
        if samples_log_stdout is not UNSET:
            field_dict["samples_log_stdout"] = samples_log_stdout
        if multiple_tqdm is not UNSET:
            field_dict["multiple_tqdm"] = multiple_tqdm
        if enable_upscale_progressbar is not UNSET:
            field_dict["enable_upscale_progressbar"] = enable_upscale_progressbar
        if print_hypernet_extra is not UNSET:
            field_dict["print_hypernet_extra"] = print_hypernet_extra
        if list_hidden_files is not UNSET:
            field_dict["list_hidden_files"] = list_hidden_files
        if disable_mmap_load_safetensors is not UNSET:
            field_dict["disable_mmap_load_safetensors"] = disable_mmap_load_safetensors
        if hide_ldm_prints is not UNSET:
            field_dict["hide_ldm_prints"] = hide_ldm_prints
        if dump_stacks_on_signal is not UNSET:
            field_dict["dump_stacks_on_signal"] = dump_stacks_on_signal
        if api_enable_requests is not UNSET:
            field_dict["api_enable_requests"] = api_enable_requests
        if api_forbid_local_requests is not UNSET:
            field_dict["api_forbid_local_requests"] = api_forbid_local_requests
        if api_useragent is not UNSET:
            field_dict["api_useragent"] = api_useragent
        if unload_models_when_training is not UNSET:
            field_dict["unload_models_when_training"] = unload_models_when_training
        if pin_memory is not UNSET:
            field_dict["pin_memory"] = pin_memory
        if save_optimizer_state is not UNSET:
            field_dict["save_optimizer_state"] = save_optimizer_state
        if save_training_settings_to_txt is not UNSET:
            field_dict["save_training_settings_to_txt"] = save_training_settings_to_txt
        if dataset_filename_word_regex is not UNSET:
            field_dict["dataset_filename_word_regex"] = dataset_filename_word_regex
        if dataset_filename_join_string is not UNSET:
            field_dict["dataset_filename_join_string"] = dataset_filename_join_string
        if training_image_repeats_per_epoch is not UNSET:
            field_dict["training_image_repeats_per_epoch"] = training_image_repeats_per_epoch
        if training_write_csv_every is not UNSET:
            field_dict["training_write_csv_every"] = training_write_csv_every
        if training_xattention_optimizations is not UNSET:
            field_dict["training_xattention_optimizations"] = training_xattention_optimizations
        if training_enable_tensorboard is not UNSET:
            field_dict["training_enable_tensorboard"] = training_enable_tensorboard
        if training_tensorboard_save_images is not UNSET:
            field_dict["training_tensorboard_save_images"] = training_tensorboard_save_images
        if training_tensorboard_flush_every is not UNSET:
            field_dict["training_tensorboard_flush_every"] = training_tensorboard_flush_every
        if sd_model_checkpoint is not UNSET:
            field_dict["sd_model_checkpoint"] = sd_model_checkpoint
        if sd_checkpoints_limit is not UNSET:
            field_dict["sd_checkpoints_limit"] = sd_checkpoints_limit
        if sd_checkpoints_keep_in_cpu is not UNSET:
            field_dict["sd_checkpoints_keep_in_cpu"] = sd_checkpoints_keep_in_cpu
        if sd_checkpoint_cache is not UNSET:
            field_dict["sd_checkpoint_cache"] = sd_checkpoint_cache
        if sd_unet is not UNSET:
            field_dict["sd_unet"] = sd_unet
        if enable_quantization is not UNSET:
            field_dict["enable_quantization"] = enable_quantization
        if emphasis is not UNSET:
            field_dict["emphasis"] = emphasis
        if enable_batch_seeds is not UNSET:
            field_dict["enable_batch_seeds"] = enable_batch_seeds
        if comma_padding_backtrack is not UNSET:
            field_dict["comma_padding_backtrack"] = comma_padding_backtrack
        if clip_stop_at_last_layers is not UNSET:
            field_dict["CLIP_stop_at_last_layers"] = clip_stop_at_last_layers
        if upcast_attn is not UNSET:
            field_dict["upcast_attn"] = upcast_attn
        if randn_source is not UNSET:
            field_dict["randn_source"] = randn_source
        if tiling is not UNSET:
            field_dict["tiling"] = tiling
        if hires_fix_refiner_pass is not UNSET:
            field_dict["hires_fix_refiner_pass"] = hires_fix_refiner_pass
        if sdxl_crop_top is not UNSET:
            field_dict["sdxl_crop_top"] = sdxl_crop_top
        if sdxl_crop_left is not UNSET:
            field_dict["sdxl_crop_left"] = sdxl_crop_left
        if sdxl_refiner_low_aesthetic_score is not UNSET:
            field_dict["sdxl_refiner_low_aesthetic_score"] = sdxl_refiner_low_aesthetic_score
        if sdxl_refiner_high_aesthetic_score is not UNSET:
            field_dict["sdxl_refiner_high_aesthetic_score"] = sdxl_refiner_high_aesthetic_score
        if sd_vae_explanation is not UNSET:
            field_dict["sd_vae_explanation"] = sd_vae_explanation
        if sd_vae_checkpoint_cache is not UNSET:
            field_dict["sd_vae_checkpoint_cache"] = sd_vae_checkpoint_cache
        if sd_vae is not UNSET:
            field_dict["sd_vae"] = sd_vae
        if sd_vae_overrides_per_model_preferences is not UNSET:
            field_dict["sd_vae_overrides_per_model_preferences"] = sd_vae_overrides_per_model_preferences
        if auto_vae_precision_bfloat16 is not UNSET:
            field_dict["auto_vae_precision_bfloat16"] = auto_vae_precision_bfloat16
        if auto_vae_precision is not UNSET:
            field_dict["auto_vae_precision"] = auto_vae_precision
        if sd_vae_encode_method is not UNSET:
            field_dict["sd_vae_encode_method"] = sd_vae_encode_method
        if sd_vae_decode_method is not UNSET:
            field_dict["sd_vae_decode_method"] = sd_vae_decode_method
        if inpainting_mask_weight is not UNSET:
            field_dict["inpainting_mask_weight"] = inpainting_mask_weight
        if initial_noise_multiplier is not UNSET:
            field_dict["initial_noise_multiplier"] = initial_noise_multiplier
        if img2img_extra_noise is not UNSET:
            field_dict["img2img_extra_noise"] = img2img_extra_noise
        if img2img_color_correction is not UNSET:
            field_dict["img2img_color_correction"] = img2img_color_correction
        if img2img_fix_steps is not UNSET:
            field_dict["img2img_fix_steps"] = img2img_fix_steps
        if img2img_background_color is not UNSET:
            field_dict["img2img_background_color"] = img2img_background_color
        if img2img_editor_height is not UNSET:
            field_dict["img2img_editor_height"] = img2img_editor_height
        if img2img_sketch_default_brush_color is not UNSET:
            field_dict["img2img_sketch_default_brush_color"] = img2img_sketch_default_brush_color
        if img2img_inpaint_mask_brush_color is not UNSET:
            field_dict["img2img_inpaint_mask_brush_color"] = img2img_inpaint_mask_brush_color
        if img2img_inpaint_sketch_default_brush_color is not UNSET:
            field_dict["img2img_inpaint_sketch_default_brush_color"] = img2img_inpaint_sketch_default_brush_color
        if return_mask is not UNSET:
            field_dict["return_mask"] = return_mask
        if return_mask_composite is not UNSET:
            field_dict["return_mask_composite"] = return_mask_composite
        if img2img_batch_show_results_limit is not UNSET:
            field_dict["img2img_batch_show_results_limit"] = img2img_batch_show_results_limit
        if overlay_inpaint is not UNSET:
            field_dict["overlay_inpaint"] = overlay_inpaint
        if cross_attention_optimization is not UNSET:
            field_dict["cross_attention_optimization"] = cross_attention_optimization
        if s_min_uncond is not UNSET:
            field_dict["s_min_uncond"] = s_min_uncond
        if token_merging_ratio is not UNSET:
            field_dict["token_merging_ratio"] = token_merging_ratio
        if token_merging_ratio_img2img is not UNSET:
            field_dict["token_merging_ratio_img2img"] = token_merging_ratio_img2img
        if token_merging_ratio_hr is not UNSET:
            field_dict["token_merging_ratio_hr"] = token_merging_ratio_hr
        if pad_cond_uncond is not UNSET:
            field_dict["pad_cond_uncond"] = pad_cond_uncond
        if pad_cond_uncond_v0 is not UNSET:
            field_dict["pad_cond_uncond_v0"] = pad_cond_uncond_v0
        if persistent_cond_cache is not UNSET:
            field_dict["persistent_cond_cache"] = persistent_cond_cache
        if batch_cond_uncond is not UNSET:
            field_dict["batch_cond_uncond"] = batch_cond_uncond
        if fp8_storage is not UNSET:
            field_dict["fp8_storage"] = fp8_storage
        if cache_fp16_weight is not UNSET:
            field_dict["cache_fp16_weight"] = cache_fp16_weight
        if auto_backcompat is not UNSET:
            field_dict["auto_backcompat"] = auto_backcompat
        if use_old_emphasis_implementation is not UNSET:
            field_dict["use_old_emphasis_implementation"] = use_old_emphasis_implementation
        if use_old_karras_scheduler_sigmas is not UNSET:
            field_dict["use_old_karras_scheduler_sigmas"] = use_old_karras_scheduler_sigmas
        if no_dpmpp_sde_batch_determinism is not UNSET:
            field_dict["no_dpmpp_sde_batch_determinism"] = no_dpmpp_sde_batch_determinism
        if use_old_hires_fix_width_height is not UNSET:
            field_dict["use_old_hires_fix_width_height"] = use_old_hires_fix_width_height
        if dont_fix_second_order_samplers_schedule is not UNSET:
            field_dict["dont_fix_second_order_samplers_schedule"] = dont_fix_second_order_samplers_schedule
        if hires_fix_use_firstpass_conds is not UNSET:
            field_dict["hires_fix_use_firstpass_conds"] = hires_fix_use_firstpass_conds
        if use_old_scheduling is not UNSET:
            field_dict["use_old_scheduling"] = use_old_scheduling
        if use_downcasted_alpha_bar is not UNSET:
            field_dict["use_downcasted_alpha_bar"] = use_downcasted_alpha_bar
        if interrogate_keep_models_in_memory is not UNSET:
            field_dict["interrogate_keep_models_in_memory"] = interrogate_keep_models_in_memory
        if interrogate_return_ranks is not UNSET:
            field_dict["interrogate_return_ranks"] = interrogate_return_ranks
        if interrogate_clip_num_beams is not UNSET:
            field_dict["interrogate_clip_num_beams"] = interrogate_clip_num_beams
        if interrogate_clip_min_length is not UNSET:
            field_dict["interrogate_clip_min_length"] = interrogate_clip_min_length
        if interrogate_clip_max_length is not UNSET:
            field_dict["interrogate_clip_max_length"] = interrogate_clip_max_length
        if interrogate_clip_dict_limit is not UNSET:
            field_dict["interrogate_clip_dict_limit"] = interrogate_clip_dict_limit
        if interrogate_clip_skip_categories is not UNSET:
            field_dict["interrogate_clip_skip_categories"] = interrogate_clip_skip_categories
        if interrogate_deepbooru_score_threshold is not UNSET:
            field_dict["interrogate_deepbooru_score_threshold"] = interrogate_deepbooru_score_threshold
        if deepbooru_sort_alpha is not UNSET:
            field_dict["deepbooru_sort_alpha"] = deepbooru_sort_alpha
        if deepbooru_use_spaces is not UNSET:
            field_dict["deepbooru_use_spaces"] = deepbooru_use_spaces
        if deepbooru_escape is not UNSET:
            field_dict["deepbooru_escape"] = deepbooru_escape
        if deepbooru_filter_tags is not UNSET:
            field_dict["deepbooru_filter_tags"] = deepbooru_filter_tags
        if extra_networks_show_hidden_directories is not UNSET:
            field_dict["extra_networks_show_hidden_directories"] = extra_networks_show_hidden_directories
        if extra_networks_dir_button_function is not UNSET:
            field_dict["extra_networks_dir_button_function"] = extra_networks_dir_button_function
        if extra_networks_hidden_models is not UNSET:
            field_dict["extra_networks_hidden_models"] = extra_networks_hidden_models
        if extra_networks_default_multiplier is not UNSET:
            field_dict["extra_networks_default_multiplier"] = extra_networks_default_multiplier
        if extra_networks_card_width is not UNSET:
            field_dict["extra_networks_card_width"] = extra_networks_card_width
        if extra_networks_card_height is not UNSET:
            field_dict["extra_networks_card_height"] = extra_networks_card_height
        if extra_networks_card_text_scale is not UNSET:
            field_dict["extra_networks_card_text_scale"] = extra_networks_card_text_scale
        if extra_networks_card_show_desc is not UNSET:
            field_dict["extra_networks_card_show_desc"] = extra_networks_card_show_desc
        if extra_networks_card_description_is_html is not UNSET:
            field_dict["extra_networks_card_description_is_html"] = extra_networks_card_description_is_html
        if extra_networks_card_order_field is not UNSET:
            field_dict["extra_networks_card_order_field"] = extra_networks_card_order_field
        if extra_networks_card_order is not UNSET:
            field_dict["extra_networks_card_order"] = extra_networks_card_order
        if extra_networks_tree_view_default_enabled is not UNSET:
            field_dict["extra_networks_tree_view_default_enabled"] = extra_networks_tree_view_default_enabled
        if extra_networks_add_text_separator is not UNSET:
            field_dict["extra_networks_add_text_separator"] = extra_networks_add_text_separator
        if ui_extra_networks_tab_reorder is not UNSET:
            field_dict["ui_extra_networks_tab_reorder"] = ui_extra_networks_tab_reorder
        if textual_inversion_print_at_load is not UNSET:
            field_dict["textual_inversion_print_at_load"] = textual_inversion_print_at_load
        if textual_inversion_add_hashes_to_infotext is not UNSET:
            field_dict["textual_inversion_add_hashes_to_infotext"] = textual_inversion_add_hashes_to_infotext
        if sd_hypernetwork is not UNSET:
            field_dict["sd_hypernetwork"] = sd_hypernetwork
        if keyedit_precision_attention is not UNSET:
            field_dict["keyedit_precision_attention"] = keyedit_precision_attention
        if keyedit_precision_extra is not UNSET:
            field_dict["keyedit_precision_extra"] = keyedit_precision_extra
        if keyedit_delimiters is not UNSET:
            field_dict["keyedit_delimiters"] = keyedit_delimiters
        if keyedit_delimiters_whitespace is not UNSET:
            field_dict["keyedit_delimiters_whitespace"] = keyedit_delimiters_whitespace
        if keyedit_move is not UNSET:
            field_dict["keyedit_move"] = keyedit_move
        if disable_token_counters is not UNSET:
            field_dict["disable_token_counters"] = disable_token_counters
        if include_styles_into_token_counters is not UNSET:
            field_dict["include_styles_into_token_counters"] = include_styles_into_token_counters
        if return_grid is not UNSET:
            field_dict["return_grid"] = return_grid
        if do_not_show_images is not UNSET:
            field_dict["do_not_show_images"] = do_not_show_images
        if js_modal_lightbox is not UNSET:
            field_dict["js_modal_lightbox"] = js_modal_lightbox
        if js_modal_lightbox_initially_zoomed is not UNSET:
            field_dict["js_modal_lightbox_initially_zoomed"] = js_modal_lightbox_initially_zoomed
        if js_modal_lightbox_gamepad is not UNSET:
            field_dict["js_modal_lightbox_gamepad"] = js_modal_lightbox_gamepad
        if js_modal_lightbox_gamepad_repeat is not UNSET:
            field_dict["js_modal_lightbox_gamepad_repeat"] = js_modal_lightbox_gamepad_repeat
        if sd_webui_modal_lightbox_icon_opacity is not UNSET:
            field_dict["sd_webui_modal_lightbox_icon_opacity"] = sd_webui_modal_lightbox_icon_opacity
        if sd_webui_modal_lightbox_toolbar_opacity is not UNSET:
            field_dict["sd_webui_modal_lightbox_toolbar_opacity"] = sd_webui_modal_lightbox_toolbar_opacity
        if gallery_height is not UNSET:
            field_dict["gallery_height"] = gallery_height
        if open_dir_button_choice is not UNSET:
            field_dict["open_dir_button_choice"] = open_dir_button_choice
        if compact_prompt_box is not UNSET:
            field_dict["compact_prompt_box"] = compact_prompt_box
        if samplers_in_dropdown is not UNSET:
            field_dict["samplers_in_dropdown"] = samplers_in_dropdown
        if dimensions_and_batch_together is not UNSET:
            field_dict["dimensions_and_batch_together"] = dimensions_and_batch_together
        if sd_checkpoint_dropdown_use_short is not UNSET:
            field_dict["sd_checkpoint_dropdown_use_short"] = sd_checkpoint_dropdown_use_short
        if hires_fix_show_sampler is not UNSET:
            field_dict["hires_fix_show_sampler"] = hires_fix_show_sampler
        if hires_fix_show_prompts is not UNSET:
            field_dict["hires_fix_show_prompts"] = hires_fix_show_prompts
        if txt2img_settings_accordion is not UNSET:
            field_dict["txt2img_settings_accordion"] = txt2img_settings_accordion
        if img2img_settings_accordion is not UNSET:
            field_dict["img2img_settings_accordion"] = img2img_settings_accordion
        if interrupt_after_current is not UNSET:
            field_dict["interrupt_after_current"] = interrupt_after_current
        if localization is not UNSET:
            field_dict["localization"] = localization
        if quicksettings_list is not UNSET:
            field_dict["quicksettings_list"] = quicksettings_list
        if ui_tab_order is not UNSET:
            field_dict["ui_tab_order"] = ui_tab_order
        if hidden_tabs is not UNSET:
            field_dict["hidden_tabs"] = hidden_tabs
        if ui_reorder_list is not UNSET:
            field_dict["ui_reorder_list"] = ui_reorder_list
        if gradio_theme is not UNSET:
            field_dict["gradio_theme"] = gradio_theme
        if gradio_themes_cache is not UNSET:
            field_dict["gradio_themes_cache"] = gradio_themes_cache
        if show_progress_in_title is not UNSET:
            field_dict["show_progress_in_title"] = show_progress_in_title
        if send_seed is not UNSET:
            field_dict["send_seed"] = send_seed
        if send_size is not UNSET:
            field_dict["send_size"] = send_size
        if infotext_explanation is not UNSET:
            field_dict["infotext_explanation"] = infotext_explanation
        if enable_pnginfo is not UNSET:
            field_dict["enable_pnginfo"] = enable_pnginfo
        if save_txt is not UNSET:
            field_dict["save_txt"] = save_txt
        if add_model_name_to_info is not UNSET:
            field_dict["add_model_name_to_info"] = add_model_name_to_info
        if add_model_hash_to_info is not UNSET:
            field_dict["add_model_hash_to_info"] = add_model_hash_to_info
        if add_vae_name_to_info is not UNSET:
            field_dict["add_vae_name_to_info"] = add_vae_name_to_info
        if add_vae_hash_to_info is not UNSET:
            field_dict["add_vae_hash_to_info"] = add_vae_hash_to_info
        if add_user_name_to_info is not UNSET:
            field_dict["add_user_name_to_info"] = add_user_name_to_info
        if add_version_to_infotext is not UNSET:
            field_dict["add_version_to_infotext"] = add_version_to_infotext
        if disable_weights_auto_swap is not UNSET:
            field_dict["disable_weights_auto_swap"] = disable_weights_auto_swap
        if infotext_skip_pasting is not UNSET:
            field_dict["infotext_skip_pasting"] = infotext_skip_pasting
        if infotext_styles is not UNSET:
            field_dict["infotext_styles"] = infotext_styles
        if show_progressbar is not UNSET:
            field_dict["show_progressbar"] = show_progressbar
        if live_previews_enable is not UNSET:
            field_dict["live_previews_enable"] = live_previews_enable
        if live_previews_image_format is not UNSET:
            field_dict["live_previews_image_format"] = live_previews_image_format
        if show_progress_grid is not UNSET:
            field_dict["show_progress_grid"] = show_progress_grid
        if show_progress_every_n_steps is not UNSET:
            field_dict["show_progress_every_n_steps"] = show_progress_every_n_steps
        if show_progress_type is not UNSET:
            field_dict["show_progress_type"] = show_progress_type
        if live_preview_allow_lowvram_full is not UNSET:
            field_dict["live_preview_allow_lowvram_full"] = live_preview_allow_lowvram_full
        if live_preview_content is not UNSET:
            field_dict["live_preview_content"] = live_preview_content
        if live_preview_refresh_period is not UNSET:
            field_dict["live_preview_refresh_period"] = live_preview_refresh_period
        if live_preview_fast_interrupt is not UNSET:
            field_dict["live_preview_fast_interrupt"] = live_preview_fast_interrupt
        if js_live_preview_in_modal_lightbox is not UNSET:
            field_dict["js_live_preview_in_modal_lightbox"] = js_live_preview_in_modal_lightbox
        if hide_samplers is not UNSET:
            field_dict["hide_samplers"] = hide_samplers
        if eta_ddim is not UNSET:
            field_dict["eta_ddim"] = eta_ddim
        if eta_ancestral is not UNSET:
            field_dict["eta_ancestral"] = eta_ancestral
        if ddim_discretize is not UNSET:
            field_dict["ddim_discretize"] = ddim_discretize
        if s_churn is not UNSET:
            field_dict["s_churn"] = s_churn
        if s_tmin is not UNSET:
            field_dict["s_tmin"] = s_tmin
        if s_tmax is not UNSET:
            field_dict["s_tmax"] = s_tmax
        if s_noise is not UNSET:
            field_dict["s_noise"] = s_noise
        if k_sched_type is not UNSET:
            field_dict["k_sched_type"] = k_sched_type
        if sigma_min is not UNSET:
            field_dict["sigma_min"] = sigma_min
        if sigma_max is not UNSET:
            field_dict["sigma_max"] = sigma_max
        if rho is not UNSET:
            field_dict["rho"] = rho
        if eta_noise_seed_delta is not UNSET:
            field_dict["eta_noise_seed_delta"] = eta_noise_seed_delta
        if always_discard_next_to_last_sigma is not UNSET:
            field_dict["always_discard_next_to_last_sigma"] = always_discard_next_to_last_sigma
        if sgm_noise_multiplier is not UNSET:
            field_dict["sgm_noise_multiplier"] = sgm_noise_multiplier
        if uni_pc_variant is not UNSET:
            field_dict["uni_pc_variant"] = uni_pc_variant
        if uni_pc_skip_type is not UNSET:
            field_dict["uni_pc_skip_type"] = uni_pc_skip_type
        if uni_pc_order is not UNSET:
            field_dict["uni_pc_order"] = uni_pc_order
        if uni_pc_lower_order_final is not UNSET:
            field_dict["uni_pc_lower_order_final"] = uni_pc_lower_order_final
        if sd_noise_schedule is not UNSET:
            field_dict["sd_noise_schedule"] = sd_noise_schedule
        if postprocessing_enable_in_main_ui is not UNSET:
            field_dict["postprocessing_enable_in_main_ui"] = postprocessing_enable_in_main_ui
        if postprocessing_operation_order is not UNSET:
            field_dict["postprocessing_operation_order"] = postprocessing_operation_order
        if upscaling_max_images_in_cache is not UNSET:
            field_dict["upscaling_max_images_in_cache"] = upscaling_max_images_in_cache
        if postprocessing_existing_caption_action is not UNSET:
            field_dict["postprocessing_existing_caption_action"] = postprocessing_existing_caption_action
        if disabled_extensions is not UNSET:
            field_dict["disabled_extensions"] = disabled_extensions
        if disable_all_extensions is not UNSET:
            field_dict["disable_all_extensions"] = disable_all_extensions
        if restore_config_state_file is not UNSET:
            field_dict["restore_config_state_file"] = restore_config_state_file
        if sd_checkpoint_hash is not UNSET:
            field_dict["sd_checkpoint_hash"] = sd_checkpoint_hash
        if sd_lora is not UNSET:
            field_dict["sd_lora"] = sd_lora
        if lora_preferred_name is not UNSET:
            field_dict["lora_preferred_name"] = lora_preferred_name
        if lora_add_hashes_to_infotext is not UNSET:
            field_dict["lora_add_hashes_to_infotext"] = lora_add_hashes_to_infotext
        if lora_show_all is not UNSET:
            field_dict["lora_show_all"] = lora_show_all
        if lora_hide_unknown_for_versions is not UNSET:
            field_dict["lora_hide_unknown_for_versions"] = lora_hide_unknown_for_versions
        if lora_in_memory_limit is not UNSET:
            field_dict["lora_in_memory_limit"] = lora_in_memory_limit
        if lora_not_found_warning_console is not UNSET:
            field_dict["lora_not_found_warning_console"] = lora_not_found_warning_console
        if lora_not_found_gradio_warning is not UNSET:
            field_dict["lora_not_found_gradio_warning"] = lora_not_found_gradio_warning
        if lora_functional is not UNSET:
            field_dict["lora_functional"] = lora_functional
        if canvas_hotkey_zoom is not UNSET:
            field_dict["canvas_hotkey_zoom"] = canvas_hotkey_zoom
        if canvas_hotkey_adjust is not UNSET:
            field_dict["canvas_hotkey_adjust"] = canvas_hotkey_adjust
        if canvas_hotkey_shrink_brush is not UNSET:
            field_dict["canvas_hotkey_shrink_brush"] = canvas_hotkey_shrink_brush
        if canvas_hotkey_grow_brush is not UNSET:
            field_dict["canvas_hotkey_grow_brush"] = canvas_hotkey_grow_brush
        if canvas_hotkey_move is not UNSET:
            field_dict["canvas_hotkey_move"] = canvas_hotkey_move
        if canvas_hotkey_fullscreen is not UNSET:
            field_dict["canvas_hotkey_fullscreen"] = canvas_hotkey_fullscreen
        if canvas_hotkey_reset is not UNSET:
            field_dict["canvas_hotkey_reset"] = canvas_hotkey_reset
        if canvas_hotkey_overlap is not UNSET:
            field_dict["canvas_hotkey_overlap"] = canvas_hotkey_overlap
        if canvas_show_tooltip is not UNSET:
            field_dict["canvas_show_tooltip"] = canvas_show_tooltip
        if canvas_auto_expand is not UNSET:
            field_dict["canvas_auto_expand"] = canvas_auto_expand
        if canvas_blur_prompt is not UNSET:
            field_dict["canvas_blur_prompt"] = canvas_blur_prompt
        if canvas_disabled_functions is not UNSET:
            field_dict["canvas_disabled_functions"] = canvas_disabled_functions
        if settings_in_ui is not UNSET:
            field_dict["settings_in_ui"] = settings_in_ui
        if extra_options_txt2img is not UNSET:
            field_dict["extra_options_txt2img"] = extra_options_txt2img
        if extra_options_img2img is not UNSET:
            field_dict["extra_options_img2img"] = extra_options_img2img
        if extra_options_cols is not UNSET:
            field_dict["extra_options_cols"] = extra_options_cols
        if extra_options_accordion is not UNSET:
            field_dict["extra_options_accordion"] = extra_options_accordion

        return field_dict

    @classmethod
    def from_dict(cls: Type[T], src_dict: Dict[str, Any]) -> T:
        d = src_dict.copy()
        samples_save = d.pop("samples_save", UNSET)

        samples_format = d.pop("samples_format", UNSET)

        samples_filename_pattern = d.pop("samples_filename_pattern", UNSET)

        save_images_add_number = d.pop("save_images_add_number", UNSET)

        save_images_replace_action = d.pop("save_images_replace_action", UNSET)

        grid_save = d.pop("grid_save", UNSET)

        grid_format = d.pop("grid_format", UNSET)

        grid_extended_filename = d.pop("grid_extended_filename", UNSET)

        grid_only_if_multiple = d.pop("grid_only_if_multiple", UNSET)

        grid_prevent_empty_spots = d.pop("grid_prevent_empty_spots", UNSET)

        grid_zip_filename_pattern = d.pop("grid_zip_filename_pattern", UNSET)

        n_rows = d.pop("n_rows", UNSET)

        font = d.pop("font", UNSET)

        grid_text_active_color = d.pop("grid_text_active_color", UNSET)

        grid_text_inactive_color = d.pop("grid_text_inactive_color", UNSET)

        grid_background_color = d.pop("grid_background_color", UNSET)

        save_images_before_face_restoration = d.pop("save_images_before_face_restoration", UNSET)

        save_images_before_highres_fix = d.pop("save_images_before_highres_fix", UNSET)

        save_images_before_color_correction = d.pop("save_images_before_color_correction", UNSET)

        save_mask = d.pop("save_mask", UNSET)

        save_mask_composite = d.pop("save_mask_composite", UNSET)

        jpeg_quality = d.pop("jpeg_quality", UNSET)

        webp_lossless = d.pop("webp_lossless", UNSET)

        export_for_4chan = d.pop("export_for_4chan", UNSET)

        img_downscale_threshold = d.pop("img_downscale_threshold", UNSET)

        target_side_length = d.pop("target_side_length", UNSET)

        img_max_size_mp = d.pop("img_max_size_mp", UNSET)

        use_original_name_batch = d.pop("use_original_name_batch", UNSET)

        use_upscaler_name_as_suffix = d.pop("use_upscaler_name_as_suffix", UNSET)

        save_selected_only = d.pop("save_selected_only", UNSET)

        save_init_img = d.pop("save_init_img", UNSET)

        temp_dir = d.pop("temp_dir", UNSET)

        clean_temp_dir_at_start = d.pop("clean_temp_dir_at_start", UNSET)

        save_incomplete_images = d.pop("save_incomplete_images", UNSET)

        notification_audio = d.pop("notification_audio", UNSET)

        notification_volume = d.pop("notification_volume", UNSET)

        outdir_samples = d.pop("outdir_samples", UNSET)

        outdir_txt2img_samples = d.pop("outdir_txt2img_samples", UNSET)

        outdir_img2img_samples = d.pop("outdir_img2img_samples", UNSET)

        outdir_extras_samples = d.pop("outdir_extras_samples", UNSET)

        outdir_grids = d.pop("outdir_grids", UNSET)

        outdir_txt2img_grids = d.pop("outdir_txt2img_grids", UNSET)

        outdir_img2img_grids = d.pop("outdir_img2img_grids", UNSET)

        outdir_save = d.pop("outdir_save", UNSET)

        outdir_init_images = d.pop("outdir_init_images", UNSET)

        save_to_dirs = d.pop("save_to_dirs", UNSET)

        grid_save_to_dirs = d.pop("grid_save_to_dirs", UNSET)

        use_save_to_dirs_for_ui = d.pop("use_save_to_dirs_for_ui", UNSET)

        directories_filename_pattern = d.pop("directories_filename_pattern", UNSET)

        directories_max_prompt_words = d.pop("directories_max_prompt_words", UNSET)

        esrgan_tile = d.pop("ESRGAN_tile", UNSET)

        esrgan_tile_overlap = d.pop("ESRGAN_tile_overlap", UNSET)

        realesrgan_enabled_models = cast(List[Any], d.pop("realesrgan_enabled_models", UNSET))

        dat_enabled_models = cast(List[Any], d.pop("dat_enabled_models", UNSET))

        dat_tile = d.pop("DAT_tile", UNSET)

        dat_tile_overlap = d.pop("DAT_tile_overlap", UNSET)

        upscaler_for_img2img = d.pop("upscaler_for_img2img", UNSET)

        face_restoration = d.pop("face_restoration", UNSET)

        face_restoration_model = d.pop("face_restoration_model", UNSET)

        code_former_weight = d.pop("code_former_weight", UNSET)

        face_restoration_unload = d.pop("face_restoration_unload", UNSET)

        auto_launch_browser = d.pop("auto_launch_browser", UNSET)

        enable_console_prompts = d.pop("enable_console_prompts", UNSET)

        show_warnings = d.pop("show_warnings", UNSET)

        show_gradio_deprecation_warnings = d.pop("show_gradio_deprecation_warnings", UNSET)

        memmon_poll_rate = d.pop("memmon_poll_rate", UNSET)

        samples_log_stdout = d.pop("samples_log_stdout", UNSET)

        multiple_tqdm = d.pop("multiple_tqdm", UNSET)

        enable_upscale_progressbar = d.pop("enable_upscale_progressbar", UNSET)

        print_hypernet_extra = d.pop("print_hypernet_extra", UNSET)

        list_hidden_files = d.pop("list_hidden_files", UNSET)

        disable_mmap_load_safetensors = d.pop("disable_mmap_load_safetensors", UNSET)

        hide_ldm_prints = d.pop("hide_ldm_prints", UNSET)

        dump_stacks_on_signal = d.pop("dump_stacks_on_signal", UNSET)

        api_enable_requests = d.pop("api_enable_requests", UNSET)

        api_forbid_local_requests = d.pop("api_forbid_local_requests", UNSET)

        api_useragent = d.pop("api_useragent", UNSET)

        unload_models_when_training = d.pop("unload_models_when_training", UNSET)

        pin_memory = d.pop("pin_memory", UNSET)

        save_optimizer_state = d.pop("save_optimizer_state", UNSET)

        save_training_settings_to_txt = d.pop("save_training_settings_to_txt", UNSET)

        dataset_filename_word_regex = d.pop("dataset_filename_word_regex", UNSET)

        dataset_filename_join_string = d.pop("dataset_filename_join_string", UNSET)

        training_image_repeats_per_epoch = d.pop("training_image_repeats_per_epoch", UNSET)

        training_write_csv_every = d.pop("training_write_csv_every", UNSET)

        training_xattention_optimizations = d.pop("training_xattention_optimizations", UNSET)

        training_enable_tensorboard = d.pop("training_enable_tensorboard", UNSET)

        training_tensorboard_save_images = d.pop("training_tensorboard_save_images", UNSET)

        training_tensorboard_flush_every = d.pop("training_tensorboard_flush_every", UNSET)

        sd_model_checkpoint = d.pop("sd_model_checkpoint", UNSET)

        sd_checkpoints_limit = d.pop("sd_checkpoints_limit", UNSET)

        sd_checkpoints_keep_in_cpu = d.pop("sd_checkpoints_keep_in_cpu", UNSET)

        sd_checkpoint_cache = d.pop("sd_checkpoint_cache", UNSET)

        sd_unet = d.pop("sd_unet", UNSET)

        enable_quantization = d.pop("enable_quantization", UNSET)

        emphasis = d.pop("emphasis", UNSET)

        enable_batch_seeds = d.pop("enable_batch_seeds", UNSET)

        comma_padding_backtrack = d.pop("comma_padding_backtrack", UNSET)

        clip_stop_at_last_layers = d.pop("CLIP_stop_at_last_layers", UNSET)

        upcast_attn = d.pop("upcast_attn", UNSET)

        randn_source = d.pop("randn_source", UNSET)

        tiling = d.pop("tiling", UNSET)

        hires_fix_refiner_pass = d.pop("hires_fix_refiner_pass", UNSET)

        sdxl_crop_top = d.pop("sdxl_crop_top", UNSET)

        sdxl_crop_left = d.pop("sdxl_crop_left", UNSET)

        sdxl_refiner_low_aesthetic_score = d.pop("sdxl_refiner_low_aesthetic_score", UNSET)

        sdxl_refiner_high_aesthetic_score = d.pop("sdxl_refiner_high_aesthetic_score", UNSET)

        sd_vae_explanation = d.pop("sd_vae_explanation", UNSET)

        sd_vae_checkpoint_cache = d.pop("sd_vae_checkpoint_cache", UNSET)

        sd_vae = d.pop("sd_vae", UNSET)

        sd_vae_overrides_per_model_preferences = d.pop("sd_vae_overrides_per_model_preferences", UNSET)

        auto_vae_precision_bfloat16 = d.pop("auto_vae_precision_bfloat16", UNSET)

        auto_vae_precision = d.pop("auto_vae_precision", UNSET)

        sd_vae_encode_method = d.pop("sd_vae_encode_method", UNSET)

        sd_vae_decode_method = d.pop("sd_vae_decode_method", UNSET)

        inpainting_mask_weight = d.pop("inpainting_mask_weight", UNSET)

        initial_noise_multiplier = d.pop("initial_noise_multiplier", UNSET)

        img2img_extra_noise = d.pop("img2img_extra_noise", UNSET)

        img2img_color_correction = d.pop("img2img_color_correction", UNSET)

        img2img_fix_steps = d.pop("img2img_fix_steps", UNSET)

        img2img_background_color = d.pop("img2img_background_color", UNSET)

        img2img_editor_height = d.pop("img2img_editor_height", UNSET)

        img2img_sketch_default_brush_color = d.pop("img2img_sketch_default_brush_color", UNSET)

        img2img_inpaint_mask_brush_color = d.pop("img2img_inpaint_mask_brush_color", UNSET)

        img2img_inpaint_sketch_default_brush_color = d.pop("img2img_inpaint_sketch_default_brush_color", UNSET)

        return_mask = d.pop("return_mask", UNSET)

        return_mask_composite = d.pop("return_mask_composite", UNSET)

        img2img_batch_show_results_limit = d.pop("img2img_batch_show_results_limit", UNSET)

        overlay_inpaint = d.pop("overlay_inpaint", UNSET)

        cross_attention_optimization = d.pop("cross_attention_optimization", UNSET)

        s_min_uncond = d.pop("s_min_uncond", UNSET)

        token_merging_ratio = d.pop("token_merging_ratio", UNSET)

        token_merging_ratio_img2img = d.pop("token_merging_ratio_img2img", UNSET)

        token_merging_ratio_hr = d.pop("token_merging_ratio_hr", UNSET)

        pad_cond_uncond = d.pop("pad_cond_uncond", UNSET)

        pad_cond_uncond_v0 = d.pop("pad_cond_uncond_v0", UNSET)

        persistent_cond_cache = d.pop("persistent_cond_cache", UNSET)

        batch_cond_uncond = d.pop("batch_cond_uncond", UNSET)

        fp8_storage = d.pop("fp8_storage", UNSET)

        cache_fp16_weight = d.pop("cache_fp16_weight", UNSET)

        auto_backcompat = d.pop("auto_backcompat", UNSET)

        use_old_emphasis_implementation = d.pop("use_old_emphasis_implementation", UNSET)

        use_old_karras_scheduler_sigmas = d.pop("use_old_karras_scheduler_sigmas", UNSET)

        no_dpmpp_sde_batch_determinism = d.pop("no_dpmpp_sde_batch_determinism", UNSET)

        use_old_hires_fix_width_height = d.pop("use_old_hires_fix_width_height", UNSET)

        dont_fix_second_order_samplers_schedule = d.pop("dont_fix_second_order_samplers_schedule", UNSET)

        hires_fix_use_firstpass_conds = d.pop("hires_fix_use_firstpass_conds", UNSET)

        use_old_scheduling = d.pop("use_old_scheduling", UNSET)

        use_downcasted_alpha_bar = d.pop("use_downcasted_alpha_bar", UNSET)

        interrogate_keep_models_in_memory = d.pop("interrogate_keep_models_in_memory", UNSET)

        interrogate_return_ranks = d.pop("interrogate_return_ranks", UNSET)

        interrogate_clip_num_beams = d.pop("interrogate_clip_num_beams", UNSET)

        interrogate_clip_min_length = d.pop("interrogate_clip_min_length", UNSET)

        interrogate_clip_max_length = d.pop("interrogate_clip_max_length", UNSET)

        interrogate_clip_dict_limit = d.pop("interrogate_clip_dict_limit", UNSET)

        interrogate_clip_skip_categories = d.pop("interrogate_clip_skip_categories", UNSET)

        interrogate_deepbooru_score_threshold = d.pop("interrogate_deepbooru_score_threshold", UNSET)

        deepbooru_sort_alpha = d.pop("deepbooru_sort_alpha", UNSET)

        deepbooru_use_spaces = d.pop("deepbooru_use_spaces", UNSET)

        deepbooru_escape = d.pop("deepbooru_escape", UNSET)

        deepbooru_filter_tags = d.pop("deepbooru_filter_tags", UNSET)

        extra_networks_show_hidden_directories = d.pop("extra_networks_show_hidden_directories", UNSET)

        extra_networks_dir_button_function = d.pop("extra_networks_dir_button_function", UNSET)

        extra_networks_hidden_models = d.pop("extra_networks_hidden_models", UNSET)

        extra_networks_default_multiplier = d.pop("extra_networks_default_multiplier", UNSET)

        extra_networks_card_width = d.pop("extra_networks_card_width", UNSET)

        extra_networks_card_height = d.pop("extra_networks_card_height", UNSET)

        extra_networks_card_text_scale = d.pop("extra_networks_card_text_scale", UNSET)

        extra_networks_card_show_desc = d.pop("extra_networks_card_show_desc", UNSET)

        extra_networks_card_description_is_html = d.pop("extra_networks_card_description_is_html", UNSET)

        extra_networks_card_order_field = d.pop("extra_networks_card_order_field", UNSET)

        extra_networks_card_order = d.pop("extra_networks_card_order", UNSET)

        extra_networks_tree_view_default_enabled = d.pop("extra_networks_tree_view_default_enabled", UNSET)

        extra_networks_add_text_separator = d.pop("extra_networks_add_text_separator", UNSET)

        ui_extra_networks_tab_reorder = d.pop("ui_extra_networks_tab_reorder", UNSET)

        textual_inversion_print_at_load = d.pop("textual_inversion_print_at_load", UNSET)

        textual_inversion_add_hashes_to_infotext = d.pop("textual_inversion_add_hashes_to_infotext", UNSET)

        sd_hypernetwork = d.pop("sd_hypernetwork", UNSET)

        keyedit_precision_attention = d.pop("keyedit_precision_attention", UNSET)

        keyedit_precision_extra = d.pop("keyedit_precision_extra", UNSET)

        keyedit_delimiters = d.pop("keyedit_delimiters", UNSET)

        keyedit_delimiters_whitespace = cast(List[Any], d.pop("keyedit_delimiters_whitespace", UNSET))

        keyedit_move = d.pop("keyedit_move", UNSET)

        disable_token_counters = d.pop("disable_token_counters", UNSET)

        include_styles_into_token_counters = d.pop("include_styles_into_token_counters", UNSET)

        return_grid = d.pop("return_grid", UNSET)

        do_not_show_images = d.pop("do_not_show_images", UNSET)

        js_modal_lightbox = d.pop("js_modal_lightbox", UNSET)

        js_modal_lightbox_initially_zoomed = d.pop("js_modal_lightbox_initially_zoomed", UNSET)

        js_modal_lightbox_gamepad = d.pop("js_modal_lightbox_gamepad", UNSET)

        js_modal_lightbox_gamepad_repeat = d.pop("js_modal_lightbox_gamepad_repeat", UNSET)

        sd_webui_modal_lightbox_icon_opacity = d.pop("sd_webui_modal_lightbox_icon_opacity", UNSET)

        sd_webui_modal_lightbox_toolbar_opacity = d.pop("sd_webui_modal_lightbox_toolbar_opacity", UNSET)

        gallery_height = d.pop("gallery_height", UNSET)

        open_dir_button_choice = d.pop("open_dir_button_choice", UNSET)

        compact_prompt_box = d.pop("compact_prompt_box", UNSET)

        samplers_in_dropdown = d.pop("samplers_in_dropdown", UNSET)

        dimensions_and_batch_together = d.pop("dimensions_and_batch_together", UNSET)

        sd_checkpoint_dropdown_use_short = d.pop("sd_checkpoint_dropdown_use_short", UNSET)

        hires_fix_show_sampler = d.pop("hires_fix_show_sampler", UNSET)

        hires_fix_show_prompts = d.pop("hires_fix_show_prompts", UNSET)

        txt2img_settings_accordion = d.pop("txt2img_settings_accordion", UNSET)

        img2img_settings_accordion = d.pop("img2img_settings_accordion", UNSET)

        interrupt_after_current = d.pop("interrupt_after_current", UNSET)

        localization = d.pop("localization", UNSET)

        quicksettings_list = cast(List[Any], d.pop("quicksettings_list", UNSET))

        ui_tab_order = d.pop("ui_tab_order", UNSET)

        hidden_tabs = d.pop("hidden_tabs", UNSET)

        ui_reorder_list = d.pop("ui_reorder_list", UNSET)

        gradio_theme = d.pop("gradio_theme", UNSET)

        gradio_themes_cache = d.pop("gradio_themes_cache", UNSET)

        show_progress_in_title = d.pop("show_progress_in_title", UNSET)

        send_seed = d.pop("send_seed", UNSET)

        send_size = d.pop("send_size", UNSET)

        infotext_explanation = d.pop("infotext_explanation", UNSET)

        enable_pnginfo = d.pop("enable_pnginfo", UNSET)

        save_txt = d.pop("save_txt", UNSET)

        add_model_name_to_info = d.pop("add_model_name_to_info", UNSET)

        add_model_hash_to_info = d.pop("add_model_hash_to_info", UNSET)

        add_vae_name_to_info = d.pop("add_vae_name_to_info", UNSET)

        add_vae_hash_to_info = d.pop("add_vae_hash_to_info", UNSET)

        add_user_name_to_info = d.pop("add_user_name_to_info", UNSET)

        add_version_to_infotext = d.pop("add_version_to_infotext", UNSET)

        disable_weights_auto_swap = d.pop("disable_weights_auto_swap", UNSET)

        infotext_skip_pasting = d.pop("infotext_skip_pasting", UNSET)

        infotext_styles = d.pop("infotext_styles", UNSET)

        show_progressbar = d.pop("show_progressbar", UNSET)

        live_previews_enable = d.pop("live_previews_enable", UNSET)

        live_previews_image_format = d.pop("live_previews_image_format", UNSET)

        show_progress_grid = d.pop("show_progress_grid", UNSET)

        show_progress_every_n_steps = d.pop("show_progress_every_n_steps", UNSET)

        show_progress_type = d.pop("show_progress_type", UNSET)

        live_preview_allow_lowvram_full = d.pop("live_preview_allow_lowvram_full", UNSET)

        live_preview_content = d.pop("live_preview_content", UNSET)

        live_preview_refresh_period = d.pop("live_preview_refresh_period", UNSET)

        live_preview_fast_interrupt = d.pop("live_preview_fast_interrupt", UNSET)

        js_live_preview_in_modal_lightbox = d.pop("js_live_preview_in_modal_lightbox", UNSET)

        hide_samplers = d.pop("hide_samplers", UNSET)

        eta_ddim = d.pop("eta_ddim", UNSET)

        eta_ancestral = d.pop("eta_ancestral", UNSET)

        ddim_discretize = d.pop("ddim_discretize", UNSET)

        s_churn = d.pop("s_churn", UNSET)

        s_tmin = d.pop("s_tmin", UNSET)

        s_tmax = d.pop("s_tmax", UNSET)

        s_noise = d.pop("s_noise", UNSET)

        k_sched_type = d.pop("k_sched_type", UNSET)

        sigma_min = d.pop("sigma_min", UNSET)

        sigma_max = d.pop("sigma_max", UNSET)

        rho = d.pop("rho", UNSET)

        eta_noise_seed_delta = d.pop("eta_noise_seed_delta", UNSET)

        always_discard_next_to_last_sigma = d.pop("always_discard_next_to_last_sigma", UNSET)

        sgm_noise_multiplier = d.pop("sgm_noise_multiplier", UNSET)

        uni_pc_variant = d.pop("uni_pc_variant", UNSET)

        uni_pc_skip_type = d.pop("uni_pc_skip_type", UNSET)

        uni_pc_order = d.pop("uni_pc_order", UNSET)

        uni_pc_lower_order_final = d.pop("uni_pc_lower_order_final", UNSET)

        sd_noise_schedule = d.pop("sd_noise_schedule", UNSET)

        postprocessing_enable_in_main_ui = d.pop("postprocessing_enable_in_main_ui", UNSET)

        postprocessing_operation_order = d.pop("postprocessing_operation_order", UNSET)

        upscaling_max_images_in_cache = d.pop("upscaling_max_images_in_cache", UNSET)

        postprocessing_existing_caption_action = d.pop("postprocessing_existing_caption_action", UNSET)

        disabled_extensions = d.pop("disabled_extensions", UNSET)

        disable_all_extensions = d.pop("disable_all_extensions", UNSET)

        restore_config_state_file = d.pop("restore_config_state_file", UNSET)

        sd_checkpoint_hash = d.pop("sd_checkpoint_hash", UNSET)

        sd_lora = d.pop("sd_lora", UNSET)

        lora_preferred_name = d.pop("lora_preferred_name", UNSET)

        lora_add_hashes_to_infotext = d.pop("lora_add_hashes_to_infotext", UNSET)

        lora_show_all = d.pop("lora_show_all", UNSET)

        lora_hide_unknown_for_versions = d.pop("lora_hide_unknown_for_versions", UNSET)

        lora_in_memory_limit = d.pop("lora_in_memory_limit", UNSET)

        lora_not_found_warning_console = d.pop("lora_not_found_warning_console", UNSET)

        lora_not_found_gradio_warning = d.pop("lora_not_found_gradio_warning", UNSET)

        lora_functional = d.pop("lora_functional", UNSET)

        canvas_hotkey_zoom = d.pop("canvas_hotkey_zoom", UNSET)

        canvas_hotkey_adjust = d.pop("canvas_hotkey_adjust", UNSET)

        canvas_hotkey_shrink_brush = d.pop("canvas_hotkey_shrink_brush", UNSET)

        canvas_hotkey_grow_brush = d.pop("canvas_hotkey_grow_brush", UNSET)

        canvas_hotkey_move = d.pop("canvas_hotkey_move", UNSET)

        canvas_hotkey_fullscreen = d.pop("canvas_hotkey_fullscreen", UNSET)

        canvas_hotkey_reset = d.pop("canvas_hotkey_reset", UNSET)

        canvas_hotkey_overlap = d.pop("canvas_hotkey_overlap", UNSET)

        canvas_show_tooltip = d.pop("canvas_show_tooltip", UNSET)

        canvas_auto_expand = d.pop("canvas_auto_expand", UNSET)

        canvas_blur_prompt = d.pop("canvas_blur_prompt", UNSET)

        canvas_disabled_functions = cast(List[Any], d.pop("canvas_disabled_functions", UNSET))

        settings_in_ui = d.pop("settings_in_ui", UNSET)

        extra_options_txt2img = d.pop("extra_options_txt2img", UNSET)

        extra_options_img2img = d.pop("extra_options_img2img", UNSET)

        extra_options_cols = d.pop("extra_options_cols", UNSET)

        extra_options_accordion = d.pop("extra_options_accordion", UNSET)

        options = cls(
            samples_save=samples_save,
            samples_format=samples_format,
            samples_filename_pattern=samples_filename_pattern,
            save_images_add_number=save_images_add_number,
            save_images_replace_action=save_images_replace_action,
            grid_save=grid_save,
            grid_format=grid_format,
            grid_extended_filename=grid_extended_filename,
            grid_only_if_multiple=grid_only_if_multiple,
            grid_prevent_empty_spots=grid_prevent_empty_spots,
            grid_zip_filename_pattern=grid_zip_filename_pattern,
            n_rows=n_rows,
            font=font,
            grid_text_active_color=grid_text_active_color,
            grid_text_inactive_color=grid_text_inactive_color,
            grid_background_color=grid_background_color,
            save_images_before_face_restoration=save_images_before_face_restoration,
            save_images_before_highres_fix=save_images_before_highres_fix,
            save_images_before_color_correction=save_images_before_color_correction,
            save_mask=save_mask,
            save_mask_composite=save_mask_composite,
            jpeg_quality=jpeg_quality,
            webp_lossless=webp_lossless,
            export_for_4chan=export_for_4chan,
            img_downscale_threshold=img_downscale_threshold,
            target_side_length=target_side_length,
            img_max_size_mp=img_max_size_mp,
            use_original_name_batch=use_original_name_batch,
            use_upscaler_name_as_suffix=use_upscaler_name_as_suffix,
            save_selected_only=save_selected_only,
            save_init_img=save_init_img,
            temp_dir=temp_dir,
            clean_temp_dir_at_start=clean_temp_dir_at_start,
            save_incomplete_images=save_incomplete_images,
            notification_audio=notification_audio,
            notification_volume=notification_volume,
            outdir_samples=outdir_samples,
            outdir_txt2img_samples=outdir_txt2img_samples,
            outdir_img2img_samples=outdir_img2img_samples,
            outdir_extras_samples=outdir_extras_samples,
            outdir_grids=outdir_grids,
            outdir_txt2img_grids=outdir_txt2img_grids,
            outdir_img2img_grids=outdir_img2img_grids,
            outdir_save=outdir_save,
            outdir_init_images=outdir_init_images,
            save_to_dirs=save_to_dirs,
            grid_save_to_dirs=grid_save_to_dirs,
            use_save_to_dirs_for_ui=use_save_to_dirs_for_ui,
            directories_filename_pattern=directories_filename_pattern,
            directories_max_prompt_words=directories_max_prompt_words,
            esrgan_tile=esrgan_tile,
            esrgan_tile_overlap=esrgan_tile_overlap,
            realesrgan_enabled_models=realesrgan_enabled_models,
            dat_enabled_models=dat_enabled_models,
            dat_tile=dat_tile,
            dat_tile_overlap=dat_tile_overlap,
            upscaler_for_img2img=upscaler_for_img2img,
            face_restoration=face_restoration,
            face_restoration_model=face_restoration_model,
            code_former_weight=code_former_weight,
            face_restoration_unload=face_restoration_unload,
            auto_launch_browser=auto_launch_browser,
            enable_console_prompts=enable_console_prompts,
            show_warnings=show_warnings,
            show_gradio_deprecation_warnings=show_gradio_deprecation_warnings,
            memmon_poll_rate=memmon_poll_rate,
            samples_log_stdout=samples_log_stdout,
            multiple_tqdm=multiple_tqdm,
            enable_upscale_progressbar=enable_upscale_progressbar,
            print_hypernet_extra=print_hypernet_extra,
            list_hidden_files=list_hidden_files,
            disable_mmap_load_safetensors=disable_mmap_load_safetensors,
            hide_ldm_prints=hide_ldm_prints,
            dump_stacks_on_signal=dump_stacks_on_signal,
            api_enable_requests=api_enable_requests,
            api_forbid_local_requests=api_forbid_local_requests,
            api_useragent=api_useragent,
            unload_models_when_training=unload_models_when_training,
            pin_memory=pin_memory,
            save_optimizer_state=save_optimizer_state,
            save_training_settings_to_txt=save_training_settings_to_txt,
            dataset_filename_word_regex=dataset_filename_word_regex,
            dataset_filename_join_string=dataset_filename_join_string,
            training_image_repeats_per_epoch=training_image_repeats_per_epoch,
            training_write_csv_every=training_write_csv_every,
            training_xattention_optimizations=training_xattention_optimizations,
            training_enable_tensorboard=training_enable_tensorboard,
            training_tensorboard_save_images=training_tensorboard_save_images,
            training_tensorboard_flush_every=training_tensorboard_flush_every,
            sd_model_checkpoint=sd_model_checkpoint,
            sd_checkpoints_limit=sd_checkpoints_limit,
            sd_checkpoints_keep_in_cpu=sd_checkpoints_keep_in_cpu,
            sd_checkpoint_cache=sd_checkpoint_cache,
            sd_unet=sd_unet,
            enable_quantization=enable_quantization,
            emphasis=emphasis,
            enable_batch_seeds=enable_batch_seeds,
            comma_padding_backtrack=comma_padding_backtrack,
            clip_stop_at_last_layers=clip_stop_at_last_layers,
            upcast_attn=upcast_attn,
            randn_source=randn_source,
            tiling=tiling,
            hires_fix_refiner_pass=hires_fix_refiner_pass,
            sdxl_crop_top=sdxl_crop_top,
            sdxl_crop_left=sdxl_crop_left,
            sdxl_refiner_low_aesthetic_score=sdxl_refiner_low_aesthetic_score,
            sdxl_refiner_high_aesthetic_score=sdxl_refiner_high_aesthetic_score,
            sd_vae_explanation=sd_vae_explanation,
            sd_vae_checkpoint_cache=sd_vae_checkpoint_cache,
            sd_vae=sd_vae,
            sd_vae_overrides_per_model_preferences=sd_vae_overrides_per_model_preferences,
            auto_vae_precision_bfloat16=auto_vae_precision_bfloat16,
            auto_vae_precision=auto_vae_precision,
            sd_vae_encode_method=sd_vae_encode_method,
            sd_vae_decode_method=sd_vae_decode_method,
            inpainting_mask_weight=inpainting_mask_weight,
            initial_noise_multiplier=initial_noise_multiplier,
            img2img_extra_noise=img2img_extra_noise,
            img2img_color_correction=img2img_color_correction,
            img2img_fix_steps=img2img_fix_steps,
            img2img_background_color=img2img_background_color,
            img2img_editor_height=img2img_editor_height,
            img2img_sketch_default_brush_color=img2img_sketch_default_brush_color,
            img2img_inpaint_mask_brush_color=img2img_inpaint_mask_brush_color,
            img2img_inpaint_sketch_default_brush_color=img2img_inpaint_sketch_default_brush_color,
            return_mask=return_mask,
            return_mask_composite=return_mask_composite,
            img2img_batch_show_results_limit=img2img_batch_show_results_limit,
            overlay_inpaint=overlay_inpaint,
            cross_attention_optimization=cross_attention_optimization,
            s_min_uncond=s_min_uncond,
            token_merging_ratio=token_merging_ratio,
            token_merging_ratio_img2img=token_merging_ratio_img2img,
            token_merging_ratio_hr=token_merging_ratio_hr,
            pad_cond_uncond=pad_cond_uncond,
            pad_cond_uncond_v0=pad_cond_uncond_v0,
            persistent_cond_cache=persistent_cond_cache,
            batch_cond_uncond=batch_cond_uncond,
            fp8_storage=fp8_storage,
            cache_fp16_weight=cache_fp16_weight,
            auto_backcompat=auto_backcompat,
            use_old_emphasis_implementation=use_old_emphasis_implementation,
            use_old_karras_scheduler_sigmas=use_old_karras_scheduler_sigmas,
            no_dpmpp_sde_batch_determinism=no_dpmpp_sde_batch_determinism,
            use_old_hires_fix_width_height=use_old_hires_fix_width_height,
            dont_fix_second_order_samplers_schedule=dont_fix_second_order_samplers_schedule,
            hires_fix_use_firstpass_conds=hires_fix_use_firstpass_conds,
            use_old_scheduling=use_old_scheduling,
            use_downcasted_alpha_bar=use_downcasted_alpha_bar,
            interrogate_keep_models_in_memory=interrogate_keep_models_in_memory,
            interrogate_return_ranks=interrogate_return_ranks,
            interrogate_clip_num_beams=interrogate_clip_num_beams,
            interrogate_clip_min_length=interrogate_clip_min_length,
            interrogate_clip_max_length=interrogate_clip_max_length,
            interrogate_clip_dict_limit=interrogate_clip_dict_limit,
            interrogate_clip_skip_categories=interrogate_clip_skip_categories,
            interrogate_deepbooru_score_threshold=interrogate_deepbooru_score_threshold,
            deepbooru_sort_alpha=deepbooru_sort_alpha,
            deepbooru_use_spaces=deepbooru_use_spaces,
            deepbooru_escape=deepbooru_escape,
            deepbooru_filter_tags=deepbooru_filter_tags,
            extra_networks_show_hidden_directories=extra_networks_show_hidden_directories,
            extra_networks_dir_button_function=extra_networks_dir_button_function,
            extra_networks_hidden_models=extra_networks_hidden_models,
            extra_networks_default_multiplier=extra_networks_default_multiplier,
            extra_networks_card_width=extra_networks_card_width,
            extra_networks_card_height=extra_networks_card_height,
            extra_networks_card_text_scale=extra_networks_card_text_scale,
            extra_networks_card_show_desc=extra_networks_card_show_desc,
            extra_networks_card_description_is_html=extra_networks_card_description_is_html,
            extra_networks_card_order_field=extra_networks_card_order_field,
            extra_networks_card_order=extra_networks_card_order,
            extra_networks_tree_view_default_enabled=extra_networks_tree_view_default_enabled,
            extra_networks_add_text_separator=extra_networks_add_text_separator,
            ui_extra_networks_tab_reorder=ui_extra_networks_tab_reorder,
            textual_inversion_print_at_load=textual_inversion_print_at_load,
            textual_inversion_add_hashes_to_infotext=textual_inversion_add_hashes_to_infotext,
            sd_hypernetwork=sd_hypernetwork,
            keyedit_precision_attention=keyedit_precision_attention,
            keyedit_precision_extra=keyedit_precision_extra,
            keyedit_delimiters=keyedit_delimiters,
            keyedit_delimiters_whitespace=keyedit_delimiters_whitespace,
            keyedit_move=keyedit_move,
            disable_token_counters=disable_token_counters,
            include_styles_into_token_counters=include_styles_into_token_counters,
            return_grid=return_grid,
            do_not_show_images=do_not_show_images,
            js_modal_lightbox=js_modal_lightbox,
            js_modal_lightbox_initially_zoomed=js_modal_lightbox_initially_zoomed,
            js_modal_lightbox_gamepad=js_modal_lightbox_gamepad,
            js_modal_lightbox_gamepad_repeat=js_modal_lightbox_gamepad_repeat,
            sd_webui_modal_lightbox_icon_opacity=sd_webui_modal_lightbox_icon_opacity,
            sd_webui_modal_lightbox_toolbar_opacity=sd_webui_modal_lightbox_toolbar_opacity,
            gallery_height=gallery_height,
            open_dir_button_choice=open_dir_button_choice,
            compact_prompt_box=compact_prompt_box,
            samplers_in_dropdown=samplers_in_dropdown,
            dimensions_and_batch_together=dimensions_and_batch_together,
            sd_checkpoint_dropdown_use_short=sd_checkpoint_dropdown_use_short,
            hires_fix_show_sampler=hires_fix_show_sampler,
            hires_fix_show_prompts=hires_fix_show_prompts,
            txt2img_settings_accordion=txt2img_settings_accordion,
            img2img_settings_accordion=img2img_settings_accordion,
            interrupt_after_current=interrupt_after_current,
            localization=localization,
            quicksettings_list=quicksettings_list,
            ui_tab_order=ui_tab_order,
            hidden_tabs=hidden_tabs,
            ui_reorder_list=ui_reorder_list,
            gradio_theme=gradio_theme,
            gradio_themes_cache=gradio_themes_cache,
            show_progress_in_title=show_progress_in_title,
            send_seed=send_seed,
            send_size=send_size,
            infotext_explanation=infotext_explanation,
            enable_pnginfo=enable_pnginfo,
            save_txt=save_txt,
            add_model_name_to_info=add_model_name_to_info,
            add_model_hash_to_info=add_model_hash_to_info,
            add_vae_name_to_info=add_vae_name_to_info,
            add_vae_hash_to_info=add_vae_hash_to_info,
            add_user_name_to_info=add_user_name_to_info,
            add_version_to_infotext=add_version_to_infotext,
            disable_weights_auto_swap=disable_weights_auto_swap,
            infotext_skip_pasting=infotext_skip_pasting,
            infotext_styles=infotext_styles,
            show_progressbar=show_progressbar,
            live_previews_enable=live_previews_enable,
            live_previews_image_format=live_previews_image_format,
            show_progress_grid=show_progress_grid,
            show_progress_every_n_steps=show_progress_every_n_steps,
            show_progress_type=show_progress_type,
            live_preview_allow_lowvram_full=live_preview_allow_lowvram_full,
            live_preview_content=live_preview_content,
            live_preview_refresh_period=live_preview_refresh_period,
            live_preview_fast_interrupt=live_preview_fast_interrupt,
            js_live_preview_in_modal_lightbox=js_live_preview_in_modal_lightbox,
            hide_samplers=hide_samplers,
            eta_ddim=eta_ddim,
            eta_ancestral=eta_ancestral,
            ddim_discretize=ddim_discretize,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
            k_sched_type=k_sched_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            eta_noise_seed_delta=eta_noise_seed_delta,
            always_discard_next_to_last_sigma=always_discard_next_to_last_sigma,
            sgm_noise_multiplier=sgm_noise_multiplier,
            uni_pc_variant=uni_pc_variant,
            uni_pc_skip_type=uni_pc_skip_type,
            uni_pc_order=uni_pc_order,
            uni_pc_lower_order_final=uni_pc_lower_order_final,
            sd_noise_schedule=sd_noise_schedule,
            postprocessing_enable_in_main_ui=postprocessing_enable_in_main_ui,
            postprocessing_operation_order=postprocessing_operation_order,
            upscaling_max_images_in_cache=upscaling_max_images_in_cache,
            postprocessing_existing_caption_action=postprocessing_existing_caption_action,
            disabled_extensions=disabled_extensions,
            disable_all_extensions=disable_all_extensions,
            restore_config_state_file=restore_config_state_file,
            sd_checkpoint_hash=sd_checkpoint_hash,
            sd_lora=sd_lora,
            lora_preferred_name=lora_preferred_name,
            lora_add_hashes_to_infotext=lora_add_hashes_to_infotext,
            lora_show_all=lora_show_all,
            lora_hide_unknown_for_versions=lora_hide_unknown_for_versions,
            lora_in_memory_limit=lora_in_memory_limit,
            lora_not_found_warning_console=lora_not_found_warning_console,
            lora_not_found_gradio_warning=lora_not_found_gradio_warning,
            lora_functional=lora_functional,
            canvas_hotkey_zoom=canvas_hotkey_zoom,
            canvas_hotkey_adjust=canvas_hotkey_adjust,
            canvas_hotkey_shrink_brush=canvas_hotkey_shrink_brush,
            canvas_hotkey_grow_brush=canvas_hotkey_grow_brush,
            canvas_hotkey_move=canvas_hotkey_move,
            canvas_hotkey_fullscreen=canvas_hotkey_fullscreen,
            canvas_hotkey_reset=canvas_hotkey_reset,
            canvas_hotkey_overlap=canvas_hotkey_overlap,
            canvas_show_tooltip=canvas_show_tooltip,
            canvas_auto_expand=canvas_auto_expand,
            canvas_blur_prompt=canvas_blur_prompt,
            canvas_disabled_functions=canvas_disabled_functions,
            settings_in_ui=settings_in_ui,
            extra_options_txt2img=extra_options_txt2img,
            extra_options_img2img=extra_options_img2img,
            extra_options_cols=extra_options_cols,
            extra_options_accordion=extra_options_accordion,
        )

        options.additional_properties = d
        return options

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
