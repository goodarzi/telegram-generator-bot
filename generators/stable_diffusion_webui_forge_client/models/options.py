from typing import Any, Dict, Type, TypeVar, Tuple, Optional, BinaryIO, TextIO, TYPE_CHECKING

from typing import List


from attrs import define as _attrs_define
from attrs import field as _attrs_field

from ..types import UNSET, Unset

from ..types import UNSET, Unset
from typing import cast, List
from typing import Union
from typing import cast, Union






T = TypeVar("T", bound="Options")


@_attrs_define
class Options:
    r""" 
        Attributes:
            samples_save (Union[None, Unset, bool]): Always save all generated images Default: True.
            samples_format (Union[None, Unset, str]): File format for images Default: 'png'.
            samples_filename_pattern (Union[Any, None, Unset]): Images filename pattern
            save_images_add_number (Union[None, Unset, bool]): Add number to filename when saving Default: True.
            save_images_replace_action (Union[None, Unset, str]): Saving the image to an existing file Default: 'Replace'.
            grid_save (Union[None, Unset, bool]): Always save all generated image grids Default: True.
            grid_format (Union[None, Unset, str]): File format for grids Default: 'png'.
            grid_extended_filename (Union[Any, None, Unset]): Add extended info (seed, prompt) to filename when saving grid
                Default: False.
            grid_only_if_multiple (Union[None, Unset, bool]): Do not save grids consisting of one picture Default: True.
            grid_prevent_empty_spots (Union[Any, None, Unset]): Prevent empty spots in grid (when set to autodetect)
                Default: False.
            grid_zip_filename_pattern (Union[Any, None, Unset]): Archive filename pattern
            n_rows (Union[None, Unset, float]): Grid row count; use -1 for autodetect and 0 for it to be same as batch size
                Default: -1.0.
            font (Union[Any, None, Unset]): Font for image grids that have text
            grid_text_active_color (Union[None, Unset, str]): Text color for image grids Default: '#000000'.
            grid_text_inactive_color (Union[None, Unset, str]): Inactive text color for image grids Default: '#999999'.
            grid_background_color (Union[None, Unset, str]): Background color for image grids Default: '#ffffff'.
            save_images_before_face_restoration (Union[Any, None, Unset]): Save a copy of image before doing face
                restoration. Default: False.
            save_images_before_highres_fix (Union[Any, None, Unset]): Save a copy of image before applying highres fix.
                Default: False.
            save_images_before_color_correction (Union[Any, None, Unset]): Save a copy of image before applying color
                correction to img2img results Default: False.
            save_mask (Union[Any, None, Unset]): For inpainting, save a copy of the greyscale mask Default: False.
            save_mask_composite (Union[Any, None, Unset]): For inpainting, save a masked composite Default: False.
            jpeg_quality (Union[None, Unset, float]): Quality for saved jpeg and avif images Default: 80.0.
            webp_lossless (Union[Any, None, Unset]): Use lossless compression for webp images Default: False.
            export_for_4chan (Union[None, Unset, bool]): Save copy of large images as JPG Default: True.
            img_downscale_threshold (Union[None, Unset, float]): File size limit for the above option, MB Default: 4.0.
            target_side_length (Union[None, Unset, float]): Width/height limit for the above option, in pixels Default:
                4000.0.
            img_max_size_mp (Union[None, Unset, float]): Maximum image size Default: 200.0.
            use_original_name_batch (Union[None, Unset, bool]): Use original name for output filename during batch process
                in extras tab Default: True.
            use_upscaler_name_as_suffix (Union[Any, None, Unset]): Use upscaler name as filename suffix in the extras tab
                Default: False.
            save_selected_only (Union[None, Unset, bool]): When using 'Save' button, only save a single selected image
                Default: True.
            save_write_log_csv (Union[None, Unset, bool]): Write log.csv when saving images using 'Save' button Default:
                True.
            save_init_img (Union[Any, None, Unset]): Save init images when using img2img Default: False.
            temp_dir (Union[Any, None, Unset]): Directory for temporary images; leave empty for default
            clean_temp_dir_at_start (Union[Any, None, Unset]): Cleanup non-default temporary directory when starting webui
                Default: False.
            save_incomplete_images (Union[Any, None, Unset]): Save incomplete images Default: False.
            notification_audio (Union[None, Unset, bool]): Play notification sound after image generation Default: True.
            notification_volume (Union[None, Unset, float]): Notification sound volume Default: 100.0.
            outdir_samples (Union[Any, None, Unset]): Output directory for images; if empty, defaults to three directories
                below
            outdir_txt2img_samples (Union[None, Unset, str]): Output directory for txt2img images Default: 'outputs/txt2img-
                images'.
            outdir_img2img_samples (Union[None, Unset, str]): Output directory for img2img images Default: 'outputs/img2img-
                images'.
            outdir_extras_samples (Union[None, Unset, str]): Output directory for images from extras tab Default:
                'outputs/extras-images'.
            outdir_grids (Union[Any, None, Unset]): Output directory for grids; if empty, defaults to two directories below
            outdir_txt2img_grids (Union[None, Unset, str]): Output directory for txt2img grids Default: 'outputs/txt2img-
                grids'.
            outdir_img2img_grids (Union[None, Unset, str]): Output directory for img2img grids Default: 'outputs/img2img-
                grids'.
            outdir_save (Union[None, Unset, str]): Directory for saving images using the Save button Default: 'log/images'.
            outdir_init_images (Union[None, Unset, str]): Directory for saving init images when using img2img Default:
                'outputs/init-images'.
            save_to_dirs (Union[None, Unset, bool]): Save images to a subdirectory Default: True.
            grid_save_to_dirs (Union[None, Unset, bool]): Save grids to a subdirectory Default: True.
            use_save_to_dirs_for_ui (Union[Any, None, Unset]): When using "Save" button, save images to a subdirectory
                Default: False.
            directories_filename_pattern (Union[None, Unset, str]): Directory name pattern Default: '[date]'.
            directories_max_prompt_words (Union[None, Unset, float]): Max prompt words for [prompt_words] pattern Default:
                8.0.
            esrgan_tile (Union[None, Unset, float]): Tile size for ESRGAN upscalers. Default: 192.0.
            esrgan_tile_overlap (Union[None, Unset, float]): Tile overlap for ESRGAN upscalers. Default: 8.0.
            realesrgan_enabled_models (Union[List[Any], None, Unset]): Select which Real-ESRGAN models to show in the web
                UI.
            dat_enabled_models (Union[List[Any], None, Unset]): Select which DAT models to show in the web UI.
            dat_tile (Union[None, Unset, float]): Tile size for DAT upscalers. Default: 192.0.
            dat_tile_overlap (Union[None, Unset, float]): Tile overlap for DAT upscalers. Default: 8.0.
            upscaler_for_img2img (Union[Any, None, Unset]): Upscaler for img2img
            set_scale_by_when_changing_upscaler (Union[Any, None, Unset]): Automatically set the Scale by factor based on
                the name of the selected Upscaler. Default: False.
            face_restoration (Union[Any, None, Unset]): Restore faces Default: False.
            face_restoration_model (Union[None, Unset, str]): Face restoration model Default: 'CodeFormer'.
            code_former_weight (Union[None, Unset, float]): CodeFormer weight Default: 0.5.
            face_restoration_unload (Union[Any, None, Unset]): Move face restoration model from VRAM into RAM after
                processing Default: False.
            auto_launch_browser (Union[None, Unset, str]): Automatically open webui in browser on startup Default: 'Local'.
            enable_console_prompts (Union[Any, None, Unset]): Print prompts to console when generating with txt2img and
                img2img. Default: False.
            show_warnings (Union[Any, None, Unset]): Show warnings in console. Default: False.
            show_gradio_deprecation_warnings (Union[None, Unset, bool]): Show gradio deprecation warnings in console.
                Default: True.
            memmon_poll_rate (Union[None, Unset, float]): VRAM usage polls per second during generation. Default: 8.0.
            samples_log_stdout (Union[Any, None, Unset]): Always print all generation info to standard output Default:
                False.
            multiple_tqdm (Union[None, Unset, bool]): Add a second progress bar to the console that shows progress for an
                entire job. Default: True.
            enable_upscale_progressbar (Union[None, Unset, bool]): Show a progress bar in the console for tiled upscaling.
                Default: True.
            print_hypernet_extra (Union[Any, None, Unset]): Print extra hypernetwork information to console. Default: False.
            list_hidden_files (Union[None, Unset, bool]): Load models/files in hidden directories Default: True.
            disable_mmap_load_safetensors (Union[Any, None, Unset]): Disable memmapping for loading .safetensors files.
                Default: False.
            hide_ldm_prints (Union[None, Unset, bool]): Prevent Stability-AI's ldm/sgm modules from printing noise to
                console. Default: True.
            dump_stacks_on_signal (Union[Any, None, Unset]): Print stack traces before exiting the program with ctrl+c.
                Default: False.
            profiling_explanation (Union[None, Unset, str]):  Default: 'Those settings allow you to enable torch profiler
                when generating pictures.\nProfiling allows you to see which code uses how much of computer\'s resources during
                generation.\nEach generation writes its own profile to one file, overwriting previous.\nThe file can be viewed
                in <a href=\\"chrome:tracing\\">Chrome</a>, or on a <a href=\\"https://ui.perfetto.dev/\\">Perfetto</a> web
                site.\nWarning: writing profile can take a lot of time, up to 30 seconds, and the file itelf can be around 500MB
                in size.'.
            profiling_enable (Union[Any, None, Unset]): Enable profiling Default: False.
            profiling_activities (Union[List[Any], None, Unset]): Activities
            profiling_record_shapes (Union[None, Unset, bool]): Record shapes Default: True.
            profiling_profile_memory (Union[None, Unset, bool]): Profile memory Default: True.
            profiling_with_stack (Union[None, Unset, bool]): Include python stack Default: True.
            profiling_filename (Union[None, Unset, str]): Profile filename Default: 'trace.json'.
            api_enable_requests (Union[None, Unset, bool]): Allow http:// and https:// URLs for input images in API Default:
                True.
            api_forbid_local_requests (Union[None, Unset, bool]): Forbid URLs to local resources Default: True.
            api_useragent (Union[Any, None, Unset]): User agent for requests
            unload_models_when_training (Union[Any, None, Unset]): Move VAE and CLIP to RAM when training if possible. Saves
                VRAM. Default: False.
            pin_memory (Union[Any, None, Unset]): Turn on pin_memory for DataLoader. Makes training slightly faster but can
                increase memory usage. Default: False.
            save_optimizer_state (Union[Any, None, Unset]): Saves Optimizer state as separate *.optim file. Training of
                embedding or HN can be resumed with the matching optim file. Default: False.
            save_training_settings_to_txt (Union[None, Unset, bool]): Save textual inversion and hypernet settings to a text
                file whenever training starts. Default: True.
            dataset_filename_word_regex (Union[Any, None, Unset]): Filename word regex
            dataset_filename_join_string (Union[None, Unset, str]): Filename join string Default: ' '.
            training_image_repeats_per_epoch (Union[None, Unset, float]): Number of repeats for a single input image per
                epoch; used only for displaying epoch number Default: 1.0.
            training_write_csv_every (Union[None, Unset, float]): Save an csv containing the loss to log directory every N
                steps, 0 to disable Default: 500.0.
            training_xattention_optimizations (Union[Any, None, Unset]): Use cross attention optimizations while training
                Default: False.
            training_enable_tensorboard (Union[Any, None, Unset]): Enable tensorboard logging. Default: False.
            training_tensorboard_save_images (Union[Any, None, Unset]): Save generated images within tensorboard. Default:
                False.
            training_tensorboard_flush_every (Union[None, Unset, float]): How often, in seconds, to flush the pending
                tensorboard events and summaries to disk. Default: 120.0.
            sd_model_checkpoint (Union[Any, None, Unset]): (Managed by Forge)
            sd_checkpoints_limit (Union[None, Unset, float]): Maximum number of checkpoints loaded at the same time Default:
                1.0.
            sd_checkpoints_keep_in_cpu (Union[None, Unset, bool]): Only keep one model on device Default: True.
            sd_checkpoint_cache (Union[Any, None, Unset]): Checkpoints to cache in RAM Default: 0.
            sd_unet (Union[None, Unset, str]): SD Unet Default: 'Automatic'.
            enable_quantization (Union[Any, None, Unset]): Enable quantization in K samplers for sharper and cleaner
                results. This may change existing seeds Default: False.
            emphasis (Union[None, Unset, str]): Emphasis mode Default: 'Original'.
            enable_batch_seeds (Union[None, Unset, bool]): Make K-diffusion samplers produce same images in a batch as when
                making a single image Default: True.
            comma_padding_backtrack (Union[None, Unset, float]): Prompt word wrap length limit Default: 20.0.
            sdxl_clip_l_skip (Union[Any, None, Unset]): Clip skip SDXL Default: False.
            clip_stop_at_last_layers (Union[None, Unset, float]): (Managed by Forge) Default: 1.0.
            upcast_attn (Union[Any, None, Unset]): Upcast cross attention layer to float32 Default: False.
            randn_source (Union[None, Unset, str]): Random number generator source. Default: 'GPU'.
            tiling (Union[Any, None, Unset]): Tiling Default: False.
            hires_fix_refiner_pass (Union[None, Unset, str]): Hires fix: which pass to enable refiner for Default: 'second
                pass'.
            sdxl_crop_top (Union[Any, None, Unset]): crop top coordinate Default: 0.
            sdxl_crop_left (Union[Any, None, Unset]): crop left coordinate Default: 0.
            sdxl_refiner_low_aesthetic_score (Union[None, Unset, float]): SDXL low aesthetic score Default: 2.5.
            sdxl_refiner_high_aesthetic_score (Union[None, Unset, float]): SDXL high aesthetic score Default: 6.0.
            sd3_enable_t5 (Union[Any, None, Unset]): Enable T5 Default: False.
            sd_vae_explanation (Union[None, Unset, str]):  Default: "<abbr title='Variational autoencoder'>VAE</abbr> is a
                neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>\nimage into latent space
                representation and back. Latent space representation is what stable diffusion is working on during
                sampling\n(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting
                image after the sampling is finished.\nFor img2img, VAE is used to process user's input image before the
                sampling, and to create an image after sampling.".
            sd_vae_checkpoint_cache (Union[Any, None, Unset]): VAE Checkpoints to cache in RAM Default: 0.
            sd_vae (Union[None, Unset, str]): (Managed by Forge) Default: 'Automatic'.
            sd_vae_overrides_per_model_preferences (Union[None, Unset, bool]): Selected VAE overrides per-model preferences
                Default: True.
            auto_vae_precision_bfloat16 (Union[Any, None, Unset]): Automatically convert VAE to bfloat16 Default: False.
            auto_vae_precision (Union[None, Unset, bool]): Automatically revert VAE to 32-bit floats Default: True.
            sd_vae_encode_method (Union[None, Unset, str]): VAE type for encode Default: 'Full'.
            sd_vae_decode_method (Union[None, Unset, str]): VAE type for decode Default: 'Full'.
            inpainting_mask_weight (Union[None, Unset, float]): Inpainting conditioning mask strength Default: 1.0.
            initial_noise_multiplier (Union[None, Unset, float]): Noise multiplier for img2img Default: 1.0.
            img2img_extra_noise (Union[Any, None, Unset]): Extra noise multiplier for img2img and hires fix Default: 0.0.
            img2img_color_correction (Union[Any, None, Unset]): Apply color correction to img2img results to match original
                colors. Default: False.
            img2img_fix_steps (Union[Any, None, Unset]): With img2img, do exactly the amount of steps the slider specifies.
                Default: False.
            img2img_background_color (Union[None, Unset, str]): With img2img, fill transparent parts of the input image with
                this color. Default: '#ffffff'.
            img2img_sketch_default_brush_color (Union[None, Unset, str]): Sketch initial brush color Default: '#ffffff'.
            img2img_inpaint_mask_brush_color (Union[None, Unset, str]): Inpaint mask brush color Default: '#ffffff'.
            img2img_inpaint_sketch_default_brush_color (Union[None, Unset, str]): Inpaint sketch initial brush color
                Default: '#ffffff'.
            img2img_inpaint_mask_high_contrast (Union[None, Unset, bool]): For inpainting, use a high-contrast brush pattern
                Default: True.
            return_mask (Union[Any, None, Unset]): For inpainting, include the greyscale mask in results for web Default:
                False.
            return_mask_composite (Union[Any, None, Unset]): For inpainting, include masked composite in results for web
                Default: False.
            img2img_batch_show_results_limit (Union[None, Unset, float]): Show the first N batch img2img results in UI
                Default: 32.0.
            overlay_inpaint (Union[None, Unset, bool]): Overlay original for inpaint Default: True.
            cross_attention_optimization (Union[None, Unset, str]): Cross attention optimization Default: 'Automatic'.
            s_min_uncond (Union[Any, None, Unset]): Negative Guidance minimum sigma Default: 0.0.
            s_min_uncond_all (Union[Any, None, Unset]): Negative Guidance minimum sigma all steps Default: False.
            token_merging_ratio (Union[Any, None, Unset]): Token merging ratio Default: 0.0.
            token_merging_ratio_img2img (Union[Any, None, Unset]): Token merging ratio for img2img Default: 0.0.
            token_merging_ratio_hr (Union[Any, None, Unset]): Token merging ratio for high-res pass Default: 0.0.
            pad_cond_uncond (Union[Any, None, Unset]): Pad prompt/negative prompt Default: False.
            pad_cond_uncond_v0 (Union[Any, None, Unset]): Pad prompt/negative prompt (v0) Default: False.
            persistent_cond_cache (Union[None, Unset, bool]): Persistent cond cache Default: True.
            batch_cond_uncond (Union[None, Unset, bool]): Batch cond/uncond Default: True.
            fp8_storage (Union[None, Unset, str]): FP8 weight Default: 'Disable'.
            cache_fp16_weight (Union[Any, None, Unset]): Cache FP16 weight for LoRA Default: False.
            forge_try_reproduce (Union[None, Unset, str]): Try to reproduce the results from external software Default:
                'None'.
            auto_backcompat (Union[None, Unset, bool]): Automatic backward compatibility Default: True.
            use_old_emphasis_implementation (Union[Any, None, Unset]): Use old emphasis implementation. Can be useful to
                reproduce old seeds. Default: False.
            use_old_karras_scheduler_sigmas (Union[Any, None, Unset]): Use old karras scheduler sigmas (0.1 to 10). Default:
                False.
            no_dpmpp_sde_batch_determinism (Union[Any, None, Unset]): Do not make DPM++ SDE deterministic across different
                batch sizes. Default: False.
            use_old_hires_fix_width_height (Union[Any, None, Unset]): For hires fix, use width/height sliders to set final
                resolution rather than first pass (disables Upscale by, Resize width/height to). Default: False.
            hires_fix_use_firstpass_conds (Union[Any, None, Unset]): For hires fix, calculate conds of second pass using
                extra networks of first pass. Default: False.
            use_old_scheduling (Union[Any, None, Unset]): Use old prompt editing timelines. Default: False.
            use_downcasted_alpha_bar (Union[Any, None, Unset]): Downcast model alphas_cumprod to fp16 before sampling. For
                reproducing old seeds. Default: False.
            refiner_switch_by_sample_steps (Union[Any, None, Unset]): Switch to refiner by sampling steps instead of model
                timesteps. Old behavior for refiner. Default: False.
            interrogate_keep_models_in_memory (Union[Any, None, Unset]): Keep models in VRAM Default: False.
            interrogate_return_ranks (Union[Any, None, Unset]): Include ranks of model tags matches in results. Default:
                False.
            interrogate_clip_num_beams (Union[None, Unset, float]): BLIP: num_beams Default: 1.0.
            interrogate_clip_min_length (Union[None, Unset, float]): BLIP: minimum description length Default: 24.0.
            interrogate_clip_max_length (Union[None, Unset, float]): BLIP: maximum description length Default: 48.0.
            interrogate_clip_dict_limit (Union[None, Unset, float]): CLIP: maximum number of lines in text file Default:
                1500.0.
            interrogate_clip_skip_categories (Union[Any, None, Unset]): CLIP: skip inquire categories Default: [].
            interrogate_deepbooru_score_threshold (Union[None, Unset, float]): deepbooru: score threshold Default: 0.5.
            deepbooru_sort_alpha (Union[None, Unset, bool]): deepbooru: sort tags alphabetically Default: True.
            deepbooru_use_spaces (Union[None, Unset, bool]): deepbooru: use spaces in tags Default: True.
            deepbooru_escape (Union[None, Unset, bool]): deepbooru: escape (\) brackets Default: True.
            deepbooru_filter_tags (Union[Any, None, Unset]): deepbooru: filter out those tags
            extra_networks_show_hidden_directories (Union[None, Unset, bool]): Show hidden directories Default: True.
            extra_networks_dir_button_function (Union[Any, None, Unset]): Add a '/' to the beginning of directory buttons
                Default: False.
            extra_networks_hidden_models (Union[None, Unset, str]): Show cards for models in hidden directories Default:
                'When searched'.
            extra_networks_default_multiplier (Union[None, Unset, float]): Default multiplier for extra networks Default:
                1.0.
            extra_networks_card_width (Union[Any, None, Unset]): Card width for Extra Networks Default: 0.
            extra_networks_card_height (Union[Any, None, Unset]): Card height for Extra Networks Default: 0.
            extra_networks_card_text_scale (Union[None, Unset, float]): Card text scale Default: 1.0.
            extra_networks_card_show_desc (Union[None, Unset, bool]): Show description on card Default: True.
            extra_networks_card_description_is_html (Union[Any, None, Unset]): Treat card description as HTML Default:
                False.
            extra_networks_card_order_field (Union[None, Unset, str]): Default order field for Extra Networks cards Default:
                'Path'.
            extra_networks_card_order (Union[None, Unset, str]): Default order for Extra Networks cards Default:
                'Ascending'.
            extra_networks_tree_view_style (Union[None, Unset, str]): Extra Networks directory view style Default: 'Dirs'.
            extra_networks_tree_view_default_enabled (Union[None, Unset, bool]): Show the Extra Networks directory view by
                default Default: True.
            extra_networks_tree_view_default_width (Union[None, Unset, float]): Default width for the Extra Networks
                directory tree view Default: 180.0.
            extra_networks_add_text_separator (Union[None, Unset, str]): Extra networks separator Default: ' '.
            ui_extra_networks_tab_reorder (Union[Any, None, Unset]): Extra networks tab order
            textual_inversion_print_at_load (Union[Any, None, Unset]): Print a list of Textual Inversion embeddings when
                loading model Default: False.
            textual_inversion_add_hashes_to_infotext (Union[None, Unset, bool]): Add Textual Inversion hashes to infotext
                Default: True.
            sd_hypernetwork (Union[None, Unset, str]): Add hypernetwork to prompt Default: 'None'.
            keyedit_precision_attention (Union[None, Unset, float]): Precision for (attention:1.1) when editing the prompt
                with Ctrl+up/down Default: 0.1.
            keyedit_precision_extra (Union[None, Unset, float]): Precision for <extra networks:0.9> when editing the prompt
                with Ctrl+up/down Default: 0.05.
            keyedit_delimiters (Union[None, Unset, str]): Word delimiters when editing the prompt with Ctrl+up/down Default:
                '.,\\/!?%^*;:{}=`~() '.
            keyedit_delimiters_whitespace (Union[List[Any], None, Unset]): Ctrl+up/down whitespace delimiters
            keyedit_move (Union[None, Unset, bool]): Alt+left/right moves prompt elements Default: True.
            disable_token_counters (Union[Any, None, Unset]): Disable prompt token counters Default: False.
            include_styles_into_token_counters (Union[None, Unset, bool]): Count tokens of enabled styles Default: True.
            return_grid (Union[None, Unset, bool]): Show grid in gallery Default: True.
            do_not_show_images (Union[Any, None, Unset]): Do not show any images in gallery Default: False.
            js_modal_lightbox (Union[None, Unset, bool]): Full page image viewer: enable Default: True.
            js_modal_lightbox_initially_zoomed (Union[None, Unset, bool]): Full page image viewer: show images zoomed in by
                default Default: True.
            js_modal_lightbox_gamepad (Union[Any, None, Unset]): Full page image viewer: navigate with gamepad Default:
                False.
            js_modal_lightbox_gamepad_repeat (Union[None, Unset, float]): Full page image viewer: gamepad repeat period
                Default: 250.0.
            sd_webui_modal_lightbox_icon_opacity (Union[None, Unset, float]): Full page image viewer: control icon unfocused
                opacity Default: 1.0.
            sd_webui_modal_lightbox_toolbar_opacity (Union[None, Unset, float]): Full page image viewer: tool bar opacity
                Default: 0.9.
            gallery_height (Union[Any, None, Unset]): Gallery height
            open_dir_button_choice (Union[None, Unset, str]): What directory the [📂] button opens Default: 'Subdirectory'.
            compact_prompt_box (Union[Any, None, Unset]): Compact prompt layout Default: False.
            samplers_in_dropdown (Union[None, Unset, bool]): Use dropdown for sampler selection instead of radio group
                Default: True.
            dimensions_and_batch_together (Union[None, Unset, bool]): Show Width/Height and Batch sliders in same row
                Default: True.
            sd_checkpoint_dropdown_use_short (Union[Any, None, Unset]): Checkpoint dropdown: use filenames without paths
                Default: False.
            hires_fix_show_sampler (Union[Any, None, Unset]): Hires fix: show hires checkpoint and sampler selection
                Default: False.
            hires_fix_show_prompts (Union[Any, None, Unset]): Hires fix: show hires prompt and negative prompt Default:
                False.
            txt2img_settings_accordion (Union[Any, None, Unset]): Settings in txt2img hidden under Accordion Default: False.
            img2img_settings_accordion (Union[Any, None, Unset]): Settings in img2img hidden under Accordion Default: False.
            interrupt_after_current (Union[None, Unset, bool]): Don't Interrupt in the middle Default: True.
            localization (Union[None, Unset, str]): Localization Default: 'None'.
            quick_setting_list (Union[Any, None, Unset]): Quicksettings list Default: [].
            ui_tab_order (Union[Any, None, Unset]): UI tab order Default: [].
            hidden_tabs (Union[Any, None, Unset]): Hidden UI tabs Default: [].
            ui_reorder_list (Union[Any, None, Unset]): UI item order for txt2img/img2img tabs Default: [].
            gradio_theme (Union[None, Unset, str]): Gradio theme Default: 'Default'.
            gradio_themes_cache (Union[None, Unset, bool]): Cache gradio themes locally Default: True.
            show_progress_in_title (Union[None, Unset, bool]): Show generation progress in window title. Default: True.
            send_seed (Union[None, Unset, bool]): Send seed when sending prompt or image to other interface Default: True.
            send_size (Union[None, Unset, bool]): Send size when sending prompt or image to another interface Default: True.
            enable_reloading_ui_scripts (Union[Any, None, Unset]): Reload UI scripts when using Reload UI option Default:
                False.
            infotext_explanation (Union[None, Unset, str]):  Default: 'Infotext is what this software calls the text that
                contains generation parameters and can be used to generate the same picture again.\nIt is displayed in UI below
                the image. To use infotext, paste it into the prompt and click the ↙️ paste button.'.
            enable_pnginfo (Union[None, Unset, bool]): Write infotext to metadata of the generated image Default: True.
            save_txt (Union[Any, None, Unset]): Create a text file with infotext next to every generated image Default:
                False.
            add_model_name_to_info (Union[None, Unset, bool]): Add model name to infotext Default: True.
            add_model_hash_to_info (Union[None, Unset, bool]): Add model hash to infotext Default: True.
            add_vae_name_to_info (Union[None, Unset, bool]): Add VAE name to infotext Default: True.
            add_vae_hash_to_info (Union[None, Unset, bool]): Add VAE hash to infotext Default: True.
            add_user_name_to_info (Union[Any, None, Unset]): Add user name to infotext when authenticated Default: False.
            add_version_to_infotext (Union[None, Unset, bool]): Add program version to infotext Default: True.
            disable_weights_auto_swap (Union[None, Unset, bool]): Disregard checkpoint information from pasted infotext
                Default: True.
            infotext_skip_pasting (Union[Any, None, Unset]): Disregard fields from pasted infotext Default: [].
            infotext_styles (Union[None, Unset, str]): Infer styles from prompts of pasted infotext Default: 'Apply if any'.
            show_progressbar (Union[None, Unset, bool]): Show progressbar Default: True.
            live_previews_enable (Union[None, Unset, bool]): Show live previews of the created image Default: True.
            live_previews_image_format (Union[None, Unset, str]): Live preview file format Default: 'png'.
            show_progress_grid (Union[None, Unset, bool]): Show previews of all images generated in a batch as a grid
                Default: True.
            show_progress_every_n_steps (Union[None, Unset, float]): Live preview display period Default: 10.0.
            show_progress_type (Union[None, Unset, str]): Live preview method Default: 'Approx NN'.
            live_preview_allow_lowvram_full (Union[Any, None, Unset]): Allow Full live preview method with lowvram/medvram
                Default: False.
            live_preview_content (Union[None, Unset, str]): Live preview subject Default: 'Prompt'.
            live_preview_refresh_period (Union[None, Unset, float]): Progressbar and preview update period Default: 1000.0.
            live_preview_fast_interrupt (Union[Any, None, Unset]): Return image with chosen live preview method on interrupt
                Default: False.
            js_live_preview_in_modal_lightbox (Union[Any, None, Unset]): Show Live preview in full page image viewer
                Default: False.
            prevent_screen_sleep_during_generation (Union[None, Unset, bool]): Prevent screen sleep during generation
                Default: True.
            hide_samplers (Union[Any, None, Unset]): Hide samplers in user interface Default: [].
            eta_ddim (Union[Any, None, Unset]): Eta for DDIM Default: 0.0.
            eta_ancestral (Union[None, Unset, float]): Eta for k-diffusion samplers Default: 1.0.
            ddim_discretize (Union[None, Unset, str]): img2img DDIM discretize Default: 'uniform'.
            s_churn (Union[Any, None, Unset]): sigma churn Default: 0.0.
            s_tmin (Union[Any, None, Unset]): sigma tmin Default: 0.0.
            s_tmax (Union[Any, None, Unset]): sigma tmax Default: 0.0.
            s_noise (Union[None, Unset, float]): sigma noise Default: 1.0.
            sigma_min (Union[Any, None, Unset]): sigma min Default: 0.0.
            sigma_max (Union[Any, None, Unset]): sigma max Default: 0.0.
            rho (Union[Any, None, Unset]): rho Default: 0.0.
            eta_noise_seed_delta (Union[Any, None, Unset]): Eta noise seed delta Default: 0.
            always_discard_next_to_last_sigma (Union[Any, None, Unset]): Always discard next-to-last sigma Default: False.
            sgm_noise_multiplier (Union[Any, None, Unset]): SGM noise multiplier Default: False.
            uni_pc_variant (Union[None, Unset, str]): UniPC variant Default: 'bh1'.
            uni_pc_skip_type (Union[None, Unset, str]): UniPC skip type Default: 'time_uniform'.
            uni_pc_order (Union[None, Unset, float]): UniPC order Default: 3.0.
            uni_pc_lower_order_final (Union[None, Unset, bool]): UniPC lower order final Default: True.
            sd_noise_schedule (Union[None, Unset, str]): Noise schedule for sampling Default: 'Default'.
            skip_early_cond (Union[Any, None, Unset]): Ignore negative prompt during early sampling Default: 0.0.
            beta_dist_alpha (Union[None, Unset, float]): Beta scheduler - alpha Default: 0.6.
            beta_dist_beta (Union[None, Unset, float]): Beta scheduler - beta Default: 0.6.
            postprocessing_enable_in_main_ui (Union[Any, None, Unset]): Enable postprocessing operations in txt2img and
                img2img tabs Default: [].
            postprocessing_disable_in_extras (Union[Any, None, Unset]): Disable postprocessing operations in extras tab
                Default: [].
            postprocessing_operation_order (Union[Any, None, Unset]): Postprocessing operation order Default: [].
            upscaling_max_images_in_cache (Union[None, Unset, float]): Maximum number of images in upscaling cache Default:
                5.0.
            postprocessing_existing_caption_action (Union[None, Unset, str]): Action for existing captions Default:
                'Ignore'.
            disabled_extensions (Union[Any, None, Unset]): Disable these extensions Default: [].
            disable_all_extensions (Union[None, Unset, str]): Disable all extensions (preserves the list of disabled
                extensions) Default: 'none'.
            restore_config_state_file (Union[Any, None, Unset]): Config state file to restore from, under 'config-states/'
                folder
            sd_checkpoint_hash (Union[Any, None, Unset]): SHA256 hash of the current checkpoint
            forge_unet_storage_dtype (Union[None, Unset, str]):  Default: 'Automatic'.
            forge_inference_memory (Union[None, Unset, float]):  Default: 1024.0.
            forge_async_loading (Union[None, Unset, str]):  Default: 'Queue'.
            forge_pin_shared_memory (Union[None, Unset, str]):  Default: 'CPU'.
            forge_preset (Union[None, Unset, str]):  Default: 'sd'.
            forge_additional_modules (Union[Any, None, Unset]):  Default: [].
            settings_in_ui (Union[None, Unset, str]):  Default: 'This page allows you to add some settings to the main
                interface of txt2img and img2img tabs.'.
            extra_options_txt2img (Union[Any, None, Unset]): Settings for txt2img Default: [].
            extra_options_img2img (Union[Any, None, Unset]): Settings for img2img Default: [].
            extra_options_cols (Union[None, Unset, float]): Number of columns for added settings Default: 1.0.
            extra_options_accordion (Union[Any, None, Unset]): Place added settings into an accordion Default: False.
     """

    samples_save: Union[None, Unset, bool] = True
    samples_format: Union[None, Unset, str] = 'png'
    samples_filename_pattern: Union[Any, None, Unset] = ''
    save_images_add_number: Union[None, Unset, bool] = True
    save_images_replace_action: Union[None, Unset, str] = 'Replace'
    grid_save: Union[None, Unset, bool] = True
    grid_format: Union[None, Unset, str] = 'png'
    grid_extended_filename: Union[Any, None, Unset] = False
    grid_only_if_multiple: Union[None, Unset, bool] = True
    grid_prevent_empty_spots: Union[Any, None, Unset] = False
    grid_zip_filename_pattern: Union[Any, None, Unset] = ''
    n_rows: Union[None, Unset, float] = -1.0
    font: Union[Any, None, Unset] = ''
    grid_text_active_color: Union[None, Unset, str] = '#000000'
    grid_text_inactive_color: Union[None, Unset, str] = '#999999'
    grid_background_color: Union[None, Unset, str] = '#ffffff'
    save_images_before_face_restoration: Union[Any, None, Unset] = False
    save_images_before_highres_fix: Union[Any, None, Unset] = False
    save_images_before_color_correction: Union[Any, None, Unset] = False
    save_mask: Union[Any, None, Unset] = False
    save_mask_composite: Union[Any, None, Unset] = False
    jpeg_quality: Union[None, Unset, float] = 80.0
    webp_lossless: Union[Any, None, Unset] = False
    export_for_4chan: Union[None, Unset, bool] = True
    img_downscale_threshold: Union[None, Unset, float] = 4.0
    target_side_length: Union[None, Unset, float] = 4000.0
    img_max_size_mp: Union[None, Unset, float] = 200.0
    use_original_name_batch: Union[None, Unset, bool] = True
    use_upscaler_name_as_suffix: Union[Any, None, Unset] = False
    save_selected_only: Union[None, Unset, bool] = True
    save_write_log_csv: Union[None, Unset, bool] = True
    save_init_img: Union[Any, None, Unset] = False
    temp_dir: Union[Any, None, Unset] = ''
    clean_temp_dir_at_start: Union[Any, None, Unset] = False
    save_incomplete_images: Union[Any, None, Unset] = False
    notification_audio: Union[None, Unset, bool] = True
    notification_volume: Union[None, Unset, float] = 100.0
    outdir_samples: Union[Any, None, Unset] = ''
    outdir_txt2img_samples: Union[None, Unset, str] = 'outputs/txt2img-images'
    outdir_img2img_samples: Union[None, Unset, str] = 'outputs/img2img-images'
    outdir_extras_samples: Union[None, Unset, str] = 'outputs/extras-images'
    outdir_grids: Union[Any, None, Unset] = ''
    outdir_txt2img_grids: Union[None, Unset, str] = 'outputs/txt2img-grids'
    outdir_img2img_grids: Union[None, Unset, str] = 'outputs/img2img-grids'
    outdir_save: Union[None, Unset, str] = 'log/images'
    outdir_init_images: Union[None, Unset, str] = 'outputs/init-images'
    save_to_dirs: Union[None, Unset, bool] = True
    grid_save_to_dirs: Union[None, Unset, bool] = True
    use_save_to_dirs_for_ui: Union[Any, None, Unset] = False
    directories_filename_pattern: Union[None, Unset, str] = '[date]'
    directories_max_prompt_words: Union[None, Unset, float] = 8.0
    esrgan_tile: Union[None, Unset, float] = 192.0
    esrgan_tile_overlap: Union[None, Unset, float] = 8.0
    realesrgan_enabled_models: Union[List[Any], None, Unset] = UNSET
    dat_enabled_models: Union[List[Any], None, Unset] = UNSET
    dat_tile: Union[None, Unset, float] = 192.0
    dat_tile_overlap: Union[None, Unset, float] = 8.0
    upscaler_for_img2img: Union[Any, None, Unset] = UNSET
    set_scale_by_when_changing_upscaler: Union[Any, None, Unset] = False
    face_restoration: Union[Any, None, Unset] = False
    face_restoration_model: Union[None, Unset, str] = 'CodeFormer'
    code_former_weight: Union[None, Unset, float] = 0.5
    face_restoration_unload: Union[Any, None, Unset] = False
    auto_launch_browser: Union[None, Unset, str] = 'Local'
    enable_console_prompts: Union[Any, None, Unset] = False
    show_warnings: Union[Any, None, Unset] = False
    show_gradio_deprecation_warnings: Union[None, Unset, bool] = True
    memmon_poll_rate: Union[None, Unset, float] = 8.0
    samples_log_stdout: Union[Any, None, Unset] = False
    multiple_tqdm: Union[None, Unset, bool] = True
    enable_upscale_progressbar: Union[None, Unset, bool] = True
    print_hypernet_extra: Union[Any, None, Unset] = False
    list_hidden_files: Union[None, Unset, bool] = True
    disable_mmap_load_safetensors: Union[Any, None, Unset] = False
    hide_ldm_prints: Union[None, Unset, bool] = True
    dump_stacks_on_signal: Union[Any, None, Unset] = False
    profiling_explanation: Union[None, Unset, str] = 'Those settings allow you to enable torch profiler when generating pictures.\nProfiling allows you to see which code uses how much of computer\'s resources during generation.\nEach generation writes its own profile to one file, overwriting previous.\nThe file can be viewed in <a href=\\"chrome:tracing\\">Chrome</a>, or on a <a href=\\"https://ui.perfetto.dev/\\">Perfetto</a> web site.\nWarning: writing profile can take a lot of time, up to 30 seconds, and the file itelf can be around 500MB in size.'
    profiling_enable: Union[Any, None, Unset] = False
    profiling_activities: Union[List[Any], None, Unset] = UNSET
    profiling_record_shapes: Union[None, Unset, bool] = True
    profiling_profile_memory: Union[None, Unset, bool] = True
    profiling_with_stack: Union[None, Unset, bool] = True
    profiling_filename: Union[None, Unset, str] = 'trace.json'
    api_enable_requests: Union[None, Unset, bool] = True
    api_forbid_local_requests: Union[None, Unset, bool] = True
    api_useragent: Union[Any, None, Unset] = ''
    unload_models_when_training: Union[Any, None, Unset] = False
    pin_memory: Union[Any, None, Unset] = False
    save_optimizer_state: Union[Any, None, Unset] = False
    save_training_settings_to_txt: Union[None, Unset, bool] = True
    dataset_filename_word_regex: Union[Any, None, Unset] = ''
    dataset_filename_join_string: Union[None, Unset, str] = ' '
    training_image_repeats_per_epoch: Union[None, Unset, float] = 1.0
    training_write_csv_every: Union[None, Unset, float] = 500.0
    training_xattention_optimizations: Union[Any, None, Unset] = False
    training_enable_tensorboard: Union[Any, None, Unset] = False
    training_tensorboard_save_images: Union[Any, None, Unset] = False
    training_tensorboard_flush_every: Union[None, Unset, float] = 120.0
    sd_model_checkpoint: Union[Any, None, Unset] = UNSET
    sd_checkpoints_limit: Union[None, Unset, float] = 1.0
    sd_checkpoints_keep_in_cpu: Union[None, Unset, bool] = True
    sd_checkpoint_cache: Union[Any, None, Unset] = 0
    sd_unet: Union[None, Unset, str] = 'Automatic'
    enable_quantization: Union[Any, None, Unset] = False
    emphasis: Union[None, Unset, str] = 'Original'
    enable_batch_seeds: Union[None, Unset, bool] = True
    comma_padding_backtrack: Union[None, Unset, float] = 20.0
    sdxl_clip_l_skip: Union[Any, None, Unset] = False
    clip_stop_at_last_layers: Union[None, Unset, float] = 1.0
    upcast_attn: Union[Any, None, Unset] = False
    randn_source: Union[None, Unset, str] = 'GPU'
    tiling: Union[Any, None, Unset] = False
    hires_fix_refiner_pass: Union[None, Unset, str] = 'second pass'
    sdxl_crop_top: Union[Any, None, Unset] = 0
    sdxl_crop_left: Union[Any, None, Unset] = 0
    sdxl_refiner_low_aesthetic_score: Union[None, Unset, float] = 2.5
    sdxl_refiner_high_aesthetic_score: Union[None, Unset, float] = 6.0
    sd3_enable_t5: Union[Any, None, Unset] = False
    sd_vae_explanation: Union[None, Unset, str] = "<abbr title='Variational autoencoder'>VAE</abbr> is a neural network that transforms a standard <abbr title='red/green/blue'>RGB</abbr>\nimage into latent space representation and back. Latent space representation is what stable diffusion is working on during sampling\n(i.e. when the progress bar is between empty and full). For txt2img, VAE is used to create a resulting image after the sampling is finished.\nFor img2img, VAE is used to process user's input image before the sampling, and to create an image after sampling."
    sd_vae_checkpoint_cache: Union[Any, None, Unset] = 0
    sd_vae: Union[None, Unset, str] = 'Automatic'
    sd_vae_overrides_per_model_preferences: Union[None, Unset, bool] = True
    auto_vae_precision_bfloat16: Union[Any, None, Unset] = False
    auto_vae_precision: Union[None, Unset, bool] = True
    sd_vae_encode_method: Union[None, Unset, str] = 'Full'
    sd_vae_decode_method: Union[None, Unset, str] = 'Full'
    inpainting_mask_weight: Union[None, Unset, float] = 1.0
    initial_noise_multiplier: Union[None, Unset, float] = 1.0
    img2img_extra_noise: Union[Any, None, Unset] = 0.0
    img2img_color_correction: Union[Any, None, Unset] = False
    img2img_fix_steps: Union[Any, None, Unset] = False
    img2img_background_color: Union[None, Unset, str] = '#ffffff'
    img2img_sketch_default_brush_color: Union[None, Unset, str] = '#ffffff'
    img2img_inpaint_mask_brush_color: Union[None, Unset, str] = '#ffffff'
    img2img_inpaint_sketch_default_brush_color: Union[None, Unset, str] = '#ffffff'
    img2img_inpaint_mask_high_contrast: Union[None, Unset, bool] = True
    return_mask: Union[Any, None, Unset] = False
    return_mask_composite: Union[Any, None, Unset] = False
    img2img_batch_show_results_limit: Union[None, Unset, float] = 32.0
    overlay_inpaint: Union[None, Unset, bool] = True
    cross_attention_optimization: Union[None, Unset, str] = 'Automatic'
    s_min_uncond: Union[Any, None, Unset] = 0.0
    s_min_uncond_all: Union[Any, None, Unset] = False
    token_merging_ratio: Union[Any, None, Unset] = 0.0
    token_merging_ratio_img2img: Union[Any, None, Unset] = 0.0
    token_merging_ratio_hr: Union[Any, None, Unset] = 0.0
    pad_cond_uncond: Union[Any, None, Unset] = False
    pad_cond_uncond_v0: Union[Any, None, Unset] = False
    persistent_cond_cache: Union[None, Unset, bool] = True
    batch_cond_uncond: Union[None, Unset, bool] = True
    fp8_storage: Union[None, Unset, str] = 'Disable'
    cache_fp16_weight: Union[Any, None, Unset] = False
    forge_try_reproduce: Union[None, Unset, str] = 'None'
    auto_backcompat: Union[None, Unset, bool] = True
    use_old_emphasis_implementation: Union[Any, None, Unset] = False
    use_old_karras_scheduler_sigmas: Union[Any, None, Unset] = False
    no_dpmpp_sde_batch_determinism: Union[Any, None, Unset] = False
    use_old_hires_fix_width_height: Union[Any, None, Unset] = False
    hires_fix_use_firstpass_conds: Union[Any, None, Unset] = False
    use_old_scheduling: Union[Any, None, Unset] = False
    use_downcasted_alpha_bar: Union[Any, None, Unset] = False
    refiner_switch_by_sample_steps: Union[Any, None, Unset] = False
    interrogate_keep_models_in_memory: Union[Any, None, Unset] = False
    interrogate_return_ranks: Union[Any, None, Unset] = False
    interrogate_clip_num_beams: Union[None, Unset, float] = 1.0
    interrogate_clip_min_length: Union[None, Unset, float] = 24.0
    interrogate_clip_max_length: Union[None, Unset, float] = 48.0
    interrogate_clip_dict_limit: Union[None, Unset, float] = 1500.0
    interrogate_clip_skip_categories: Union[Any, None, Unset] = []
    interrogate_deepbooru_score_threshold: Union[None, Unset, float] = 0.5
    deepbooru_sort_alpha: Union[None, Unset, bool] = True
    deepbooru_use_spaces: Union[None, Unset, bool] = True
    deepbooru_escape: Union[None, Unset, bool] = True
    deepbooru_filter_tags: Union[Any, None, Unset] = ''
    extra_networks_show_hidden_directories: Union[None, Unset, bool] = True
    extra_networks_dir_button_function: Union[Any, None, Unset] = False
    extra_networks_hidden_models: Union[None, Unset, str] = 'When searched'
    extra_networks_default_multiplier: Union[None, Unset, float] = 1.0
    extra_networks_card_width: Union[Any, None, Unset] = 0
    extra_networks_card_height: Union[Any, None, Unset] = 0
    extra_networks_card_text_scale: Union[None, Unset, float] = 1.0
    extra_networks_card_show_desc: Union[None, Unset, bool] = True
    extra_networks_card_description_is_html: Union[Any, None, Unset] = False
    extra_networks_card_order_field: Union[None, Unset, str] = 'Path'
    extra_networks_card_order: Union[None, Unset, str] = 'Ascending'
    extra_networks_tree_view_style: Union[None, Unset, str] = 'Dirs'
    extra_networks_tree_view_default_enabled: Union[None, Unset, bool] = True
    extra_networks_tree_view_default_width: Union[None, Unset, float] = 180.0
    extra_networks_add_text_separator: Union[None, Unset, str] = ' '
    ui_extra_networks_tab_reorder: Union[Any, None, Unset] = ''
    textual_inversion_print_at_load: Union[Any, None, Unset] = False
    textual_inversion_add_hashes_to_infotext: Union[None, Unset, bool] = True
    sd_hypernetwork: Union[None, Unset, str] = 'None'
    keyedit_precision_attention: Union[None, Unset, float] = 0.1
    keyedit_precision_extra: Union[None, Unset, float] = 0.05
    keyedit_delimiters: Union[None, Unset, str] = '.,\\/!?%^*;:{}=`~() '
    keyedit_delimiters_whitespace: Union[List[Any], None, Unset] = UNSET
    keyedit_move: Union[None, Unset, bool] = True
    disable_token_counters: Union[Any, None, Unset] = False
    include_styles_into_token_counters: Union[None, Unset, bool] = True
    return_grid: Union[None, Unset, bool] = True
    do_not_show_images: Union[Any, None, Unset] = False
    js_modal_lightbox: Union[None, Unset, bool] = True
    js_modal_lightbox_initially_zoomed: Union[None, Unset, bool] = True
    js_modal_lightbox_gamepad: Union[Any, None, Unset] = False
    js_modal_lightbox_gamepad_repeat: Union[None, Unset, float] = 250.0
    sd_webui_modal_lightbox_icon_opacity: Union[None, Unset, float] = 1.0
    sd_webui_modal_lightbox_toolbar_opacity: Union[None, Unset, float] = 0.9
    gallery_height: Union[Any, None, Unset] = ''
    open_dir_button_choice: Union[None, Unset, str] = 'Subdirectory'
    compact_prompt_box: Union[Any, None, Unset] = False
    samplers_in_dropdown: Union[None, Unset, bool] = True
    dimensions_and_batch_together: Union[None, Unset, bool] = True
    sd_checkpoint_dropdown_use_short: Union[Any, None, Unset] = False
    hires_fix_show_sampler: Union[Any, None, Unset] = False
    hires_fix_show_prompts: Union[Any, None, Unset] = False
    txt2img_settings_accordion: Union[Any, None, Unset] = False
    img2img_settings_accordion: Union[Any, None, Unset] = False
    interrupt_after_current: Union[None, Unset, bool] = True
    localization: Union[None, Unset, str] = 'None'
    quick_setting_list: Union[Any, None, Unset] = []
    ui_tab_order: Union[Any, None, Unset] = []
    hidden_tabs: Union[Any, None, Unset] = []
    ui_reorder_list: Union[Any, None, Unset] = []
    gradio_theme: Union[None, Unset, str] = 'Default'
    gradio_themes_cache: Union[None, Unset, bool] = True
    show_progress_in_title: Union[None, Unset, bool] = True
    send_seed: Union[None, Unset, bool] = True
    send_size: Union[None, Unset, bool] = True
    enable_reloading_ui_scripts: Union[Any, None, Unset] = False
    infotext_explanation: Union[None, Unset, str] = 'Infotext is what this software calls the text that contains generation parameters and can be used to generate the same picture again.\nIt is displayed in UI below the image. To use infotext, paste it into the prompt and click the ↙️ paste button.'
    enable_pnginfo: Union[None, Unset, bool] = True
    save_txt: Union[Any, None, Unset] = False
    add_model_name_to_info: Union[None, Unset, bool] = True
    add_model_hash_to_info: Union[None, Unset, bool] = True
    add_vae_name_to_info: Union[None, Unset, bool] = True
    add_vae_hash_to_info: Union[None, Unset, bool] = True
    add_user_name_to_info: Union[Any, None, Unset] = False
    add_version_to_infotext: Union[None, Unset, bool] = True
    disable_weights_auto_swap: Union[None, Unset, bool] = True
    infotext_skip_pasting: Union[Any, None, Unset] = []
    infotext_styles: Union[None, Unset, str] = 'Apply if any'
    show_progressbar: Union[None, Unset, bool] = True
    live_previews_enable: Union[None, Unset, bool] = True
    live_previews_image_format: Union[None, Unset, str] = 'png'
    show_progress_grid: Union[None, Unset, bool] = True
    show_progress_every_n_steps: Union[None, Unset, float] = 10.0
    show_progress_type: Union[None, Unset, str] = 'Approx NN'
    live_preview_allow_lowvram_full: Union[Any, None, Unset] = False
    live_preview_content: Union[None, Unset, str] = 'Prompt'
    live_preview_refresh_period: Union[None, Unset, float] = 1000.0
    live_preview_fast_interrupt: Union[Any, None, Unset] = False
    js_live_preview_in_modal_lightbox: Union[Any, None, Unset] = False
    prevent_screen_sleep_during_generation: Union[None, Unset, bool] = True
    hide_samplers: Union[Any, None, Unset] = []
    eta_ddim: Union[Any, None, Unset] = 0.0
    eta_ancestral: Union[None, Unset, float] = 1.0
    ddim_discretize: Union[None, Unset, str] = 'uniform'
    s_churn: Union[Any, None, Unset] = 0.0
    s_tmin: Union[Any, None, Unset] = 0.0
    s_tmax: Union[Any, None, Unset] = 0.0
    s_noise: Union[None, Unset, float] = 1.0
    sigma_min: Union[Any, None, Unset] = 0.0
    sigma_max: Union[Any, None, Unset] = 0.0
    rho: Union[Any, None, Unset] = 0.0
    eta_noise_seed_delta: Union[Any, None, Unset] = 0
    always_discard_next_to_last_sigma: Union[Any, None, Unset] = False
    sgm_noise_multiplier: Union[Any, None, Unset] = False
    uni_pc_variant: Union[None, Unset, str] = 'bh1'
    uni_pc_skip_type: Union[None, Unset, str] = 'time_uniform'
    uni_pc_order: Union[None, Unset, float] = 3.0
    uni_pc_lower_order_final: Union[None, Unset, bool] = True
    sd_noise_schedule: Union[None, Unset, str] = 'Default'
    skip_early_cond: Union[Any, None, Unset] = 0.0
    beta_dist_alpha: Union[None, Unset, float] = 0.6
    beta_dist_beta: Union[None, Unset, float] = 0.6
    postprocessing_enable_in_main_ui: Union[Any, None, Unset] = []
    postprocessing_disable_in_extras: Union[Any, None, Unset] = []
    postprocessing_operation_order: Union[Any, None, Unset] = []
    upscaling_max_images_in_cache: Union[None, Unset, float] = 5.0
    postprocessing_existing_caption_action: Union[None, Unset, str] = 'Ignore'
    disabled_extensions: Union[Any, None, Unset] = []
    disable_all_extensions: Union[None, Unset, str] = 'none'
    restore_config_state_file: Union[Any, None, Unset] = ''
    sd_checkpoint_hash: Union[Any, None, Unset] = ''
    forge_unet_storage_dtype: Union[None, Unset, str] = 'Automatic'
    forge_inference_memory: Union[None, Unset, float] = 1024.0
    forge_async_loading: Union[None, Unset, str] = 'Queue'
    forge_pin_shared_memory: Union[None, Unset, str] = 'CPU'
    forge_preset: Union[None, Unset, str] = 'sd'
    forge_additional_modules: Union[Any, None, Unset] = []
    settings_in_ui: Union[None, Unset, str] = 'This page allows you to add some settings to the main interface of txt2img and img2img tabs.'
    extra_options_txt2img: Union[Any, None, Unset] = []
    extra_options_img2img: Union[Any, None, Unset] = []
    extra_options_cols: Union[None, Unset, float] = 1.0
    extra_options_accordion: Union[Any, None, Unset] = False
    additional_properties: Dict[str, Any] = _attrs_field(init=False, factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        samples_save: Union[None, Unset, bool]
        if isinstance(self.samples_save, Unset):
            samples_save = UNSET
        else:
            samples_save = self.samples_save

        samples_format: Union[None, Unset, str]
        if isinstance(self.samples_format, Unset):
            samples_format = UNSET
        else:
            samples_format = self.samples_format

        samples_filename_pattern: Union[Any, None, Unset]
        if isinstance(self.samples_filename_pattern, Unset):
            samples_filename_pattern = UNSET
        else:
            samples_filename_pattern = self.samples_filename_pattern

        save_images_add_number: Union[None, Unset, bool]
        if isinstance(self.save_images_add_number, Unset):
            save_images_add_number = UNSET
        else:
            save_images_add_number = self.save_images_add_number

        save_images_replace_action: Union[None, Unset, str]
        if isinstance(self.save_images_replace_action, Unset):
            save_images_replace_action = UNSET
        else:
            save_images_replace_action = self.save_images_replace_action

        grid_save: Union[None, Unset, bool]
        if isinstance(self.grid_save, Unset):
            grid_save = UNSET
        else:
            grid_save = self.grid_save

        grid_format: Union[None, Unset, str]
        if isinstance(self.grid_format, Unset):
            grid_format = UNSET
        else:
            grid_format = self.grid_format

        grid_extended_filename: Union[Any, None, Unset]
        if isinstance(self.grid_extended_filename, Unset):
            grid_extended_filename = UNSET
        else:
            grid_extended_filename = self.grid_extended_filename

        grid_only_if_multiple: Union[None, Unset, bool]
        if isinstance(self.grid_only_if_multiple, Unset):
            grid_only_if_multiple = UNSET
        else:
            grid_only_if_multiple = self.grid_only_if_multiple

        grid_prevent_empty_spots: Union[Any, None, Unset]
        if isinstance(self.grid_prevent_empty_spots, Unset):
            grid_prevent_empty_spots = UNSET
        else:
            grid_prevent_empty_spots = self.grid_prevent_empty_spots

        grid_zip_filename_pattern: Union[Any, None, Unset]
        if isinstance(self.grid_zip_filename_pattern, Unset):
            grid_zip_filename_pattern = UNSET
        else:
            grid_zip_filename_pattern = self.grid_zip_filename_pattern

        n_rows: Union[None, Unset, float]
        if isinstance(self.n_rows, Unset):
            n_rows = UNSET
        else:
            n_rows = self.n_rows

        font: Union[Any, None, Unset]
        if isinstance(self.font, Unset):
            font = UNSET
        else:
            font = self.font

        grid_text_active_color: Union[None, Unset, str]
        if isinstance(self.grid_text_active_color, Unset):
            grid_text_active_color = UNSET
        else:
            grid_text_active_color = self.grid_text_active_color

        grid_text_inactive_color: Union[None, Unset, str]
        if isinstance(self.grid_text_inactive_color, Unset):
            grid_text_inactive_color = UNSET
        else:
            grid_text_inactive_color = self.grid_text_inactive_color

        grid_background_color: Union[None, Unset, str]
        if isinstance(self.grid_background_color, Unset):
            grid_background_color = UNSET
        else:
            grid_background_color = self.grid_background_color

        save_images_before_face_restoration: Union[Any, None, Unset]
        if isinstance(self.save_images_before_face_restoration, Unset):
            save_images_before_face_restoration = UNSET
        else:
            save_images_before_face_restoration = self.save_images_before_face_restoration

        save_images_before_highres_fix: Union[Any, None, Unset]
        if isinstance(self.save_images_before_highres_fix, Unset):
            save_images_before_highres_fix = UNSET
        else:
            save_images_before_highres_fix = self.save_images_before_highres_fix

        save_images_before_color_correction: Union[Any, None, Unset]
        if isinstance(self.save_images_before_color_correction, Unset):
            save_images_before_color_correction = UNSET
        else:
            save_images_before_color_correction = self.save_images_before_color_correction

        save_mask: Union[Any, None, Unset]
        if isinstance(self.save_mask, Unset):
            save_mask = UNSET
        else:
            save_mask = self.save_mask

        save_mask_composite: Union[Any, None, Unset]
        if isinstance(self.save_mask_composite, Unset):
            save_mask_composite = UNSET
        else:
            save_mask_composite = self.save_mask_composite

        jpeg_quality: Union[None, Unset, float]
        if isinstance(self.jpeg_quality, Unset):
            jpeg_quality = UNSET
        else:
            jpeg_quality = self.jpeg_quality

        webp_lossless: Union[Any, None, Unset]
        if isinstance(self.webp_lossless, Unset):
            webp_lossless = UNSET
        else:
            webp_lossless = self.webp_lossless

        export_for_4chan: Union[None, Unset, bool]
        if isinstance(self.export_for_4chan, Unset):
            export_for_4chan = UNSET
        else:
            export_for_4chan = self.export_for_4chan

        img_downscale_threshold: Union[None, Unset, float]
        if isinstance(self.img_downscale_threshold, Unset):
            img_downscale_threshold = UNSET
        else:
            img_downscale_threshold = self.img_downscale_threshold

        target_side_length: Union[None, Unset, float]
        if isinstance(self.target_side_length, Unset):
            target_side_length = UNSET
        else:
            target_side_length = self.target_side_length

        img_max_size_mp: Union[None, Unset, float]
        if isinstance(self.img_max_size_mp, Unset):
            img_max_size_mp = UNSET
        else:
            img_max_size_mp = self.img_max_size_mp

        use_original_name_batch: Union[None, Unset, bool]
        if isinstance(self.use_original_name_batch, Unset):
            use_original_name_batch = UNSET
        else:
            use_original_name_batch = self.use_original_name_batch

        use_upscaler_name_as_suffix: Union[Any, None, Unset]
        if isinstance(self.use_upscaler_name_as_suffix, Unset):
            use_upscaler_name_as_suffix = UNSET
        else:
            use_upscaler_name_as_suffix = self.use_upscaler_name_as_suffix

        save_selected_only: Union[None, Unset, bool]
        if isinstance(self.save_selected_only, Unset):
            save_selected_only = UNSET
        else:
            save_selected_only = self.save_selected_only

        save_write_log_csv: Union[None, Unset, bool]
        if isinstance(self.save_write_log_csv, Unset):
            save_write_log_csv = UNSET
        else:
            save_write_log_csv = self.save_write_log_csv

        save_init_img: Union[Any, None, Unset]
        if isinstance(self.save_init_img, Unset):
            save_init_img = UNSET
        else:
            save_init_img = self.save_init_img

        temp_dir: Union[Any, None, Unset]
        if isinstance(self.temp_dir, Unset):
            temp_dir = UNSET
        else:
            temp_dir = self.temp_dir

        clean_temp_dir_at_start: Union[Any, None, Unset]
        if isinstance(self.clean_temp_dir_at_start, Unset):
            clean_temp_dir_at_start = UNSET
        else:
            clean_temp_dir_at_start = self.clean_temp_dir_at_start

        save_incomplete_images: Union[Any, None, Unset]
        if isinstance(self.save_incomplete_images, Unset):
            save_incomplete_images = UNSET
        else:
            save_incomplete_images = self.save_incomplete_images

        notification_audio: Union[None, Unset, bool]
        if isinstance(self.notification_audio, Unset):
            notification_audio = UNSET
        else:
            notification_audio = self.notification_audio

        notification_volume: Union[None, Unset, float]
        if isinstance(self.notification_volume, Unset):
            notification_volume = UNSET
        else:
            notification_volume = self.notification_volume

        outdir_samples: Union[Any, None, Unset]
        if isinstance(self.outdir_samples, Unset):
            outdir_samples = UNSET
        else:
            outdir_samples = self.outdir_samples

        outdir_txt2img_samples: Union[None, Unset, str]
        if isinstance(self.outdir_txt2img_samples, Unset):
            outdir_txt2img_samples = UNSET
        else:
            outdir_txt2img_samples = self.outdir_txt2img_samples

        outdir_img2img_samples: Union[None, Unset, str]
        if isinstance(self.outdir_img2img_samples, Unset):
            outdir_img2img_samples = UNSET
        else:
            outdir_img2img_samples = self.outdir_img2img_samples

        outdir_extras_samples: Union[None, Unset, str]
        if isinstance(self.outdir_extras_samples, Unset):
            outdir_extras_samples = UNSET
        else:
            outdir_extras_samples = self.outdir_extras_samples

        outdir_grids: Union[Any, None, Unset]
        if isinstance(self.outdir_grids, Unset):
            outdir_grids = UNSET
        else:
            outdir_grids = self.outdir_grids

        outdir_txt2img_grids: Union[None, Unset, str]
        if isinstance(self.outdir_txt2img_grids, Unset):
            outdir_txt2img_grids = UNSET
        else:
            outdir_txt2img_grids = self.outdir_txt2img_grids

        outdir_img2img_grids: Union[None, Unset, str]
        if isinstance(self.outdir_img2img_grids, Unset):
            outdir_img2img_grids = UNSET
        else:
            outdir_img2img_grids = self.outdir_img2img_grids

        outdir_save: Union[None, Unset, str]
        if isinstance(self.outdir_save, Unset):
            outdir_save = UNSET
        else:
            outdir_save = self.outdir_save

        outdir_init_images: Union[None, Unset, str]
        if isinstance(self.outdir_init_images, Unset):
            outdir_init_images = UNSET
        else:
            outdir_init_images = self.outdir_init_images

        save_to_dirs: Union[None, Unset, bool]
        if isinstance(self.save_to_dirs, Unset):
            save_to_dirs = UNSET
        else:
            save_to_dirs = self.save_to_dirs

        grid_save_to_dirs: Union[None, Unset, bool]
        if isinstance(self.grid_save_to_dirs, Unset):
            grid_save_to_dirs = UNSET
        else:
            grid_save_to_dirs = self.grid_save_to_dirs

        use_save_to_dirs_for_ui: Union[Any, None, Unset]
        if isinstance(self.use_save_to_dirs_for_ui, Unset):
            use_save_to_dirs_for_ui = UNSET
        else:
            use_save_to_dirs_for_ui = self.use_save_to_dirs_for_ui

        directories_filename_pattern: Union[None, Unset, str]
        if isinstance(self.directories_filename_pattern, Unset):
            directories_filename_pattern = UNSET
        else:
            directories_filename_pattern = self.directories_filename_pattern

        directories_max_prompt_words: Union[None, Unset, float]
        if isinstance(self.directories_max_prompt_words, Unset):
            directories_max_prompt_words = UNSET
        else:
            directories_max_prompt_words = self.directories_max_prompt_words

        esrgan_tile: Union[None, Unset, float]
        if isinstance(self.esrgan_tile, Unset):
            esrgan_tile = UNSET
        else:
            esrgan_tile = self.esrgan_tile

        esrgan_tile_overlap: Union[None, Unset, float]
        if isinstance(self.esrgan_tile_overlap, Unset):
            esrgan_tile_overlap = UNSET
        else:
            esrgan_tile_overlap = self.esrgan_tile_overlap

        realesrgan_enabled_models: Union[List[Any], None, Unset]
        if isinstance(self.realesrgan_enabled_models, Unset):
            realesrgan_enabled_models = UNSET
        elif isinstance(self.realesrgan_enabled_models, list):
            realesrgan_enabled_models = self.realesrgan_enabled_models


        else:
            realesrgan_enabled_models = self.realesrgan_enabled_models

        dat_enabled_models: Union[List[Any], None, Unset]
        if isinstance(self.dat_enabled_models, Unset):
            dat_enabled_models = UNSET
        elif isinstance(self.dat_enabled_models, list):
            dat_enabled_models = self.dat_enabled_models


        else:
            dat_enabled_models = self.dat_enabled_models

        dat_tile: Union[None, Unset, float]
        if isinstance(self.dat_tile, Unset):
            dat_tile = UNSET
        else:
            dat_tile = self.dat_tile

        dat_tile_overlap: Union[None, Unset, float]
        if isinstance(self.dat_tile_overlap, Unset):
            dat_tile_overlap = UNSET
        else:
            dat_tile_overlap = self.dat_tile_overlap

        upscaler_for_img2img: Union[Any, None, Unset]
        if isinstance(self.upscaler_for_img2img, Unset):
            upscaler_for_img2img = UNSET
        else:
            upscaler_for_img2img = self.upscaler_for_img2img

        set_scale_by_when_changing_upscaler: Union[Any, None, Unset]
        if isinstance(self.set_scale_by_when_changing_upscaler, Unset):
            set_scale_by_when_changing_upscaler = UNSET
        else:
            set_scale_by_when_changing_upscaler = self.set_scale_by_when_changing_upscaler

        face_restoration: Union[Any, None, Unset]
        if isinstance(self.face_restoration, Unset):
            face_restoration = UNSET
        else:
            face_restoration = self.face_restoration

        face_restoration_model: Union[None, Unset, str]
        if isinstance(self.face_restoration_model, Unset):
            face_restoration_model = UNSET
        else:
            face_restoration_model = self.face_restoration_model

        code_former_weight: Union[None, Unset, float]
        if isinstance(self.code_former_weight, Unset):
            code_former_weight = UNSET
        else:
            code_former_weight = self.code_former_weight

        face_restoration_unload: Union[Any, None, Unset]
        if isinstance(self.face_restoration_unload, Unset):
            face_restoration_unload = UNSET
        else:
            face_restoration_unload = self.face_restoration_unload

        auto_launch_browser: Union[None, Unset, str]
        if isinstance(self.auto_launch_browser, Unset):
            auto_launch_browser = UNSET
        else:
            auto_launch_browser = self.auto_launch_browser

        enable_console_prompts: Union[Any, None, Unset]
        if isinstance(self.enable_console_prompts, Unset):
            enable_console_prompts = UNSET
        else:
            enable_console_prompts = self.enable_console_prompts

        show_warnings: Union[Any, None, Unset]
        if isinstance(self.show_warnings, Unset):
            show_warnings = UNSET
        else:
            show_warnings = self.show_warnings

        show_gradio_deprecation_warnings: Union[None, Unset, bool]
        if isinstance(self.show_gradio_deprecation_warnings, Unset):
            show_gradio_deprecation_warnings = UNSET
        else:
            show_gradio_deprecation_warnings = self.show_gradio_deprecation_warnings

        memmon_poll_rate: Union[None, Unset, float]
        if isinstance(self.memmon_poll_rate, Unset):
            memmon_poll_rate = UNSET
        else:
            memmon_poll_rate = self.memmon_poll_rate

        samples_log_stdout: Union[Any, None, Unset]
        if isinstance(self.samples_log_stdout, Unset):
            samples_log_stdout = UNSET
        else:
            samples_log_stdout = self.samples_log_stdout

        multiple_tqdm: Union[None, Unset, bool]
        if isinstance(self.multiple_tqdm, Unset):
            multiple_tqdm = UNSET
        else:
            multiple_tqdm = self.multiple_tqdm

        enable_upscale_progressbar: Union[None, Unset, bool]
        if isinstance(self.enable_upscale_progressbar, Unset):
            enable_upscale_progressbar = UNSET
        else:
            enable_upscale_progressbar = self.enable_upscale_progressbar

        print_hypernet_extra: Union[Any, None, Unset]
        if isinstance(self.print_hypernet_extra, Unset):
            print_hypernet_extra = UNSET
        else:
            print_hypernet_extra = self.print_hypernet_extra

        list_hidden_files: Union[None, Unset, bool]
        if isinstance(self.list_hidden_files, Unset):
            list_hidden_files = UNSET
        else:
            list_hidden_files = self.list_hidden_files

        disable_mmap_load_safetensors: Union[Any, None, Unset]
        if isinstance(self.disable_mmap_load_safetensors, Unset):
            disable_mmap_load_safetensors = UNSET
        else:
            disable_mmap_load_safetensors = self.disable_mmap_load_safetensors

        hide_ldm_prints: Union[None, Unset, bool]
        if isinstance(self.hide_ldm_prints, Unset):
            hide_ldm_prints = UNSET
        else:
            hide_ldm_prints = self.hide_ldm_prints

        dump_stacks_on_signal: Union[Any, None, Unset]
        if isinstance(self.dump_stacks_on_signal, Unset):
            dump_stacks_on_signal = UNSET
        else:
            dump_stacks_on_signal = self.dump_stacks_on_signal

        profiling_explanation: Union[None, Unset, str]
        if isinstance(self.profiling_explanation, Unset):
            profiling_explanation = UNSET
        else:
            profiling_explanation = self.profiling_explanation

        profiling_enable: Union[Any, None, Unset]
        if isinstance(self.profiling_enable, Unset):
            profiling_enable = UNSET
        else:
            profiling_enable = self.profiling_enable

        profiling_activities: Union[List[Any], None, Unset]
        if isinstance(self.profiling_activities, Unset):
            profiling_activities = UNSET
        elif isinstance(self.profiling_activities, list):
            profiling_activities = self.profiling_activities


        else:
            profiling_activities = self.profiling_activities

        profiling_record_shapes: Union[None, Unset, bool]
        if isinstance(self.profiling_record_shapes, Unset):
            profiling_record_shapes = UNSET
        else:
            profiling_record_shapes = self.profiling_record_shapes

        profiling_profile_memory: Union[None, Unset, bool]
        if isinstance(self.profiling_profile_memory, Unset):
            profiling_profile_memory = UNSET
        else:
            profiling_profile_memory = self.profiling_profile_memory

        profiling_with_stack: Union[None, Unset, bool]
        if isinstance(self.profiling_with_stack, Unset):
            profiling_with_stack = UNSET
        else:
            profiling_with_stack = self.profiling_with_stack

        profiling_filename: Union[None, Unset, str]
        if isinstance(self.profiling_filename, Unset):
            profiling_filename = UNSET
        else:
            profiling_filename = self.profiling_filename

        api_enable_requests: Union[None, Unset, bool]
        if isinstance(self.api_enable_requests, Unset):
            api_enable_requests = UNSET
        else:
            api_enable_requests = self.api_enable_requests

        api_forbid_local_requests: Union[None, Unset, bool]
        if isinstance(self.api_forbid_local_requests, Unset):
            api_forbid_local_requests = UNSET
        else:
            api_forbid_local_requests = self.api_forbid_local_requests

        api_useragent: Union[Any, None, Unset]
        if isinstance(self.api_useragent, Unset):
            api_useragent = UNSET
        else:
            api_useragent = self.api_useragent

        unload_models_when_training: Union[Any, None, Unset]
        if isinstance(self.unload_models_when_training, Unset):
            unload_models_when_training = UNSET
        else:
            unload_models_when_training = self.unload_models_when_training

        pin_memory: Union[Any, None, Unset]
        if isinstance(self.pin_memory, Unset):
            pin_memory = UNSET
        else:
            pin_memory = self.pin_memory

        save_optimizer_state: Union[Any, None, Unset]
        if isinstance(self.save_optimizer_state, Unset):
            save_optimizer_state = UNSET
        else:
            save_optimizer_state = self.save_optimizer_state

        save_training_settings_to_txt: Union[None, Unset, bool]
        if isinstance(self.save_training_settings_to_txt, Unset):
            save_training_settings_to_txt = UNSET
        else:
            save_training_settings_to_txt = self.save_training_settings_to_txt

        dataset_filename_word_regex: Union[Any, None, Unset]
        if isinstance(self.dataset_filename_word_regex, Unset):
            dataset_filename_word_regex = UNSET
        else:
            dataset_filename_word_regex = self.dataset_filename_word_regex

        dataset_filename_join_string: Union[None, Unset, str]
        if isinstance(self.dataset_filename_join_string, Unset):
            dataset_filename_join_string = UNSET
        else:
            dataset_filename_join_string = self.dataset_filename_join_string

        training_image_repeats_per_epoch: Union[None, Unset, float]
        if isinstance(self.training_image_repeats_per_epoch, Unset):
            training_image_repeats_per_epoch = UNSET
        else:
            training_image_repeats_per_epoch = self.training_image_repeats_per_epoch

        training_write_csv_every: Union[None, Unset, float]
        if isinstance(self.training_write_csv_every, Unset):
            training_write_csv_every = UNSET
        else:
            training_write_csv_every = self.training_write_csv_every

        training_xattention_optimizations: Union[Any, None, Unset]
        if isinstance(self.training_xattention_optimizations, Unset):
            training_xattention_optimizations = UNSET
        else:
            training_xattention_optimizations = self.training_xattention_optimizations

        training_enable_tensorboard: Union[Any, None, Unset]
        if isinstance(self.training_enable_tensorboard, Unset):
            training_enable_tensorboard = UNSET
        else:
            training_enable_tensorboard = self.training_enable_tensorboard

        training_tensorboard_save_images: Union[Any, None, Unset]
        if isinstance(self.training_tensorboard_save_images, Unset):
            training_tensorboard_save_images = UNSET
        else:
            training_tensorboard_save_images = self.training_tensorboard_save_images

        training_tensorboard_flush_every: Union[None, Unset, float]
        if isinstance(self.training_tensorboard_flush_every, Unset):
            training_tensorboard_flush_every = UNSET
        else:
            training_tensorboard_flush_every = self.training_tensorboard_flush_every

        sd_model_checkpoint: Union[Any, None, Unset]
        if isinstance(self.sd_model_checkpoint, Unset):
            sd_model_checkpoint = UNSET
        else:
            sd_model_checkpoint = self.sd_model_checkpoint

        sd_checkpoints_limit: Union[None, Unset, float]
        if isinstance(self.sd_checkpoints_limit, Unset):
            sd_checkpoints_limit = UNSET
        else:
            sd_checkpoints_limit = self.sd_checkpoints_limit

        sd_checkpoints_keep_in_cpu: Union[None, Unset, bool]
        if isinstance(self.sd_checkpoints_keep_in_cpu, Unset):
            sd_checkpoints_keep_in_cpu = UNSET
        else:
            sd_checkpoints_keep_in_cpu = self.sd_checkpoints_keep_in_cpu

        sd_checkpoint_cache: Union[Any, None, Unset]
        if isinstance(self.sd_checkpoint_cache, Unset):
            sd_checkpoint_cache = UNSET
        else:
            sd_checkpoint_cache = self.sd_checkpoint_cache

        sd_unet: Union[None, Unset, str]
        if isinstance(self.sd_unet, Unset):
            sd_unet = UNSET
        else:
            sd_unet = self.sd_unet

        enable_quantization: Union[Any, None, Unset]
        if isinstance(self.enable_quantization, Unset):
            enable_quantization = UNSET
        else:
            enable_quantization = self.enable_quantization

        emphasis: Union[None, Unset, str]
        if isinstance(self.emphasis, Unset):
            emphasis = UNSET
        else:
            emphasis = self.emphasis

        enable_batch_seeds: Union[None, Unset, bool]
        if isinstance(self.enable_batch_seeds, Unset):
            enable_batch_seeds = UNSET
        else:
            enable_batch_seeds = self.enable_batch_seeds

        comma_padding_backtrack: Union[None, Unset, float]
        if isinstance(self.comma_padding_backtrack, Unset):
            comma_padding_backtrack = UNSET
        else:
            comma_padding_backtrack = self.comma_padding_backtrack

        sdxl_clip_l_skip: Union[Any, None, Unset]
        if isinstance(self.sdxl_clip_l_skip, Unset):
            sdxl_clip_l_skip = UNSET
        else:
            sdxl_clip_l_skip = self.sdxl_clip_l_skip

        clip_stop_at_last_layers: Union[None, Unset, float]
        if isinstance(self.clip_stop_at_last_layers, Unset):
            clip_stop_at_last_layers = UNSET
        else:
            clip_stop_at_last_layers = self.clip_stop_at_last_layers

        upcast_attn: Union[Any, None, Unset]
        if isinstance(self.upcast_attn, Unset):
            upcast_attn = UNSET
        else:
            upcast_attn = self.upcast_attn

        randn_source: Union[None, Unset, str]
        if isinstance(self.randn_source, Unset):
            randn_source = UNSET
        else:
            randn_source = self.randn_source

        tiling: Union[Any, None, Unset]
        if isinstance(self.tiling, Unset):
            tiling = UNSET
        else:
            tiling = self.tiling

        hires_fix_refiner_pass: Union[None, Unset, str]
        if isinstance(self.hires_fix_refiner_pass, Unset):
            hires_fix_refiner_pass = UNSET
        else:
            hires_fix_refiner_pass = self.hires_fix_refiner_pass

        sdxl_crop_top: Union[Any, None, Unset]
        if isinstance(self.sdxl_crop_top, Unset):
            sdxl_crop_top = UNSET
        else:
            sdxl_crop_top = self.sdxl_crop_top

        sdxl_crop_left: Union[Any, None, Unset]
        if isinstance(self.sdxl_crop_left, Unset):
            sdxl_crop_left = UNSET
        else:
            sdxl_crop_left = self.sdxl_crop_left

        sdxl_refiner_low_aesthetic_score: Union[None, Unset, float]
        if isinstance(self.sdxl_refiner_low_aesthetic_score, Unset):
            sdxl_refiner_low_aesthetic_score = UNSET
        else:
            sdxl_refiner_low_aesthetic_score = self.sdxl_refiner_low_aesthetic_score

        sdxl_refiner_high_aesthetic_score: Union[None, Unset, float]
        if isinstance(self.sdxl_refiner_high_aesthetic_score, Unset):
            sdxl_refiner_high_aesthetic_score = UNSET
        else:
            sdxl_refiner_high_aesthetic_score = self.sdxl_refiner_high_aesthetic_score

        sd3_enable_t5: Union[Any, None, Unset]
        if isinstance(self.sd3_enable_t5, Unset):
            sd3_enable_t5 = UNSET
        else:
            sd3_enable_t5 = self.sd3_enable_t5

        sd_vae_explanation: Union[None, Unset, str]
        if isinstance(self.sd_vae_explanation, Unset):
            sd_vae_explanation = UNSET
        else:
            sd_vae_explanation = self.sd_vae_explanation

        sd_vae_checkpoint_cache: Union[Any, None, Unset]
        if isinstance(self.sd_vae_checkpoint_cache, Unset):
            sd_vae_checkpoint_cache = UNSET
        else:
            sd_vae_checkpoint_cache = self.sd_vae_checkpoint_cache

        sd_vae: Union[None, Unset, str]
        if isinstance(self.sd_vae, Unset):
            sd_vae = UNSET
        else:
            sd_vae = self.sd_vae

        sd_vae_overrides_per_model_preferences: Union[None, Unset, bool]
        if isinstance(self.sd_vae_overrides_per_model_preferences, Unset):
            sd_vae_overrides_per_model_preferences = UNSET
        else:
            sd_vae_overrides_per_model_preferences = self.sd_vae_overrides_per_model_preferences

        auto_vae_precision_bfloat16: Union[Any, None, Unset]
        if isinstance(self.auto_vae_precision_bfloat16, Unset):
            auto_vae_precision_bfloat16 = UNSET
        else:
            auto_vae_precision_bfloat16 = self.auto_vae_precision_bfloat16

        auto_vae_precision: Union[None, Unset, bool]
        if isinstance(self.auto_vae_precision, Unset):
            auto_vae_precision = UNSET
        else:
            auto_vae_precision = self.auto_vae_precision

        sd_vae_encode_method: Union[None, Unset, str]
        if isinstance(self.sd_vae_encode_method, Unset):
            sd_vae_encode_method = UNSET
        else:
            sd_vae_encode_method = self.sd_vae_encode_method

        sd_vae_decode_method: Union[None, Unset, str]
        if isinstance(self.sd_vae_decode_method, Unset):
            sd_vae_decode_method = UNSET
        else:
            sd_vae_decode_method = self.sd_vae_decode_method

        inpainting_mask_weight: Union[None, Unset, float]
        if isinstance(self.inpainting_mask_weight, Unset):
            inpainting_mask_weight = UNSET
        else:
            inpainting_mask_weight = self.inpainting_mask_weight

        initial_noise_multiplier: Union[None, Unset, float]
        if isinstance(self.initial_noise_multiplier, Unset):
            initial_noise_multiplier = UNSET
        else:
            initial_noise_multiplier = self.initial_noise_multiplier

        img2img_extra_noise: Union[Any, None, Unset]
        if isinstance(self.img2img_extra_noise, Unset):
            img2img_extra_noise = UNSET
        else:
            img2img_extra_noise = self.img2img_extra_noise

        img2img_color_correction: Union[Any, None, Unset]
        if isinstance(self.img2img_color_correction, Unset):
            img2img_color_correction = UNSET
        else:
            img2img_color_correction = self.img2img_color_correction

        img2img_fix_steps: Union[Any, None, Unset]
        if isinstance(self.img2img_fix_steps, Unset):
            img2img_fix_steps = UNSET
        else:
            img2img_fix_steps = self.img2img_fix_steps

        img2img_background_color: Union[None, Unset, str]
        if isinstance(self.img2img_background_color, Unset):
            img2img_background_color = UNSET
        else:
            img2img_background_color = self.img2img_background_color

        img2img_sketch_default_brush_color: Union[None, Unset, str]
        if isinstance(self.img2img_sketch_default_brush_color, Unset):
            img2img_sketch_default_brush_color = UNSET
        else:
            img2img_sketch_default_brush_color = self.img2img_sketch_default_brush_color

        img2img_inpaint_mask_brush_color: Union[None, Unset, str]
        if isinstance(self.img2img_inpaint_mask_brush_color, Unset):
            img2img_inpaint_mask_brush_color = UNSET
        else:
            img2img_inpaint_mask_brush_color = self.img2img_inpaint_mask_brush_color

        img2img_inpaint_sketch_default_brush_color: Union[None, Unset, str]
        if isinstance(self.img2img_inpaint_sketch_default_brush_color, Unset):
            img2img_inpaint_sketch_default_brush_color = UNSET
        else:
            img2img_inpaint_sketch_default_brush_color = self.img2img_inpaint_sketch_default_brush_color

        img2img_inpaint_mask_high_contrast: Union[None, Unset, bool]
        if isinstance(self.img2img_inpaint_mask_high_contrast, Unset):
            img2img_inpaint_mask_high_contrast = UNSET
        else:
            img2img_inpaint_mask_high_contrast = self.img2img_inpaint_mask_high_contrast

        return_mask: Union[Any, None, Unset]
        if isinstance(self.return_mask, Unset):
            return_mask = UNSET
        else:
            return_mask = self.return_mask

        return_mask_composite: Union[Any, None, Unset]
        if isinstance(self.return_mask_composite, Unset):
            return_mask_composite = UNSET
        else:
            return_mask_composite = self.return_mask_composite

        img2img_batch_show_results_limit: Union[None, Unset, float]
        if isinstance(self.img2img_batch_show_results_limit, Unset):
            img2img_batch_show_results_limit = UNSET
        else:
            img2img_batch_show_results_limit = self.img2img_batch_show_results_limit

        overlay_inpaint: Union[None, Unset, bool]
        if isinstance(self.overlay_inpaint, Unset):
            overlay_inpaint = UNSET
        else:
            overlay_inpaint = self.overlay_inpaint

        cross_attention_optimization: Union[None, Unset, str]
        if isinstance(self.cross_attention_optimization, Unset):
            cross_attention_optimization = UNSET
        else:
            cross_attention_optimization = self.cross_attention_optimization

        s_min_uncond: Union[Any, None, Unset]
        if isinstance(self.s_min_uncond, Unset):
            s_min_uncond = UNSET
        else:
            s_min_uncond = self.s_min_uncond

        s_min_uncond_all: Union[Any, None, Unset]
        if isinstance(self.s_min_uncond_all, Unset):
            s_min_uncond_all = UNSET
        else:
            s_min_uncond_all = self.s_min_uncond_all

        token_merging_ratio: Union[Any, None, Unset]
        if isinstance(self.token_merging_ratio, Unset):
            token_merging_ratio = UNSET
        else:
            token_merging_ratio = self.token_merging_ratio

        token_merging_ratio_img2img: Union[Any, None, Unset]
        if isinstance(self.token_merging_ratio_img2img, Unset):
            token_merging_ratio_img2img = UNSET
        else:
            token_merging_ratio_img2img = self.token_merging_ratio_img2img

        token_merging_ratio_hr: Union[Any, None, Unset]
        if isinstance(self.token_merging_ratio_hr, Unset):
            token_merging_ratio_hr = UNSET
        else:
            token_merging_ratio_hr = self.token_merging_ratio_hr

        pad_cond_uncond: Union[Any, None, Unset]
        if isinstance(self.pad_cond_uncond, Unset):
            pad_cond_uncond = UNSET
        else:
            pad_cond_uncond = self.pad_cond_uncond

        pad_cond_uncond_v0: Union[Any, None, Unset]
        if isinstance(self.pad_cond_uncond_v0, Unset):
            pad_cond_uncond_v0 = UNSET
        else:
            pad_cond_uncond_v0 = self.pad_cond_uncond_v0

        persistent_cond_cache: Union[None, Unset, bool]
        if isinstance(self.persistent_cond_cache, Unset):
            persistent_cond_cache = UNSET
        else:
            persistent_cond_cache = self.persistent_cond_cache

        batch_cond_uncond: Union[None, Unset, bool]
        if isinstance(self.batch_cond_uncond, Unset):
            batch_cond_uncond = UNSET
        else:
            batch_cond_uncond = self.batch_cond_uncond

        fp8_storage: Union[None, Unset, str]
        if isinstance(self.fp8_storage, Unset):
            fp8_storage = UNSET
        else:
            fp8_storage = self.fp8_storage

        cache_fp16_weight: Union[Any, None, Unset]
        if isinstance(self.cache_fp16_weight, Unset):
            cache_fp16_weight = UNSET
        else:
            cache_fp16_weight = self.cache_fp16_weight

        forge_try_reproduce: Union[None, Unset, str]
        if isinstance(self.forge_try_reproduce, Unset):
            forge_try_reproduce = UNSET
        else:
            forge_try_reproduce = self.forge_try_reproduce

        auto_backcompat: Union[None, Unset, bool]
        if isinstance(self.auto_backcompat, Unset):
            auto_backcompat = UNSET
        else:
            auto_backcompat = self.auto_backcompat

        use_old_emphasis_implementation: Union[Any, None, Unset]
        if isinstance(self.use_old_emphasis_implementation, Unset):
            use_old_emphasis_implementation = UNSET
        else:
            use_old_emphasis_implementation = self.use_old_emphasis_implementation

        use_old_karras_scheduler_sigmas: Union[Any, None, Unset]
        if isinstance(self.use_old_karras_scheduler_sigmas, Unset):
            use_old_karras_scheduler_sigmas = UNSET
        else:
            use_old_karras_scheduler_sigmas = self.use_old_karras_scheduler_sigmas

        no_dpmpp_sde_batch_determinism: Union[Any, None, Unset]
        if isinstance(self.no_dpmpp_sde_batch_determinism, Unset):
            no_dpmpp_sde_batch_determinism = UNSET
        else:
            no_dpmpp_sde_batch_determinism = self.no_dpmpp_sde_batch_determinism

        use_old_hires_fix_width_height: Union[Any, None, Unset]
        if isinstance(self.use_old_hires_fix_width_height, Unset):
            use_old_hires_fix_width_height = UNSET
        else:
            use_old_hires_fix_width_height = self.use_old_hires_fix_width_height

        hires_fix_use_firstpass_conds: Union[Any, None, Unset]
        if isinstance(self.hires_fix_use_firstpass_conds, Unset):
            hires_fix_use_firstpass_conds = UNSET
        else:
            hires_fix_use_firstpass_conds = self.hires_fix_use_firstpass_conds

        use_old_scheduling: Union[Any, None, Unset]
        if isinstance(self.use_old_scheduling, Unset):
            use_old_scheduling = UNSET
        else:
            use_old_scheduling = self.use_old_scheduling

        use_downcasted_alpha_bar: Union[Any, None, Unset]
        if isinstance(self.use_downcasted_alpha_bar, Unset):
            use_downcasted_alpha_bar = UNSET
        else:
            use_downcasted_alpha_bar = self.use_downcasted_alpha_bar

        refiner_switch_by_sample_steps: Union[Any, None, Unset]
        if isinstance(self.refiner_switch_by_sample_steps, Unset):
            refiner_switch_by_sample_steps = UNSET
        else:
            refiner_switch_by_sample_steps = self.refiner_switch_by_sample_steps

        interrogate_keep_models_in_memory: Union[Any, None, Unset]
        if isinstance(self.interrogate_keep_models_in_memory, Unset):
            interrogate_keep_models_in_memory = UNSET
        else:
            interrogate_keep_models_in_memory = self.interrogate_keep_models_in_memory

        interrogate_return_ranks: Union[Any, None, Unset]
        if isinstance(self.interrogate_return_ranks, Unset):
            interrogate_return_ranks = UNSET
        else:
            interrogate_return_ranks = self.interrogate_return_ranks

        interrogate_clip_num_beams: Union[None, Unset, float]
        if isinstance(self.interrogate_clip_num_beams, Unset):
            interrogate_clip_num_beams = UNSET
        else:
            interrogate_clip_num_beams = self.interrogate_clip_num_beams

        interrogate_clip_min_length: Union[None, Unset, float]
        if isinstance(self.interrogate_clip_min_length, Unset):
            interrogate_clip_min_length = UNSET
        else:
            interrogate_clip_min_length = self.interrogate_clip_min_length

        interrogate_clip_max_length: Union[None, Unset, float]
        if isinstance(self.interrogate_clip_max_length, Unset):
            interrogate_clip_max_length = UNSET
        else:
            interrogate_clip_max_length = self.interrogate_clip_max_length

        interrogate_clip_dict_limit: Union[None, Unset, float]
        if isinstance(self.interrogate_clip_dict_limit, Unset):
            interrogate_clip_dict_limit = UNSET
        else:
            interrogate_clip_dict_limit = self.interrogate_clip_dict_limit

        interrogate_clip_skip_categories: Union[Any, None, Unset]
        if isinstance(self.interrogate_clip_skip_categories, Unset):
            interrogate_clip_skip_categories = UNSET
        else:
            interrogate_clip_skip_categories = self.interrogate_clip_skip_categories

        interrogate_deepbooru_score_threshold: Union[None, Unset, float]
        if isinstance(self.interrogate_deepbooru_score_threshold, Unset):
            interrogate_deepbooru_score_threshold = UNSET
        else:
            interrogate_deepbooru_score_threshold = self.interrogate_deepbooru_score_threshold

        deepbooru_sort_alpha: Union[None, Unset, bool]
        if isinstance(self.deepbooru_sort_alpha, Unset):
            deepbooru_sort_alpha = UNSET
        else:
            deepbooru_sort_alpha = self.deepbooru_sort_alpha

        deepbooru_use_spaces: Union[None, Unset, bool]
        if isinstance(self.deepbooru_use_spaces, Unset):
            deepbooru_use_spaces = UNSET
        else:
            deepbooru_use_spaces = self.deepbooru_use_spaces

        deepbooru_escape: Union[None, Unset, bool]
        if isinstance(self.deepbooru_escape, Unset):
            deepbooru_escape = UNSET
        else:
            deepbooru_escape = self.deepbooru_escape

        deepbooru_filter_tags: Union[Any, None, Unset]
        if isinstance(self.deepbooru_filter_tags, Unset):
            deepbooru_filter_tags = UNSET
        else:
            deepbooru_filter_tags = self.deepbooru_filter_tags

        extra_networks_show_hidden_directories: Union[None, Unset, bool]
        if isinstance(self.extra_networks_show_hidden_directories, Unset):
            extra_networks_show_hidden_directories = UNSET
        else:
            extra_networks_show_hidden_directories = self.extra_networks_show_hidden_directories

        extra_networks_dir_button_function: Union[Any, None, Unset]
        if isinstance(self.extra_networks_dir_button_function, Unset):
            extra_networks_dir_button_function = UNSET
        else:
            extra_networks_dir_button_function = self.extra_networks_dir_button_function

        extra_networks_hidden_models: Union[None, Unset, str]
        if isinstance(self.extra_networks_hidden_models, Unset):
            extra_networks_hidden_models = UNSET
        else:
            extra_networks_hidden_models = self.extra_networks_hidden_models

        extra_networks_default_multiplier: Union[None, Unset, float]
        if isinstance(self.extra_networks_default_multiplier, Unset):
            extra_networks_default_multiplier = UNSET
        else:
            extra_networks_default_multiplier = self.extra_networks_default_multiplier

        extra_networks_card_width: Union[Any, None, Unset]
        if isinstance(self.extra_networks_card_width, Unset):
            extra_networks_card_width = UNSET
        else:
            extra_networks_card_width = self.extra_networks_card_width

        extra_networks_card_height: Union[Any, None, Unset]
        if isinstance(self.extra_networks_card_height, Unset):
            extra_networks_card_height = UNSET
        else:
            extra_networks_card_height = self.extra_networks_card_height

        extra_networks_card_text_scale: Union[None, Unset, float]
        if isinstance(self.extra_networks_card_text_scale, Unset):
            extra_networks_card_text_scale = UNSET
        else:
            extra_networks_card_text_scale = self.extra_networks_card_text_scale

        extra_networks_card_show_desc: Union[None, Unset, bool]
        if isinstance(self.extra_networks_card_show_desc, Unset):
            extra_networks_card_show_desc = UNSET
        else:
            extra_networks_card_show_desc = self.extra_networks_card_show_desc

        extra_networks_card_description_is_html: Union[Any, None, Unset]
        if isinstance(self.extra_networks_card_description_is_html, Unset):
            extra_networks_card_description_is_html = UNSET
        else:
            extra_networks_card_description_is_html = self.extra_networks_card_description_is_html

        extra_networks_card_order_field: Union[None, Unset, str]
        if isinstance(self.extra_networks_card_order_field, Unset):
            extra_networks_card_order_field = UNSET
        else:
            extra_networks_card_order_field = self.extra_networks_card_order_field

        extra_networks_card_order: Union[None, Unset, str]
        if isinstance(self.extra_networks_card_order, Unset):
            extra_networks_card_order = UNSET
        else:
            extra_networks_card_order = self.extra_networks_card_order

        extra_networks_tree_view_style: Union[None, Unset, str]
        if isinstance(self.extra_networks_tree_view_style, Unset):
            extra_networks_tree_view_style = UNSET
        else:
            extra_networks_tree_view_style = self.extra_networks_tree_view_style

        extra_networks_tree_view_default_enabled: Union[None, Unset, bool]
        if isinstance(self.extra_networks_tree_view_default_enabled, Unset):
            extra_networks_tree_view_default_enabled = UNSET
        else:
            extra_networks_tree_view_default_enabled = self.extra_networks_tree_view_default_enabled

        extra_networks_tree_view_default_width: Union[None, Unset, float]
        if isinstance(self.extra_networks_tree_view_default_width, Unset):
            extra_networks_tree_view_default_width = UNSET
        else:
            extra_networks_tree_view_default_width = self.extra_networks_tree_view_default_width

        extra_networks_add_text_separator: Union[None, Unset, str]
        if isinstance(self.extra_networks_add_text_separator, Unset):
            extra_networks_add_text_separator = UNSET
        else:
            extra_networks_add_text_separator = self.extra_networks_add_text_separator

        ui_extra_networks_tab_reorder: Union[Any, None, Unset]
        if isinstance(self.ui_extra_networks_tab_reorder, Unset):
            ui_extra_networks_tab_reorder = UNSET
        else:
            ui_extra_networks_tab_reorder = self.ui_extra_networks_tab_reorder

        textual_inversion_print_at_load: Union[Any, None, Unset]
        if isinstance(self.textual_inversion_print_at_load, Unset):
            textual_inversion_print_at_load = UNSET
        else:
            textual_inversion_print_at_load = self.textual_inversion_print_at_load

        textual_inversion_add_hashes_to_infotext: Union[None, Unset, bool]
        if isinstance(self.textual_inversion_add_hashes_to_infotext, Unset):
            textual_inversion_add_hashes_to_infotext = UNSET
        else:
            textual_inversion_add_hashes_to_infotext = self.textual_inversion_add_hashes_to_infotext

        sd_hypernetwork: Union[None, Unset, str]
        if isinstance(self.sd_hypernetwork, Unset):
            sd_hypernetwork = UNSET
        else:
            sd_hypernetwork = self.sd_hypernetwork

        keyedit_precision_attention: Union[None, Unset, float]
        if isinstance(self.keyedit_precision_attention, Unset):
            keyedit_precision_attention = UNSET
        else:
            keyedit_precision_attention = self.keyedit_precision_attention

        keyedit_precision_extra: Union[None, Unset, float]
        if isinstance(self.keyedit_precision_extra, Unset):
            keyedit_precision_extra = UNSET
        else:
            keyedit_precision_extra = self.keyedit_precision_extra

        keyedit_delimiters: Union[None, Unset, str]
        if isinstance(self.keyedit_delimiters, Unset):
            keyedit_delimiters = UNSET
        else:
            keyedit_delimiters = self.keyedit_delimiters

        keyedit_delimiters_whitespace: Union[List[Any], None, Unset]
        if isinstance(self.keyedit_delimiters_whitespace, Unset):
            keyedit_delimiters_whitespace = UNSET
        elif isinstance(self.keyedit_delimiters_whitespace, list):
            keyedit_delimiters_whitespace = self.keyedit_delimiters_whitespace


        else:
            keyedit_delimiters_whitespace = self.keyedit_delimiters_whitespace

        keyedit_move: Union[None, Unset, bool]
        if isinstance(self.keyedit_move, Unset):
            keyedit_move = UNSET
        else:
            keyedit_move = self.keyedit_move

        disable_token_counters: Union[Any, None, Unset]
        if isinstance(self.disable_token_counters, Unset):
            disable_token_counters = UNSET
        else:
            disable_token_counters = self.disable_token_counters

        include_styles_into_token_counters: Union[None, Unset, bool]
        if isinstance(self.include_styles_into_token_counters, Unset):
            include_styles_into_token_counters = UNSET
        else:
            include_styles_into_token_counters = self.include_styles_into_token_counters

        return_grid: Union[None, Unset, bool]
        if isinstance(self.return_grid, Unset):
            return_grid = UNSET
        else:
            return_grid = self.return_grid

        do_not_show_images: Union[Any, None, Unset]
        if isinstance(self.do_not_show_images, Unset):
            do_not_show_images = UNSET
        else:
            do_not_show_images = self.do_not_show_images

        js_modal_lightbox: Union[None, Unset, bool]
        if isinstance(self.js_modal_lightbox, Unset):
            js_modal_lightbox = UNSET
        else:
            js_modal_lightbox = self.js_modal_lightbox

        js_modal_lightbox_initially_zoomed: Union[None, Unset, bool]
        if isinstance(self.js_modal_lightbox_initially_zoomed, Unset):
            js_modal_lightbox_initially_zoomed = UNSET
        else:
            js_modal_lightbox_initially_zoomed = self.js_modal_lightbox_initially_zoomed

        js_modal_lightbox_gamepad: Union[Any, None, Unset]
        if isinstance(self.js_modal_lightbox_gamepad, Unset):
            js_modal_lightbox_gamepad = UNSET
        else:
            js_modal_lightbox_gamepad = self.js_modal_lightbox_gamepad

        js_modal_lightbox_gamepad_repeat: Union[None, Unset, float]
        if isinstance(self.js_modal_lightbox_gamepad_repeat, Unset):
            js_modal_lightbox_gamepad_repeat = UNSET
        else:
            js_modal_lightbox_gamepad_repeat = self.js_modal_lightbox_gamepad_repeat

        sd_webui_modal_lightbox_icon_opacity: Union[None, Unset, float]
        if isinstance(self.sd_webui_modal_lightbox_icon_opacity, Unset):
            sd_webui_modal_lightbox_icon_opacity = UNSET
        else:
            sd_webui_modal_lightbox_icon_opacity = self.sd_webui_modal_lightbox_icon_opacity

        sd_webui_modal_lightbox_toolbar_opacity: Union[None, Unset, float]
        if isinstance(self.sd_webui_modal_lightbox_toolbar_opacity, Unset):
            sd_webui_modal_lightbox_toolbar_opacity = UNSET
        else:
            sd_webui_modal_lightbox_toolbar_opacity = self.sd_webui_modal_lightbox_toolbar_opacity

        gallery_height: Union[Any, None, Unset]
        if isinstance(self.gallery_height, Unset):
            gallery_height = UNSET
        else:
            gallery_height = self.gallery_height

        open_dir_button_choice: Union[None, Unset, str]
        if isinstance(self.open_dir_button_choice, Unset):
            open_dir_button_choice = UNSET
        else:
            open_dir_button_choice = self.open_dir_button_choice

        compact_prompt_box: Union[Any, None, Unset]
        if isinstance(self.compact_prompt_box, Unset):
            compact_prompt_box = UNSET
        else:
            compact_prompt_box = self.compact_prompt_box

        samplers_in_dropdown: Union[None, Unset, bool]
        if isinstance(self.samplers_in_dropdown, Unset):
            samplers_in_dropdown = UNSET
        else:
            samplers_in_dropdown = self.samplers_in_dropdown

        dimensions_and_batch_together: Union[None, Unset, bool]
        if isinstance(self.dimensions_and_batch_together, Unset):
            dimensions_and_batch_together = UNSET
        else:
            dimensions_and_batch_together = self.dimensions_and_batch_together

        sd_checkpoint_dropdown_use_short: Union[Any, None, Unset]
        if isinstance(self.sd_checkpoint_dropdown_use_short, Unset):
            sd_checkpoint_dropdown_use_short = UNSET
        else:
            sd_checkpoint_dropdown_use_short = self.sd_checkpoint_dropdown_use_short

        hires_fix_show_sampler: Union[Any, None, Unset]
        if isinstance(self.hires_fix_show_sampler, Unset):
            hires_fix_show_sampler = UNSET
        else:
            hires_fix_show_sampler = self.hires_fix_show_sampler

        hires_fix_show_prompts: Union[Any, None, Unset]
        if isinstance(self.hires_fix_show_prompts, Unset):
            hires_fix_show_prompts = UNSET
        else:
            hires_fix_show_prompts = self.hires_fix_show_prompts

        txt2img_settings_accordion: Union[Any, None, Unset]
        if isinstance(self.txt2img_settings_accordion, Unset):
            txt2img_settings_accordion = UNSET
        else:
            txt2img_settings_accordion = self.txt2img_settings_accordion

        img2img_settings_accordion: Union[Any, None, Unset]
        if isinstance(self.img2img_settings_accordion, Unset):
            img2img_settings_accordion = UNSET
        else:
            img2img_settings_accordion = self.img2img_settings_accordion

        interrupt_after_current: Union[None, Unset, bool]
        if isinstance(self.interrupt_after_current, Unset):
            interrupt_after_current = UNSET
        else:
            interrupt_after_current = self.interrupt_after_current

        localization: Union[None, Unset, str]
        if isinstance(self.localization, Unset):
            localization = UNSET
        else:
            localization = self.localization

        quick_setting_list: Union[Any, None, Unset]
        if isinstance(self.quick_setting_list, Unset):
            quick_setting_list = UNSET
        else:
            quick_setting_list = self.quick_setting_list

        ui_tab_order: Union[Any, None, Unset]
        if isinstance(self.ui_tab_order, Unset):
            ui_tab_order = UNSET
        else:
            ui_tab_order = self.ui_tab_order

        hidden_tabs: Union[Any, None, Unset]
        if isinstance(self.hidden_tabs, Unset):
            hidden_tabs = UNSET
        else:
            hidden_tabs = self.hidden_tabs

        ui_reorder_list: Union[Any, None, Unset]
        if isinstance(self.ui_reorder_list, Unset):
            ui_reorder_list = UNSET
        else:
            ui_reorder_list = self.ui_reorder_list

        gradio_theme: Union[None, Unset, str]
        if isinstance(self.gradio_theme, Unset):
            gradio_theme = UNSET
        else:
            gradio_theme = self.gradio_theme

        gradio_themes_cache: Union[None, Unset, bool]
        if isinstance(self.gradio_themes_cache, Unset):
            gradio_themes_cache = UNSET
        else:
            gradio_themes_cache = self.gradio_themes_cache

        show_progress_in_title: Union[None, Unset, bool]
        if isinstance(self.show_progress_in_title, Unset):
            show_progress_in_title = UNSET
        else:
            show_progress_in_title = self.show_progress_in_title

        send_seed: Union[None, Unset, bool]
        if isinstance(self.send_seed, Unset):
            send_seed = UNSET
        else:
            send_seed = self.send_seed

        send_size: Union[None, Unset, bool]
        if isinstance(self.send_size, Unset):
            send_size = UNSET
        else:
            send_size = self.send_size

        enable_reloading_ui_scripts: Union[Any, None, Unset]
        if isinstance(self.enable_reloading_ui_scripts, Unset):
            enable_reloading_ui_scripts = UNSET
        else:
            enable_reloading_ui_scripts = self.enable_reloading_ui_scripts

        infotext_explanation: Union[None, Unset, str]
        if isinstance(self.infotext_explanation, Unset):
            infotext_explanation = UNSET
        else:
            infotext_explanation = self.infotext_explanation

        enable_pnginfo: Union[None, Unset, bool]
        if isinstance(self.enable_pnginfo, Unset):
            enable_pnginfo = UNSET
        else:
            enable_pnginfo = self.enable_pnginfo

        save_txt: Union[Any, None, Unset]
        if isinstance(self.save_txt, Unset):
            save_txt = UNSET
        else:
            save_txt = self.save_txt

        add_model_name_to_info: Union[None, Unset, bool]
        if isinstance(self.add_model_name_to_info, Unset):
            add_model_name_to_info = UNSET
        else:
            add_model_name_to_info = self.add_model_name_to_info

        add_model_hash_to_info: Union[None, Unset, bool]
        if isinstance(self.add_model_hash_to_info, Unset):
            add_model_hash_to_info = UNSET
        else:
            add_model_hash_to_info = self.add_model_hash_to_info

        add_vae_name_to_info: Union[None, Unset, bool]
        if isinstance(self.add_vae_name_to_info, Unset):
            add_vae_name_to_info = UNSET
        else:
            add_vae_name_to_info = self.add_vae_name_to_info

        add_vae_hash_to_info: Union[None, Unset, bool]
        if isinstance(self.add_vae_hash_to_info, Unset):
            add_vae_hash_to_info = UNSET
        else:
            add_vae_hash_to_info = self.add_vae_hash_to_info

        add_user_name_to_info: Union[Any, None, Unset]
        if isinstance(self.add_user_name_to_info, Unset):
            add_user_name_to_info = UNSET
        else:
            add_user_name_to_info = self.add_user_name_to_info

        add_version_to_infotext: Union[None, Unset, bool]
        if isinstance(self.add_version_to_infotext, Unset):
            add_version_to_infotext = UNSET
        else:
            add_version_to_infotext = self.add_version_to_infotext

        disable_weights_auto_swap: Union[None, Unset, bool]
        if isinstance(self.disable_weights_auto_swap, Unset):
            disable_weights_auto_swap = UNSET
        else:
            disable_weights_auto_swap = self.disable_weights_auto_swap

        infotext_skip_pasting: Union[Any, None, Unset]
        if isinstance(self.infotext_skip_pasting, Unset):
            infotext_skip_pasting = UNSET
        else:
            infotext_skip_pasting = self.infotext_skip_pasting

        infotext_styles: Union[None, Unset, str]
        if isinstance(self.infotext_styles, Unset):
            infotext_styles = UNSET
        else:
            infotext_styles = self.infotext_styles

        show_progressbar: Union[None, Unset, bool]
        if isinstance(self.show_progressbar, Unset):
            show_progressbar = UNSET
        else:
            show_progressbar = self.show_progressbar

        live_previews_enable: Union[None, Unset, bool]
        if isinstance(self.live_previews_enable, Unset):
            live_previews_enable = UNSET
        else:
            live_previews_enable = self.live_previews_enable

        live_previews_image_format: Union[None, Unset, str]
        if isinstance(self.live_previews_image_format, Unset):
            live_previews_image_format = UNSET
        else:
            live_previews_image_format = self.live_previews_image_format

        show_progress_grid: Union[None, Unset, bool]
        if isinstance(self.show_progress_grid, Unset):
            show_progress_grid = UNSET
        else:
            show_progress_grid = self.show_progress_grid

        show_progress_every_n_steps: Union[None, Unset, float]
        if isinstance(self.show_progress_every_n_steps, Unset):
            show_progress_every_n_steps = UNSET
        else:
            show_progress_every_n_steps = self.show_progress_every_n_steps

        show_progress_type: Union[None, Unset, str]
        if isinstance(self.show_progress_type, Unset):
            show_progress_type = UNSET
        else:
            show_progress_type = self.show_progress_type

        live_preview_allow_lowvram_full: Union[Any, None, Unset]
        if isinstance(self.live_preview_allow_lowvram_full, Unset):
            live_preview_allow_lowvram_full = UNSET
        else:
            live_preview_allow_lowvram_full = self.live_preview_allow_lowvram_full

        live_preview_content: Union[None, Unset, str]
        if isinstance(self.live_preview_content, Unset):
            live_preview_content = UNSET
        else:
            live_preview_content = self.live_preview_content

        live_preview_refresh_period: Union[None, Unset, float]
        if isinstance(self.live_preview_refresh_period, Unset):
            live_preview_refresh_period = UNSET
        else:
            live_preview_refresh_period = self.live_preview_refresh_period

        live_preview_fast_interrupt: Union[Any, None, Unset]
        if isinstance(self.live_preview_fast_interrupt, Unset):
            live_preview_fast_interrupt = UNSET
        else:
            live_preview_fast_interrupt = self.live_preview_fast_interrupt

        js_live_preview_in_modal_lightbox: Union[Any, None, Unset]
        if isinstance(self.js_live_preview_in_modal_lightbox, Unset):
            js_live_preview_in_modal_lightbox = UNSET
        else:
            js_live_preview_in_modal_lightbox = self.js_live_preview_in_modal_lightbox

        prevent_screen_sleep_during_generation: Union[None, Unset, bool]
        if isinstance(self.prevent_screen_sleep_during_generation, Unset):
            prevent_screen_sleep_during_generation = UNSET
        else:
            prevent_screen_sleep_during_generation = self.prevent_screen_sleep_during_generation

        hide_samplers: Union[Any, None, Unset]
        if isinstance(self.hide_samplers, Unset):
            hide_samplers = UNSET
        else:
            hide_samplers = self.hide_samplers

        eta_ddim: Union[Any, None, Unset]
        if isinstance(self.eta_ddim, Unset):
            eta_ddim = UNSET
        else:
            eta_ddim = self.eta_ddim

        eta_ancestral: Union[None, Unset, float]
        if isinstance(self.eta_ancestral, Unset):
            eta_ancestral = UNSET
        else:
            eta_ancestral = self.eta_ancestral

        ddim_discretize: Union[None, Unset, str]
        if isinstance(self.ddim_discretize, Unset):
            ddim_discretize = UNSET
        else:
            ddim_discretize = self.ddim_discretize

        s_churn: Union[Any, None, Unset]
        if isinstance(self.s_churn, Unset):
            s_churn = UNSET
        else:
            s_churn = self.s_churn

        s_tmin: Union[Any, None, Unset]
        if isinstance(self.s_tmin, Unset):
            s_tmin = UNSET
        else:
            s_tmin = self.s_tmin

        s_tmax: Union[Any, None, Unset]
        if isinstance(self.s_tmax, Unset):
            s_tmax = UNSET
        else:
            s_tmax = self.s_tmax

        s_noise: Union[None, Unset, float]
        if isinstance(self.s_noise, Unset):
            s_noise = UNSET
        else:
            s_noise = self.s_noise

        sigma_min: Union[Any, None, Unset]
        if isinstance(self.sigma_min, Unset):
            sigma_min = UNSET
        else:
            sigma_min = self.sigma_min

        sigma_max: Union[Any, None, Unset]
        if isinstance(self.sigma_max, Unset):
            sigma_max = UNSET
        else:
            sigma_max = self.sigma_max

        rho: Union[Any, None, Unset]
        if isinstance(self.rho, Unset):
            rho = UNSET
        else:
            rho = self.rho

        eta_noise_seed_delta: Union[Any, None, Unset]
        if isinstance(self.eta_noise_seed_delta, Unset):
            eta_noise_seed_delta = UNSET
        else:
            eta_noise_seed_delta = self.eta_noise_seed_delta

        always_discard_next_to_last_sigma: Union[Any, None, Unset]
        if isinstance(self.always_discard_next_to_last_sigma, Unset):
            always_discard_next_to_last_sigma = UNSET
        else:
            always_discard_next_to_last_sigma = self.always_discard_next_to_last_sigma

        sgm_noise_multiplier: Union[Any, None, Unset]
        if isinstance(self.sgm_noise_multiplier, Unset):
            sgm_noise_multiplier = UNSET
        else:
            sgm_noise_multiplier = self.sgm_noise_multiplier

        uni_pc_variant: Union[None, Unset, str]
        if isinstance(self.uni_pc_variant, Unset):
            uni_pc_variant = UNSET
        else:
            uni_pc_variant = self.uni_pc_variant

        uni_pc_skip_type: Union[None, Unset, str]
        if isinstance(self.uni_pc_skip_type, Unset):
            uni_pc_skip_type = UNSET
        else:
            uni_pc_skip_type = self.uni_pc_skip_type

        uni_pc_order: Union[None, Unset, float]
        if isinstance(self.uni_pc_order, Unset):
            uni_pc_order = UNSET
        else:
            uni_pc_order = self.uni_pc_order

        uni_pc_lower_order_final: Union[None, Unset, bool]
        if isinstance(self.uni_pc_lower_order_final, Unset):
            uni_pc_lower_order_final = UNSET
        else:
            uni_pc_lower_order_final = self.uni_pc_lower_order_final

        sd_noise_schedule: Union[None, Unset, str]
        if isinstance(self.sd_noise_schedule, Unset):
            sd_noise_schedule = UNSET
        else:
            sd_noise_schedule = self.sd_noise_schedule

        skip_early_cond: Union[Any, None, Unset]
        if isinstance(self.skip_early_cond, Unset):
            skip_early_cond = UNSET
        else:
            skip_early_cond = self.skip_early_cond

        beta_dist_alpha: Union[None, Unset, float]
        if isinstance(self.beta_dist_alpha, Unset):
            beta_dist_alpha = UNSET
        else:
            beta_dist_alpha = self.beta_dist_alpha

        beta_dist_beta: Union[None, Unset, float]
        if isinstance(self.beta_dist_beta, Unset):
            beta_dist_beta = UNSET
        else:
            beta_dist_beta = self.beta_dist_beta

        postprocessing_enable_in_main_ui: Union[Any, None, Unset]
        if isinstance(self.postprocessing_enable_in_main_ui, Unset):
            postprocessing_enable_in_main_ui = UNSET
        else:
            postprocessing_enable_in_main_ui = self.postprocessing_enable_in_main_ui

        postprocessing_disable_in_extras: Union[Any, None, Unset]
        if isinstance(self.postprocessing_disable_in_extras, Unset):
            postprocessing_disable_in_extras = UNSET
        else:
            postprocessing_disable_in_extras = self.postprocessing_disable_in_extras

        postprocessing_operation_order: Union[Any, None, Unset]
        if isinstance(self.postprocessing_operation_order, Unset):
            postprocessing_operation_order = UNSET
        else:
            postprocessing_operation_order = self.postprocessing_operation_order

        upscaling_max_images_in_cache: Union[None, Unset, float]
        if isinstance(self.upscaling_max_images_in_cache, Unset):
            upscaling_max_images_in_cache = UNSET
        else:
            upscaling_max_images_in_cache = self.upscaling_max_images_in_cache

        postprocessing_existing_caption_action: Union[None, Unset, str]
        if isinstance(self.postprocessing_existing_caption_action, Unset):
            postprocessing_existing_caption_action = UNSET
        else:
            postprocessing_existing_caption_action = self.postprocessing_existing_caption_action

        disabled_extensions: Union[Any, None, Unset]
        if isinstance(self.disabled_extensions, Unset):
            disabled_extensions = UNSET
        else:
            disabled_extensions = self.disabled_extensions

        disable_all_extensions: Union[None, Unset, str]
        if isinstance(self.disable_all_extensions, Unset):
            disable_all_extensions = UNSET
        else:
            disable_all_extensions = self.disable_all_extensions

        restore_config_state_file: Union[Any, None, Unset]
        if isinstance(self.restore_config_state_file, Unset):
            restore_config_state_file = UNSET
        else:
            restore_config_state_file = self.restore_config_state_file

        sd_checkpoint_hash: Union[Any, None, Unset]
        if isinstance(self.sd_checkpoint_hash, Unset):
            sd_checkpoint_hash = UNSET
        else:
            sd_checkpoint_hash = self.sd_checkpoint_hash

        forge_unet_storage_dtype: Union[None, Unset, str]
        if isinstance(self.forge_unet_storage_dtype, Unset):
            forge_unet_storage_dtype = UNSET
        else:
            forge_unet_storage_dtype = self.forge_unet_storage_dtype

        forge_inference_memory: Union[None, Unset, float]
        if isinstance(self.forge_inference_memory, Unset):
            forge_inference_memory = UNSET
        else:
            forge_inference_memory = self.forge_inference_memory

        forge_async_loading: Union[None, Unset, str]
        if isinstance(self.forge_async_loading, Unset):
            forge_async_loading = UNSET
        else:
            forge_async_loading = self.forge_async_loading

        forge_pin_shared_memory: Union[None, Unset, str]
        if isinstance(self.forge_pin_shared_memory, Unset):
            forge_pin_shared_memory = UNSET
        else:
            forge_pin_shared_memory = self.forge_pin_shared_memory

        forge_preset: Union[None, Unset, str]
        if isinstance(self.forge_preset, Unset):
            forge_preset = UNSET
        else:
            forge_preset = self.forge_preset

        forge_additional_modules: Union[Any, None, Unset]
        if isinstance(self.forge_additional_modules, Unset):
            forge_additional_modules = UNSET
        else:
            forge_additional_modules = self.forge_additional_modules

        settings_in_ui: Union[None, Unset, str]
        if isinstance(self.settings_in_ui, Unset):
            settings_in_ui = UNSET
        else:
            settings_in_ui = self.settings_in_ui

        extra_options_txt2img: Union[Any, None, Unset]
        if isinstance(self.extra_options_txt2img, Unset):
            extra_options_txt2img = UNSET
        else:
            extra_options_txt2img = self.extra_options_txt2img

        extra_options_img2img: Union[Any, None, Unset]
        if isinstance(self.extra_options_img2img, Unset):
            extra_options_img2img = UNSET
        else:
            extra_options_img2img = self.extra_options_img2img

        extra_options_cols: Union[None, Unset, float]
        if isinstance(self.extra_options_cols, Unset):
            extra_options_cols = UNSET
        else:
            extra_options_cols = self.extra_options_cols

        extra_options_accordion: Union[Any, None, Unset]
        if isinstance(self.extra_options_accordion, Unset):
            extra_options_accordion = UNSET
        else:
            extra_options_accordion = self.extra_options_accordion


        field_dict: Dict[str, Any] = {}
        field_dict.update(self.additional_properties)
        field_dict.update({
        })
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
        if save_write_log_csv is not UNSET:
            field_dict["save_write_log_csv"] = save_write_log_csv
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
        if set_scale_by_when_changing_upscaler is not UNSET:
            field_dict["set_scale_by_when_changing_upscaler"] = set_scale_by_when_changing_upscaler
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
        if profiling_explanation is not UNSET:
            field_dict["profiling_explanation"] = profiling_explanation
        if profiling_enable is not UNSET:
            field_dict["profiling_enable"] = profiling_enable
        if profiling_activities is not UNSET:
            field_dict["profiling_activities"] = profiling_activities
        if profiling_record_shapes is not UNSET:
            field_dict["profiling_record_shapes"] = profiling_record_shapes
        if profiling_profile_memory is not UNSET:
            field_dict["profiling_profile_memory"] = profiling_profile_memory
        if profiling_with_stack is not UNSET:
            field_dict["profiling_with_stack"] = profiling_with_stack
        if profiling_filename is not UNSET:
            field_dict["profiling_filename"] = profiling_filename
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
        if sdxl_clip_l_skip is not UNSET:
            field_dict["sdxl_clip_l_skip"] = sdxl_clip_l_skip
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
        if sd3_enable_t5 is not UNSET:
            field_dict["sd3_enable_t5"] = sd3_enable_t5
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
        if img2img_sketch_default_brush_color is not UNSET:
            field_dict["img2img_sketch_default_brush_color"] = img2img_sketch_default_brush_color
        if img2img_inpaint_mask_brush_color is not UNSET:
            field_dict["img2img_inpaint_mask_brush_color"] = img2img_inpaint_mask_brush_color
        if img2img_inpaint_sketch_default_brush_color is not UNSET:
            field_dict["img2img_inpaint_sketch_default_brush_color"] = img2img_inpaint_sketch_default_brush_color
        if img2img_inpaint_mask_high_contrast is not UNSET:
            field_dict["img2img_inpaint_mask_high_contrast"] = img2img_inpaint_mask_high_contrast
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
        if s_min_uncond_all is not UNSET:
            field_dict["s_min_uncond_all"] = s_min_uncond_all
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
        if forge_try_reproduce is not UNSET:
            field_dict["forge_try_reproduce"] = forge_try_reproduce
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
        if hires_fix_use_firstpass_conds is not UNSET:
            field_dict["hires_fix_use_firstpass_conds"] = hires_fix_use_firstpass_conds
        if use_old_scheduling is not UNSET:
            field_dict["use_old_scheduling"] = use_old_scheduling
        if use_downcasted_alpha_bar is not UNSET:
            field_dict["use_downcasted_alpha_bar"] = use_downcasted_alpha_bar
        if refiner_switch_by_sample_steps is not UNSET:
            field_dict["refiner_switch_by_sample_steps"] = refiner_switch_by_sample_steps
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
        if extra_networks_tree_view_style is not UNSET:
            field_dict["extra_networks_tree_view_style"] = extra_networks_tree_view_style
        if extra_networks_tree_view_default_enabled is not UNSET:
            field_dict["extra_networks_tree_view_default_enabled"] = extra_networks_tree_view_default_enabled
        if extra_networks_tree_view_default_width is not UNSET:
            field_dict["extra_networks_tree_view_default_width"] = extra_networks_tree_view_default_width
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
        if quick_setting_list is not UNSET:
            field_dict["quick_setting_list"] = quick_setting_list
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
        if enable_reloading_ui_scripts is not UNSET:
            field_dict["enable_reloading_ui_scripts"] = enable_reloading_ui_scripts
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
        if prevent_screen_sleep_during_generation is not UNSET:
            field_dict["prevent_screen_sleep_during_generation"] = prevent_screen_sleep_during_generation
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
        if skip_early_cond is not UNSET:
            field_dict["skip_early_cond"] = skip_early_cond
        if beta_dist_alpha is not UNSET:
            field_dict["beta_dist_alpha"] = beta_dist_alpha
        if beta_dist_beta is not UNSET:
            field_dict["beta_dist_beta"] = beta_dist_beta
        if postprocessing_enable_in_main_ui is not UNSET:
            field_dict["postprocessing_enable_in_main_ui"] = postprocessing_enable_in_main_ui
        if postprocessing_disable_in_extras is not UNSET:
            field_dict["postprocessing_disable_in_extras"] = postprocessing_disable_in_extras
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
        if forge_unet_storage_dtype is not UNSET:
            field_dict["forge_unet_storage_dtype"] = forge_unet_storage_dtype
        if forge_inference_memory is not UNSET:
            field_dict["forge_inference_memory"] = forge_inference_memory
        if forge_async_loading is not UNSET:
            field_dict["forge_async_loading"] = forge_async_loading
        if forge_pin_shared_memory is not UNSET:
            field_dict["forge_pin_shared_memory"] = forge_pin_shared_memory
        if forge_preset is not UNSET:
            field_dict["forge_preset"] = forge_preset
        if forge_additional_modules is not UNSET:
            field_dict["forge_additional_modules"] = forge_additional_modules
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
        def _parse_samples_save(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        samples_save = _parse_samples_save(d.pop("samples_save", UNSET))


        def _parse_samples_format(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        samples_format = _parse_samples_format(d.pop("samples_format", UNSET))


        def _parse_samples_filename_pattern(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        samples_filename_pattern = _parse_samples_filename_pattern(d.pop("samples_filename_pattern", UNSET))


        def _parse_save_images_add_number(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        save_images_add_number = _parse_save_images_add_number(d.pop("save_images_add_number", UNSET))


        def _parse_save_images_replace_action(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        save_images_replace_action = _parse_save_images_replace_action(d.pop("save_images_replace_action", UNSET))


        def _parse_grid_save(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        grid_save = _parse_grid_save(d.pop("grid_save", UNSET))


        def _parse_grid_format(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        grid_format = _parse_grid_format(d.pop("grid_format", UNSET))


        def _parse_grid_extended_filename(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        grid_extended_filename = _parse_grid_extended_filename(d.pop("grid_extended_filename", UNSET))


        def _parse_grid_only_if_multiple(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        grid_only_if_multiple = _parse_grid_only_if_multiple(d.pop("grid_only_if_multiple", UNSET))


        def _parse_grid_prevent_empty_spots(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        grid_prevent_empty_spots = _parse_grid_prevent_empty_spots(d.pop("grid_prevent_empty_spots", UNSET))


        def _parse_grid_zip_filename_pattern(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        grid_zip_filename_pattern = _parse_grid_zip_filename_pattern(d.pop("grid_zip_filename_pattern", UNSET))


        def _parse_n_rows(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        n_rows = _parse_n_rows(d.pop("n_rows", UNSET))


        def _parse_font(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        font = _parse_font(d.pop("font", UNSET))


        def _parse_grid_text_active_color(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        grid_text_active_color = _parse_grid_text_active_color(d.pop("grid_text_active_color", UNSET))


        def _parse_grid_text_inactive_color(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        grid_text_inactive_color = _parse_grid_text_inactive_color(d.pop("grid_text_inactive_color", UNSET))


        def _parse_grid_background_color(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        grid_background_color = _parse_grid_background_color(d.pop("grid_background_color", UNSET))


        def _parse_save_images_before_face_restoration(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_images_before_face_restoration = _parse_save_images_before_face_restoration(d.pop("save_images_before_face_restoration", UNSET))


        def _parse_save_images_before_highres_fix(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_images_before_highres_fix = _parse_save_images_before_highres_fix(d.pop("save_images_before_highres_fix", UNSET))


        def _parse_save_images_before_color_correction(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_images_before_color_correction = _parse_save_images_before_color_correction(d.pop("save_images_before_color_correction", UNSET))


        def _parse_save_mask(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_mask = _parse_save_mask(d.pop("save_mask", UNSET))


        def _parse_save_mask_composite(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_mask_composite = _parse_save_mask_composite(d.pop("save_mask_composite", UNSET))


        def _parse_jpeg_quality(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        jpeg_quality = _parse_jpeg_quality(d.pop("jpeg_quality", UNSET))


        def _parse_webp_lossless(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        webp_lossless = _parse_webp_lossless(d.pop("webp_lossless", UNSET))


        def _parse_export_for_4chan(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        export_for_4chan = _parse_export_for_4chan(d.pop("export_for_4chan", UNSET))


        def _parse_img_downscale_threshold(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        img_downscale_threshold = _parse_img_downscale_threshold(d.pop("img_downscale_threshold", UNSET))


        def _parse_target_side_length(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        target_side_length = _parse_target_side_length(d.pop("target_side_length", UNSET))


        def _parse_img_max_size_mp(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        img_max_size_mp = _parse_img_max_size_mp(d.pop("img_max_size_mp", UNSET))


        def _parse_use_original_name_batch(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        use_original_name_batch = _parse_use_original_name_batch(d.pop("use_original_name_batch", UNSET))


        def _parse_use_upscaler_name_as_suffix(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        use_upscaler_name_as_suffix = _parse_use_upscaler_name_as_suffix(d.pop("use_upscaler_name_as_suffix", UNSET))


        def _parse_save_selected_only(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        save_selected_only = _parse_save_selected_only(d.pop("save_selected_only", UNSET))


        def _parse_save_write_log_csv(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        save_write_log_csv = _parse_save_write_log_csv(d.pop("save_write_log_csv", UNSET))


        def _parse_save_init_img(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_init_img = _parse_save_init_img(d.pop("save_init_img", UNSET))


        def _parse_temp_dir(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        temp_dir = _parse_temp_dir(d.pop("temp_dir", UNSET))


        def _parse_clean_temp_dir_at_start(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        clean_temp_dir_at_start = _parse_clean_temp_dir_at_start(d.pop("clean_temp_dir_at_start", UNSET))


        def _parse_save_incomplete_images(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_incomplete_images = _parse_save_incomplete_images(d.pop("save_incomplete_images", UNSET))


        def _parse_notification_audio(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        notification_audio = _parse_notification_audio(d.pop("notification_audio", UNSET))


        def _parse_notification_volume(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        notification_volume = _parse_notification_volume(d.pop("notification_volume", UNSET))


        def _parse_outdir_samples(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        outdir_samples = _parse_outdir_samples(d.pop("outdir_samples", UNSET))


        def _parse_outdir_txt2img_samples(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        outdir_txt2img_samples = _parse_outdir_txt2img_samples(d.pop("outdir_txt2img_samples", UNSET))


        def _parse_outdir_img2img_samples(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        outdir_img2img_samples = _parse_outdir_img2img_samples(d.pop("outdir_img2img_samples", UNSET))


        def _parse_outdir_extras_samples(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        outdir_extras_samples = _parse_outdir_extras_samples(d.pop("outdir_extras_samples", UNSET))


        def _parse_outdir_grids(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        outdir_grids = _parse_outdir_grids(d.pop("outdir_grids", UNSET))


        def _parse_outdir_txt2img_grids(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        outdir_txt2img_grids = _parse_outdir_txt2img_grids(d.pop("outdir_txt2img_grids", UNSET))


        def _parse_outdir_img2img_grids(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        outdir_img2img_grids = _parse_outdir_img2img_grids(d.pop("outdir_img2img_grids", UNSET))


        def _parse_outdir_save(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        outdir_save = _parse_outdir_save(d.pop("outdir_save", UNSET))


        def _parse_outdir_init_images(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        outdir_init_images = _parse_outdir_init_images(d.pop("outdir_init_images", UNSET))


        def _parse_save_to_dirs(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        save_to_dirs = _parse_save_to_dirs(d.pop("save_to_dirs", UNSET))


        def _parse_grid_save_to_dirs(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        grid_save_to_dirs = _parse_grid_save_to_dirs(d.pop("grid_save_to_dirs", UNSET))


        def _parse_use_save_to_dirs_for_ui(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        use_save_to_dirs_for_ui = _parse_use_save_to_dirs_for_ui(d.pop("use_save_to_dirs_for_ui", UNSET))


        def _parse_directories_filename_pattern(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        directories_filename_pattern = _parse_directories_filename_pattern(d.pop("directories_filename_pattern", UNSET))


        def _parse_directories_max_prompt_words(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        directories_max_prompt_words = _parse_directories_max_prompt_words(d.pop("directories_max_prompt_words", UNSET))


        def _parse_esrgan_tile(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        esrgan_tile = _parse_esrgan_tile(d.pop("ESRGAN_tile", UNSET))


        def _parse_esrgan_tile_overlap(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        esrgan_tile_overlap = _parse_esrgan_tile_overlap(d.pop("ESRGAN_tile_overlap", UNSET))


        def _parse_realesrgan_enabled_models(data: object) -> Union[List[Any], None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                realesrgan_enabled_models_type_0 = cast(List[Any], data)

                return realesrgan_enabled_models_type_0
            except: # noqa: E722
                pass
            return cast(Union[List[Any], None, Unset], data)

        realesrgan_enabled_models = _parse_realesrgan_enabled_models(d.pop("realesrgan_enabled_models", UNSET))


        def _parse_dat_enabled_models(data: object) -> Union[List[Any], None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                dat_enabled_models_type_0 = cast(List[Any], data)

                return dat_enabled_models_type_0
            except: # noqa: E722
                pass
            return cast(Union[List[Any], None, Unset], data)

        dat_enabled_models = _parse_dat_enabled_models(d.pop("dat_enabled_models", UNSET))


        def _parse_dat_tile(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        dat_tile = _parse_dat_tile(d.pop("DAT_tile", UNSET))


        def _parse_dat_tile_overlap(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        dat_tile_overlap = _parse_dat_tile_overlap(d.pop("DAT_tile_overlap", UNSET))


        def _parse_upscaler_for_img2img(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        upscaler_for_img2img = _parse_upscaler_for_img2img(d.pop("upscaler_for_img2img", UNSET))


        def _parse_set_scale_by_when_changing_upscaler(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        set_scale_by_when_changing_upscaler = _parse_set_scale_by_when_changing_upscaler(d.pop("set_scale_by_when_changing_upscaler", UNSET))


        def _parse_face_restoration(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        face_restoration = _parse_face_restoration(d.pop("face_restoration", UNSET))


        def _parse_face_restoration_model(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        face_restoration_model = _parse_face_restoration_model(d.pop("face_restoration_model", UNSET))


        def _parse_code_former_weight(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        code_former_weight = _parse_code_former_weight(d.pop("code_former_weight", UNSET))


        def _parse_face_restoration_unload(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        face_restoration_unload = _parse_face_restoration_unload(d.pop("face_restoration_unload", UNSET))


        def _parse_auto_launch_browser(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        auto_launch_browser = _parse_auto_launch_browser(d.pop("auto_launch_browser", UNSET))


        def _parse_enable_console_prompts(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        enable_console_prompts = _parse_enable_console_prompts(d.pop("enable_console_prompts", UNSET))


        def _parse_show_warnings(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        show_warnings = _parse_show_warnings(d.pop("show_warnings", UNSET))


        def _parse_show_gradio_deprecation_warnings(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        show_gradio_deprecation_warnings = _parse_show_gradio_deprecation_warnings(d.pop("show_gradio_deprecation_warnings", UNSET))


        def _parse_memmon_poll_rate(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        memmon_poll_rate = _parse_memmon_poll_rate(d.pop("memmon_poll_rate", UNSET))


        def _parse_samples_log_stdout(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        samples_log_stdout = _parse_samples_log_stdout(d.pop("samples_log_stdout", UNSET))


        def _parse_multiple_tqdm(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        multiple_tqdm = _parse_multiple_tqdm(d.pop("multiple_tqdm", UNSET))


        def _parse_enable_upscale_progressbar(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        enable_upscale_progressbar = _parse_enable_upscale_progressbar(d.pop("enable_upscale_progressbar", UNSET))


        def _parse_print_hypernet_extra(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        print_hypernet_extra = _parse_print_hypernet_extra(d.pop("print_hypernet_extra", UNSET))


        def _parse_list_hidden_files(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        list_hidden_files = _parse_list_hidden_files(d.pop("list_hidden_files", UNSET))


        def _parse_disable_mmap_load_safetensors(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        disable_mmap_load_safetensors = _parse_disable_mmap_load_safetensors(d.pop("disable_mmap_load_safetensors", UNSET))


        def _parse_hide_ldm_prints(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        hide_ldm_prints = _parse_hide_ldm_prints(d.pop("hide_ldm_prints", UNSET))


        def _parse_dump_stacks_on_signal(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        dump_stacks_on_signal = _parse_dump_stacks_on_signal(d.pop("dump_stacks_on_signal", UNSET))


        def _parse_profiling_explanation(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        profiling_explanation = _parse_profiling_explanation(d.pop("profiling_explanation", UNSET))


        def _parse_profiling_enable(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        profiling_enable = _parse_profiling_enable(d.pop("profiling_enable", UNSET))


        def _parse_profiling_activities(data: object) -> Union[List[Any], None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                profiling_activities_type_0 = cast(List[Any], data)

                return profiling_activities_type_0
            except: # noqa: E722
                pass
            return cast(Union[List[Any], None, Unset], data)

        profiling_activities = _parse_profiling_activities(d.pop("profiling_activities", UNSET))


        def _parse_profiling_record_shapes(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        profiling_record_shapes = _parse_profiling_record_shapes(d.pop("profiling_record_shapes", UNSET))


        def _parse_profiling_profile_memory(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        profiling_profile_memory = _parse_profiling_profile_memory(d.pop("profiling_profile_memory", UNSET))


        def _parse_profiling_with_stack(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        profiling_with_stack = _parse_profiling_with_stack(d.pop("profiling_with_stack", UNSET))


        def _parse_profiling_filename(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        profiling_filename = _parse_profiling_filename(d.pop("profiling_filename", UNSET))


        def _parse_api_enable_requests(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        api_enable_requests = _parse_api_enable_requests(d.pop("api_enable_requests", UNSET))


        def _parse_api_forbid_local_requests(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        api_forbid_local_requests = _parse_api_forbid_local_requests(d.pop("api_forbid_local_requests", UNSET))


        def _parse_api_useragent(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        api_useragent = _parse_api_useragent(d.pop("api_useragent", UNSET))


        def _parse_unload_models_when_training(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        unload_models_when_training = _parse_unload_models_when_training(d.pop("unload_models_when_training", UNSET))


        def _parse_pin_memory(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        pin_memory = _parse_pin_memory(d.pop("pin_memory", UNSET))


        def _parse_save_optimizer_state(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_optimizer_state = _parse_save_optimizer_state(d.pop("save_optimizer_state", UNSET))


        def _parse_save_training_settings_to_txt(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        save_training_settings_to_txt = _parse_save_training_settings_to_txt(d.pop("save_training_settings_to_txt", UNSET))


        def _parse_dataset_filename_word_regex(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        dataset_filename_word_regex = _parse_dataset_filename_word_regex(d.pop("dataset_filename_word_regex", UNSET))


        def _parse_dataset_filename_join_string(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        dataset_filename_join_string = _parse_dataset_filename_join_string(d.pop("dataset_filename_join_string", UNSET))


        def _parse_training_image_repeats_per_epoch(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        training_image_repeats_per_epoch = _parse_training_image_repeats_per_epoch(d.pop("training_image_repeats_per_epoch", UNSET))


        def _parse_training_write_csv_every(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        training_write_csv_every = _parse_training_write_csv_every(d.pop("training_write_csv_every", UNSET))


        def _parse_training_xattention_optimizations(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        training_xattention_optimizations = _parse_training_xattention_optimizations(d.pop("training_xattention_optimizations", UNSET))


        def _parse_training_enable_tensorboard(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        training_enable_tensorboard = _parse_training_enable_tensorboard(d.pop("training_enable_tensorboard", UNSET))


        def _parse_training_tensorboard_save_images(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        training_tensorboard_save_images = _parse_training_tensorboard_save_images(d.pop("training_tensorboard_save_images", UNSET))


        def _parse_training_tensorboard_flush_every(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        training_tensorboard_flush_every = _parse_training_tensorboard_flush_every(d.pop("training_tensorboard_flush_every", UNSET))


        def _parse_sd_model_checkpoint(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sd_model_checkpoint = _parse_sd_model_checkpoint(d.pop("sd_model_checkpoint", UNSET))


        def _parse_sd_checkpoints_limit(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        sd_checkpoints_limit = _parse_sd_checkpoints_limit(d.pop("sd_checkpoints_limit", UNSET))


        def _parse_sd_checkpoints_keep_in_cpu(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        sd_checkpoints_keep_in_cpu = _parse_sd_checkpoints_keep_in_cpu(d.pop("sd_checkpoints_keep_in_cpu", UNSET))


        def _parse_sd_checkpoint_cache(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sd_checkpoint_cache = _parse_sd_checkpoint_cache(d.pop("sd_checkpoint_cache", UNSET))


        def _parse_sd_unet(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sd_unet = _parse_sd_unet(d.pop("sd_unet", UNSET))


        def _parse_enable_quantization(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        enable_quantization = _parse_enable_quantization(d.pop("enable_quantization", UNSET))


        def _parse_emphasis(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        emphasis = _parse_emphasis(d.pop("emphasis", UNSET))


        def _parse_enable_batch_seeds(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        enable_batch_seeds = _parse_enable_batch_seeds(d.pop("enable_batch_seeds", UNSET))


        def _parse_comma_padding_backtrack(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        comma_padding_backtrack = _parse_comma_padding_backtrack(d.pop("comma_padding_backtrack", UNSET))


        def _parse_sdxl_clip_l_skip(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sdxl_clip_l_skip = _parse_sdxl_clip_l_skip(d.pop("sdxl_clip_l_skip", UNSET))


        def _parse_clip_stop_at_last_layers(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        clip_stop_at_last_layers = _parse_clip_stop_at_last_layers(d.pop("CLIP_stop_at_last_layers", UNSET))


        def _parse_upcast_attn(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        upcast_attn = _parse_upcast_attn(d.pop("upcast_attn", UNSET))


        def _parse_randn_source(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        randn_source = _parse_randn_source(d.pop("randn_source", UNSET))


        def _parse_tiling(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        tiling = _parse_tiling(d.pop("tiling", UNSET))


        def _parse_hires_fix_refiner_pass(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        hires_fix_refiner_pass = _parse_hires_fix_refiner_pass(d.pop("hires_fix_refiner_pass", UNSET))


        def _parse_sdxl_crop_top(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sdxl_crop_top = _parse_sdxl_crop_top(d.pop("sdxl_crop_top", UNSET))


        def _parse_sdxl_crop_left(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sdxl_crop_left = _parse_sdxl_crop_left(d.pop("sdxl_crop_left", UNSET))


        def _parse_sdxl_refiner_low_aesthetic_score(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        sdxl_refiner_low_aesthetic_score = _parse_sdxl_refiner_low_aesthetic_score(d.pop("sdxl_refiner_low_aesthetic_score", UNSET))


        def _parse_sdxl_refiner_high_aesthetic_score(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        sdxl_refiner_high_aesthetic_score = _parse_sdxl_refiner_high_aesthetic_score(d.pop("sdxl_refiner_high_aesthetic_score", UNSET))


        def _parse_sd3_enable_t5(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sd3_enable_t5 = _parse_sd3_enable_t5(d.pop("sd3_enable_t5", UNSET))


        def _parse_sd_vae_explanation(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sd_vae_explanation = _parse_sd_vae_explanation(d.pop("sd_vae_explanation", UNSET))


        def _parse_sd_vae_checkpoint_cache(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sd_vae_checkpoint_cache = _parse_sd_vae_checkpoint_cache(d.pop("sd_vae_checkpoint_cache", UNSET))


        def _parse_sd_vae(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sd_vae = _parse_sd_vae(d.pop("sd_vae", UNSET))


        def _parse_sd_vae_overrides_per_model_preferences(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        sd_vae_overrides_per_model_preferences = _parse_sd_vae_overrides_per_model_preferences(d.pop("sd_vae_overrides_per_model_preferences", UNSET))


        def _parse_auto_vae_precision_bfloat16(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        auto_vae_precision_bfloat16 = _parse_auto_vae_precision_bfloat16(d.pop("auto_vae_precision_bfloat16", UNSET))


        def _parse_auto_vae_precision(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        auto_vae_precision = _parse_auto_vae_precision(d.pop("auto_vae_precision", UNSET))


        def _parse_sd_vae_encode_method(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sd_vae_encode_method = _parse_sd_vae_encode_method(d.pop("sd_vae_encode_method", UNSET))


        def _parse_sd_vae_decode_method(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sd_vae_decode_method = _parse_sd_vae_decode_method(d.pop("sd_vae_decode_method", UNSET))


        def _parse_inpainting_mask_weight(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        inpainting_mask_weight = _parse_inpainting_mask_weight(d.pop("inpainting_mask_weight", UNSET))


        def _parse_initial_noise_multiplier(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        initial_noise_multiplier = _parse_initial_noise_multiplier(d.pop("initial_noise_multiplier", UNSET))


        def _parse_img2img_extra_noise(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        img2img_extra_noise = _parse_img2img_extra_noise(d.pop("img2img_extra_noise", UNSET))


        def _parse_img2img_color_correction(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        img2img_color_correction = _parse_img2img_color_correction(d.pop("img2img_color_correction", UNSET))


        def _parse_img2img_fix_steps(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        img2img_fix_steps = _parse_img2img_fix_steps(d.pop("img2img_fix_steps", UNSET))


        def _parse_img2img_background_color(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        img2img_background_color = _parse_img2img_background_color(d.pop("img2img_background_color", UNSET))


        def _parse_img2img_sketch_default_brush_color(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        img2img_sketch_default_brush_color = _parse_img2img_sketch_default_brush_color(d.pop("img2img_sketch_default_brush_color", UNSET))


        def _parse_img2img_inpaint_mask_brush_color(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        img2img_inpaint_mask_brush_color = _parse_img2img_inpaint_mask_brush_color(d.pop("img2img_inpaint_mask_brush_color", UNSET))


        def _parse_img2img_inpaint_sketch_default_brush_color(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        img2img_inpaint_sketch_default_brush_color = _parse_img2img_inpaint_sketch_default_brush_color(d.pop("img2img_inpaint_sketch_default_brush_color", UNSET))


        def _parse_img2img_inpaint_mask_high_contrast(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        img2img_inpaint_mask_high_contrast = _parse_img2img_inpaint_mask_high_contrast(d.pop("img2img_inpaint_mask_high_contrast", UNSET))


        def _parse_return_mask(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        return_mask = _parse_return_mask(d.pop("return_mask", UNSET))


        def _parse_return_mask_composite(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        return_mask_composite = _parse_return_mask_composite(d.pop("return_mask_composite", UNSET))


        def _parse_img2img_batch_show_results_limit(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        img2img_batch_show_results_limit = _parse_img2img_batch_show_results_limit(d.pop("img2img_batch_show_results_limit", UNSET))


        def _parse_overlay_inpaint(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        overlay_inpaint = _parse_overlay_inpaint(d.pop("overlay_inpaint", UNSET))


        def _parse_cross_attention_optimization(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        cross_attention_optimization = _parse_cross_attention_optimization(d.pop("cross_attention_optimization", UNSET))


        def _parse_s_min_uncond(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        s_min_uncond = _parse_s_min_uncond(d.pop("s_min_uncond", UNSET))


        def _parse_s_min_uncond_all(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        s_min_uncond_all = _parse_s_min_uncond_all(d.pop("s_min_uncond_all", UNSET))


        def _parse_token_merging_ratio(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        token_merging_ratio = _parse_token_merging_ratio(d.pop("token_merging_ratio", UNSET))


        def _parse_token_merging_ratio_img2img(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        token_merging_ratio_img2img = _parse_token_merging_ratio_img2img(d.pop("token_merging_ratio_img2img", UNSET))


        def _parse_token_merging_ratio_hr(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        token_merging_ratio_hr = _parse_token_merging_ratio_hr(d.pop("token_merging_ratio_hr", UNSET))


        def _parse_pad_cond_uncond(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        pad_cond_uncond = _parse_pad_cond_uncond(d.pop("pad_cond_uncond", UNSET))


        def _parse_pad_cond_uncond_v0(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        pad_cond_uncond_v0 = _parse_pad_cond_uncond_v0(d.pop("pad_cond_uncond_v0", UNSET))


        def _parse_persistent_cond_cache(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        persistent_cond_cache = _parse_persistent_cond_cache(d.pop("persistent_cond_cache", UNSET))


        def _parse_batch_cond_uncond(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        batch_cond_uncond = _parse_batch_cond_uncond(d.pop("batch_cond_uncond", UNSET))


        def _parse_fp8_storage(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        fp8_storage = _parse_fp8_storage(d.pop("fp8_storage", UNSET))


        def _parse_cache_fp16_weight(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        cache_fp16_weight = _parse_cache_fp16_weight(d.pop("cache_fp16_weight", UNSET))


        def _parse_forge_try_reproduce(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        forge_try_reproduce = _parse_forge_try_reproduce(d.pop("forge_try_reproduce", UNSET))


        def _parse_auto_backcompat(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        auto_backcompat = _parse_auto_backcompat(d.pop("auto_backcompat", UNSET))


        def _parse_use_old_emphasis_implementation(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        use_old_emphasis_implementation = _parse_use_old_emphasis_implementation(d.pop("use_old_emphasis_implementation", UNSET))


        def _parse_use_old_karras_scheduler_sigmas(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        use_old_karras_scheduler_sigmas = _parse_use_old_karras_scheduler_sigmas(d.pop("use_old_karras_scheduler_sigmas", UNSET))


        def _parse_no_dpmpp_sde_batch_determinism(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        no_dpmpp_sde_batch_determinism = _parse_no_dpmpp_sde_batch_determinism(d.pop("no_dpmpp_sde_batch_determinism", UNSET))


        def _parse_use_old_hires_fix_width_height(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        use_old_hires_fix_width_height = _parse_use_old_hires_fix_width_height(d.pop("use_old_hires_fix_width_height", UNSET))


        def _parse_hires_fix_use_firstpass_conds(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        hires_fix_use_firstpass_conds = _parse_hires_fix_use_firstpass_conds(d.pop("hires_fix_use_firstpass_conds", UNSET))


        def _parse_use_old_scheduling(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        use_old_scheduling = _parse_use_old_scheduling(d.pop("use_old_scheduling", UNSET))


        def _parse_use_downcasted_alpha_bar(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        use_downcasted_alpha_bar = _parse_use_downcasted_alpha_bar(d.pop("use_downcasted_alpha_bar", UNSET))


        def _parse_refiner_switch_by_sample_steps(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        refiner_switch_by_sample_steps = _parse_refiner_switch_by_sample_steps(d.pop("refiner_switch_by_sample_steps", UNSET))


        def _parse_interrogate_keep_models_in_memory(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        interrogate_keep_models_in_memory = _parse_interrogate_keep_models_in_memory(d.pop("interrogate_keep_models_in_memory", UNSET))


        def _parse_interrogate_return_ranks(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        interrogate_return_ranks = _parse_interrogate_return_ranks(d.pop("interrogate_return_ranks", UNSET))


        def _parse_interrogate_clip_num_beams(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        interrogate_clip_num_beams = _parse_interrogate_clip_num_beams(d.pop("interrogate_clip_num_beams", UNSET))


        def _parse_interrogate_clip_min_length(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        interrogate_clip_min_length = _parse_interrogate_clip_min_length(d.pop("interrogate_clip_min_length", UNSET))


        def _parse_interrogate_clip_max_length(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        interrogate_clip_max_length = _parse_interrogate_clip_max_length(d.pop("interrogate_clip_max_length", UNSET))


        def _parse_interrogate_clip_dict_limit(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        interrogate_clip_dict_limit = _parse_interrogate_clip_dict_limit(d.pop("interrogate_clip_dict_limit", UNSET))


        def _parse_interrogate_clip_skip_categories(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        interrogate_clip_skip_categories = _parse_interrogate_clip_skip_categories(d.pop("interrogate_clip_skip_categories", UNSET))


        def _parse_interrogate_deepbooru_score_threshold(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        interrogate_deepbooru_score_threshold = _parse_interrogate_deepbooru_score_threshold(d.pop("interrogate_deepbooru_score_threshold", UNSET))


        def _parse_deepbooru_sort_alpha(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        deepbooru_sort_alpha = _parse_deepbooru_sort_alpha(d.pop("deepbooru_sort_alpha", UNSET))


        def _parse_deepbooru_use_spaces(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        deepbooru_use_spaces = _parse_deepbooru_use_spaces(d.pop("deepbooru_use_spaces", UNSET))


        def _parse_deepbooru_escape(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        deepbooru_escape = _parse_deepbooru_escape(d.pop("deepbooru_escape", UNSET))


        def _parse_deepbooru_filter_tags(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        deepbooru_filter_tags = _parse_deepbooru_filter_tags(d.pop("deepbooru_filter_tags", UNSET))


        def _parse_extra_networks_show_hidden_directories(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        extra_networks_show_hidden_directories = _parse_extra_networks_show_hidden_directories(d.pop("extra_networks_show_hidden_directories", UNSET))


        def _parse_extra_networks_dir_button_function(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        extra_networks_dir_button_function = _parse_extra_networks_dir_button_function(d.pop("extra_networks_dir_button_function", UNSET))


        def _parse_extra_networks_hidden_models(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        extra_networks_hidden_models = _parse_extra_networks_hidden_models(d.pop("extra_networks_hidden_models", UNSET))


        def _parse_extra_networks_default_multiplier(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        extra_networks_default_multiplier = _parse_extra_networks_default_multiplier(d.pop("extra_networks_default_multiplier", UNSET))


        def _parse_extra_networks_card_width(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        extra_networks_card_width = _parse_extra_networks_card_width(d.pop("extra_networks_card_width", UNSET))


        def _parse_extra_networks_card_height(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        extra_networks_card_height = _parse_extra_networks_card_height(d.pop("extra_networks_card_height", UNSET))


        def _parse_extra_networks_card_text_scale(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        extra_networks_card_text_scale = _parse_extra_networks_card_text_scale(d.pop("extra_networks_card_text_scale", UNSET))


        def _parse_extra_networks_card_show_desc(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        extra_networks_card_show_desc = _parse_extra_networks_card_show_desc(d.pop("extra_networks_card_show_desc", UNSET))


        def _parse_extra_networks_card_description_is_html(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        extra_networks_card_description_is_html = _parse_extra_networks_card_description_is_html(d.pop("extra_networks_card_description_is_html", UNSET))


        def _parse_extra_networks_card_order_field(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        extra_networks_card_order_field = _parse_extra_networks_card_order_field(d.pop("extra_networks_card_order_field", UNSET))


        def _parse_extra_networks_card_order(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        extra_networks_card_order = _parse_extra_networks_card_order(d.pop("extra_networks_card_order", UNSET))


        def _parse_extra_networks_tree_view_style(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        extra_networks_tree_view_style = _parse_extra_networks_tree_view_style(d.pop("extra_networks_tree_view_style", UNSET))


        def _parse_extra_networks_tree_view_default_enabled(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        extra_networks_tree_view_default_enabled = _parse_extra_networks_tree_view_default_enabled(d.pop("extra_networks_tree_view_default_enabled", UNSET))


        def _parse_extra_networks_tree_view_default_width(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        extra_networks_tree_view_default_width = _parse_extra_networks_tree_view_default_width(d.pop("extra_networks_tree_view_default_width", UNSET))


        def _parse_extra_networks_add_text_separator(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        extra_networks_add_text_separator = _parse_extra_networks_add_text_separator(d.pop("extra_networks_add_text_separator", UNSET))


        def _parse_ui_extra_networks_tab_reorder(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        ui_extra_networks_tab_reorder = _parse_ui_extra_networks_tab_reorder(d.pop("ui_extra_networks_tab_reorder", UNSET))


        def _parse_textual_inversion_print_at_load(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        textual_inversion_print_at_load = _parse_textual_inversion_print_at_load(d.pop("textual_inversion_print_at_load", UNSET))


        def _parse_textual_inversion_add_hashes_to_infotext(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        textual_inversion_add_hashes_to_infotext = _parse_textual_inversion_add_hashes_to_infotext(d.pop("textual_inversion_add_hashes_to_infotext", UNSET))


        def _parse_sd_hypernetwork(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sd_hypernetwork = _parse_sd_hypernetwork(d.pop("sd_hypernetwork", UNSET))


        def _parse_keyedit_precision_attention(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        keyedit_precision_attention = _parse_keyedit_precision_attention(d.pop("keyedit_precision_attention", UNSET))


        def _parse_keyedit_precision_extra(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        keyedit_precision_extra = _parse_keyedit_precision_extra(d.pop("keyedit_precision_extra", UNSET))


        def _parse_keyedit_delimiters(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        keyedit_delimiters = _parse_keyedit_delimiters(d.pop("keyedit_delimiters", UNSET))


        def _parse_keyedit_delimiters_whitespace(data: object) -> Union[List[Any], None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            try:
                if not isinstance(data, list):
                    raise TypeError()
                keyedit_delimiters_whitespace_type_0 = cast(List[Any], data)

                return keyedit_delimiters_whitespace_type_0
            except: # noqa: E722
                pass
            return cast(Union[List[Any], None, Unset], data)

        keyedit_delimiters_whitespace = _parse_keyedit_delimiters_whitespace(d.pop("keyedit_delimiters_whitespace", UNSET))


        def _parse_keyedit_move(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        keyedit_move = _parse_keyedit_move(d.pop("keyedit_move", UNSET))


        def _parse_disable_token_counters(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        disable_token_counters = _parse_disable_token_counters(d.pop("disable_token_counters", UNSET))


        def _parse_include_styles_into_token_counters(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        include_styles_into_token_counters = _parse_include_styles_into_token_counters(d.pop("include_styles_into_token_counters", UNSET))


        def _parse_return_grid(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        return_grid = _parse_return_grid(d.pop("return_grid", UNSET))


        def _parse_do_not_show_images(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        do_not_show_images = _parse_do_not_show_images(d.pop("do_not_show_images", UNSET))


        def _parse_js_modal_lightbox(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        js_modal_lightbox = _parse_js_modal_lightbox(d.pop("js_modal_lightbox", UNSET))


        def _parse_js_modal_lightbox_initially_zoomed(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        js_modal_lightbox_initially_zoomed = _parse_js_modal_lightbox_initially_zoomed(d.pop("js_modal_lightbox_initially_zoomed", UNSET))


        def _parse_js_modal_lightbox_gamepad(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        js_modal_lightbox_gamepad = _parse_js_modal_lightbox_gamepad(d.pop("js_modal_lightbox_gamepad", UNSET))


        def _parse_js_modal_lightbox_gamepad_repeat(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        js_modal_lightbox_gamepad_repeat = _parse_js_modal_lightbox_gamepad_repeat(d.pop("js_modal_lightbox_gamepad_repeat", UNSET))


        def _parse_sd_webui_modal_lightbox_icon_opacity(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        sd_webui_modal_lightbox_icon_opacity = _parse_sd_webui_modal_lightbox_icon_opacity(d.pop("sd_webui_modal_lightbox_icon_opacity", UNSET))


        def _parse_sd_webui_modal_lightbox_toolbar_opacity(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        sd_webui_modal_lightbox_toolbar_opacity = _parse_sd_webui_modal_lightbox_toolbar_opacity(d.pop("sd_webui_modal_lightbox_toolbar_opacity", UNSET))


        def _parse_gallery_height(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        gallery_height = _parse_gallery_height(d.pop("gallery_height", UNSET))


        def _parse_open_dir_button_choice(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        open_dir_button_choice = _parse_open_dir_button_choice(d.pop("open_dir_button_choice", UNSET))


        def _parse_compact_prompt_box(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        compact_prompt_box = _parse_compact_prompt_box(d.pop("compact_prompt_box", UNSET))


        def _parse_samplers_in_dropdown(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        samplers_in_dropdown = _parse_samplers_in_dropdown(d.pop("samplers_in_dropdown", UNSET))


        def _parse_dimensions_and_batch_together(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        dimensions_and_batch_together = _parse_dimensions_and_batch_together(d.pop("dimensions_and_batch_together", UNSET))


        def _parse_sd_checkpoint_dropdown_use_short(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sd_checkpoint_dropdown_use_short = _parse_sd_checkpoint_dropdown_use_short(d.pop("sd_checkpoint_dropdown_use_short", UNSET))


        def _parse_hires_fix_show_sampler(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        hires_fix_show_sampler = _parse_hires_fix_show_sampler(d.pop("hires_fix_show_sampler", UNSET))


        def _parse_hires_fix_show_prompts(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        hires_fix_show_prompts = _parse_hires_fix_show_prompts(d.pop("hires_fix_show_prompts", UNSET))


        def _parse_txt2img_settings_accordion(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        txt2img_settings_accordion = _parse_txt2img_settings_accordion(d.pop("txt2img_settings_accordion", UNSET))


        def _parse_img2img_settings_accordion(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        img2img_settings_accordion = _parse_img2img_settings_accordion(d.pop("img2img_settings_accordion", UNSET))


        def _parse_interrupt_after_current(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        interrupt_after_current = _parse_interrupt_after_current(d.pop("interrupt_after_current", UNSET))


        def _parse_localization(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        localization = _parse_localization(d.pop("localization", UNSET))


        def _parse_quick_setting_list(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        quick_setting_list = _parse_quick_setting_list(d.pop("quick_setting_list", UNSET))


        def _parse_ui_tab_order(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        ui_tab_order = _parse_ui_tab_order(d.pop("ui_tab_order", UNSET))


        def _parse_hidden_tabs(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        hidden_tabs = _parse_hidden_tabs(d.pop("hidden_tabs", UNSET))


        def _parse_ui_reorder_list(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        ui_reorder_list = _parse_ui_reorder_list(d.pop("ui_reorder_list", UNSET))


        def _parse_gradio_theme(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        gradio_theme = _parse_gradio_theme(d.pop("gradio_theme", UNSET))


        def _parse_gradio_themes_cache(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        gradio_themes_cache = _parse_gradio_themes_cache(d.pop("gradio_themes_cache", UNSET))


        def _parse_show_progress_in_title(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        show_progress_in_title = _parse_show_progress_in_title(d.pop("show_progress_in_title", UNSET))


        def _parse_send_seed(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        send_seed = _parse_send_seed(d.pop("send_seed", UNSET))


        def _parse_send_size(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        send_size = _parse_send_size(d.pop("send_size", UNSET))


        def _parse_enable_reloading_ui_scripts(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        enable_reloading_ui_scripts = _parse_enable_reloading_ui_scripts(d.pop("enable_reloading_ui_scripts", UNSET))


        def _parse_infotext_explanation(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        infotext_explanation = _parse_infotext_explanation(d.pop("infotext_explanation", UNSET))


        def _parse_enable_pnginfo(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        enable_pnginfo = _parse_enable_pnginfo(d.pop("enable_pnginfo", UNSET))


        def _parse_save_txt(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        save_txt = _parse_save_txt(d.pop("save_txt", UNSET))


        def _parse_add_model_name_to_info(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        add_model_name_to_info = _parse_add_model_name_to_info(d.pop("add_model_name_to_info", UNSET))


        def _parse_add_model_hash_to_info(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        add_model_hash_to_info = _parse_add_model_hash_to_info(d.pop("add_model_hash_to_info", UNSET))


        def _parse_add_vae_name_to_info(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        add_vae_name_to_info = _parse_add_vae_name_to_info(d.pop("add_vae_name_to_info", UNSET))


        def _parse_add_vae_hash_to_info(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        add_vae_hash_to_info = _parse_add_vae_hash_to_info(d.pop("add_vae_hash_to_info", UNSET))


        def _parse_add_user_name_to_info(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        add_user_name_to_info = _parse_add_user_name_to_info(d.pop("add_user_name_to_info", UNSET))


        def _parse_add_version_to_infotext(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        add_version_to_infotext = _parse_add_version_to_infotext(d.pop("add_version_to_infotext", UNSET))


        def _parse_disable_weights_auto_swap(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        disable_weights_auto_swap = _parse_disable_weights_auto_swap(d.pop("disable_weights_auto_swap", UNSET))


        def _parse_infotext_skip_pasting(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        infotext_skip_pasting = _parse_infotext_skip_pasting(d.pop("infotext_skip_pasting", UNSET))


        def _parse_infotext_styles(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        infotext_styles = _parse_infotext_styles(d.pop("infotext_styles", UNSET))


        def _parse_show_progressbar(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        show_progressbar = _parse_show_progressbar(d.pop("show_progressbar", UNSET))


        def _parse_live_previews_enable(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        live_previews_enable = _parse_live_previews_enable(d.pop("live_previews_enable", UNSET))


        def _parse_live_previews_image_format(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        live_previews_image_format = _parse_live_previews_image_format(d.pop("live_previews_image_format", UNSET))


        def _parse_show_progress_grid(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        show_progress_grid = _parse_show_progress_grid(d.pop("show_progress_grid", UNSET))


        def _parse_show_progress_every_n_steps(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        show_progress_every_n_steps = _parse_show_progress_every_n_steps(d.pop("show_progress_every_n_steps", UNSET))


        def _parse_show_progress_type(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        show_progress_type = _parse_show_progress_type(d.pop("show_progress_type", UNSET))


        def _parse_live_preview_allow_lowvram_full(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        live_preview_allow_lowvram_full = _parse_live_preview_allow_lowvram_full(d.pop("live_preview_allow_lowvram_full", UNSET))


        def _parse_live_preview_content(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        live_preview_content = _parse_live_preview_content(d.pop("live_preview_content", UNSET))


        def _parse_live_preview_refresh_period(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        live_preview_refresh_period = _parse_live_preview_refresh_period(d.pop("live_preview_refresh_period", UNSET))


        def _parse_live_preview_fast_interrupt(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        live_preview_fast_interrupt = _parse_live_preview_fast_interrupt(d.pop("live_preview_fast_interrupt", UNSET))


        def _parse_js_live_preview_in_modal_lightbox(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        js_live_preview_in_modal_lightbox = _parse_js_live_preview_in_modal_lightbox(d.pop("js_live_preview_in_modal_lightbox", UNSET))


        def _parse_prevent_screen_sleep_during_generation(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        prevent_screen_sleep_during_generation = _parse_prevent_screen_sleep_during_generation(d.pop("prevent_screen_sleep_during_generation", UNSET))


        def _parse_hide_samplers(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        hide_samplers = _parse_hide_samplers(d.pop("hide_samplers", UNSET))


        def _parse_eta_ddim(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        eta_ddim = _parse_eta_ddim(d.pop("eta_ddim", UNSET))


        def _parse_eta_ancestral(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        eta_ancestral = _parse_eta_ancestral(d.pop("eta_ancestral", UNSET))


        def _parse_ddim_discretize(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        ddim_discretize = _parse_ddim_discretize(d.pop("ddim_discretize", UNSET))


        def _parse_s_churn(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        s_churn = _parse_s_churn(d.pop("s_churn", UNSET))


        def _parse_s_tmin(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        s_tmin = _parse_s_tmin(d.pop("s_tmin", UNSET))


        def _parse_s_tmax(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        s_tmax = _parse_s_tmax(d.pop("s_tmax", UNSET))


        def _parse_s_noise(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        s_noise = _parse_s_noise(d.pop("s_noise", UNSET))


        def _parse_sigma_min(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sigma_min = _parse_sigma_min(d.pop("sigma_min", UNSET))


        def _parse_sigma_max(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sigma_max = _parse_sigma_max(d.pop("sigma_max", UNSET))


        def _parse_rho(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        rho = _parse_rho(d.pop("rho", UNSET))


        def _parse_eta_noise_seed_delta(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        eta_noise_seed_delta = _parse_eta_noise_seed_delta(d.pop("eta_noise_seed_delta", UNSET))


        def _parse_always_discard_next_to_last_sigma(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        always_discard_next_to_last_sigma = _parse_always_discard_next_to_last_sigma(d.pop("always_discard_next_to_last_sigma", UNSET))


        def _parse_sgm_noise_multiplier(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sgm_noise_multiplier = _parse_sgm_noise_multiplier(d.pop("sgm_noise_multiplier", UNSET))


        def _parse_uni_pc_variant(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        uni_pc_variant = _parse_uni_pc_variant(d.pop("uni_pc_variant", UNSET))


        def _parse_uni_pc_skip_type(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        uni_pc_skip_type = _parse_uni_pc_skip_type(d.pop("uni_pc_skip_type", UNSET))


        def _parse_uni_pc_order(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        uni_pc_order = _parse_uni_pc_order(d.pop("uni_pc_order", UNSET))


        def _parse_uni_pc_lower_order_final(data: object) -> Union[None, Unset, bool]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, bool], data)

        uni_pc_lower_order_final = _parse_uni_pc_lower_order_final(d.pop("uni_pc_lower_order_final", UNSET))


        def _parse_sd_noise_schedule(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        sd_noise_schedule = _parse_sd_noise_schedule(d.pop("sd_noise_schedule", UNSET))


        def _parse_skip_early_cond(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        skip_early_cond = _parse_skip_early_cond(d.pop("skip_early_cond", UNSET))


        def _parse_beta_dist_alpha(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        beta_dist_alpha = _parse_beta_dist_alpha(d.pop("beta_dist_alpha", UNSET))


        def _parse_beta_dist_beta(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        beta_dist_beta = _parse_beta_dist_beta(d.pop("beta_dist_beta", UNSET))


        def _parse_postprocessing_enable_in_main_ui(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        postprocessing_enable_in_main_ui = _parse_postprocessing_enable_in_main_ui(d.pop("postprocessing_enable_in_main_ui", UNSET))


        def _parse_postprocessing_disable_in_extras(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        postprocessing_disable_in_extras = _parse_postprocessing_disable_in_extras(d.pop("postprocessing_disable_in_extras", UNSET))


        def _parse_postprocessing_operation_order(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        postprocessing_operation_order = _parse_postprocessing_operation_order(d.pop("postprocessing_operation_order", UNSET))


        def _parse_upscaling_max_images_in_cache(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        upscaling_max_images_in_cache = _parse_upscaling_max_images_in_cache(d.pop("upscaling_max_images_in_cache", UNSET))


        def _parse_postprocessing_existing_caption_action(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        postprocessing_existing_caption_action = _parse_postprocessing_existing_caption_action(d.pop("postprocessing_existing_caption_action", UNSET))


        def _parse_disabled_extensions(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        disabled_extensions = _parse_disabled_extensions(d.pop("disabled_extensions", UNSET))


        def _parse_disable_all_extensions(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        disable_all_extensions = _parse_disable_all_extensions(d.pop("disable_all_extensions", UNSET))


        def _parse_restore_config_state_file(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        restore_config_state_file = _parse_restore_config_state_file(d.pop("restore_config_state_file", UNSET))


        def _parse_sd_checkpoint_hash(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        sd_checkpoint_hash = _parse_sd_checkpoint_hash(d.pop("sd_checkpoint_hash", UNSET))


        def _parse_forge_unet_storage_dtype(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        forge_unet_storage_dtype = _parse_forge_unet_storage_dtype(d.pop("forge_unet_storage_dtype", UNSET))


        def _parse_forge_inference_memory(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        forge_inference_memory = _parse_forge_inference_memory(d.pop("forge_inference_memory", UNSET))


        def _parse_forge_async_loading(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        forge_async_loading = _parse_forge_async_loading(d.pop("forge_async_loading", UNSET))


        def _parse_forge_pin_shared_memory(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        forge_pin_shared_memory = _parse_forge_pin_shared_memory(d.pop("forge_pin_shared_memory", UNSET))


        def _parse_forge_preset(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        forge_preset = _parse_forge_preset(d.pop("forge_preset", UNSET))


        def _parse_forge_additional_modules(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        forge_additional_modules = _parse_forge_additional_modules(d.pop("forge_additional_modules", UNSET))


        def _parse_settings_in_ui(data: object) -> Union[None, Unset, str]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, str], data)

        settings_in_ui = _parse_settings_in_ui(d.pop("settings_in_ui", UNSET))


        def _parse_extra_options_txt2img(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        extra_options_txt2img = _parse_extra_options_txt2img(d.pop("extra_options_txt2img", UNSET))


        def _parse_extra_options_img2img(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        extra_options_img2img = _parse_extra_options_img2img(d.pop("extra_options_img2img", UNSET))


        def _parse_extra_options_cols(data: object) -> Union[None, Unset, float]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[None, Unset, float], data)

        extra_options_cols = _parse_extra_options_cols(d.pop("extra_options_cols", UNSET))


        def _parse_extra_options_accordion(data: object) -> Union[Any, None, Unset]:
            if data is None:
                return data
            if isinstance(data, Unset):
                return data
            return cast(Union[Any, None, Unset], data)

        extra_options_accordion = _parse_extra_options_accordion(d.pop("extra_options_accordion", UNSET))


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
            save_write_log_csv=save_write_log_csv,
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
            set_scale_by_when_changing_upscaler=set_scale_by_when_changing_upscaler,
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
            profiling_explanation=profiling_explanation,
            profiling_enable=profiling_enable,
            profiling_activities=profiling_activities,
            profiling_record_shapes=profiling_record_shapes,
            profiling_profile_memory=profiling_profile_memory,
            profiling_with_stack=profiling_with_stack,
            profiling_filename=profiling_filename,
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
            sdxl_clip_l_skip=sdxl_clip_l_skip,
            clip_stop_at_last_layers=clip_stop_at_last_layers,
            upcast_attn=upcast_attn,
            randn_source=randn_source,
            tiling=tiling,
            hires_fix_refiner_pass=hires_fix_refiner_pass,
            sdxl_crop_top=sdxl_crop_top,
            sdxl_crop_left=sdxl_crop_left,
            sdxl_refiner_low_aesthetic_score=sdxl_refiner_low_aesthetic_score,
            sdxl_refiner_high_aesthetic_score=sdxl_refiner_high_aesthetic_score,
            sd3_enable_t5=sd3_enable_t5,
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
            img2img_sketch_default_brush_color=img2img_sketch_default_brush_color,
            img2img_inpaint_mask_brush_color=img2img_inpaint_mask_brush_color,
            img2img_inpaint_sketch_default_brush_color=img2img_inpaint_sketch_default_brush_color,
            img2img_inpaint_mask_high_contrast=img2img_inpaint_mask_high_contrast,
            return_mask=return_mask,
            return_mask_composite=return_mask_composite,
            img2img_batch_show_results_limit=img2img_batch_show_results_limit,
            overlay_inpaint=overlay_inpaint,
            cross_attention_optimization=cross_attention_optimization,
            s_min_uncond=s_min_uncond,
            s_min_uncond_all=s_min_uncond_all,
            token_merging_ratio=token_merging_ratio,
            token_merging_ratio_img2img=token_merging_ratio_img2img,
            token_merging_ratio_hr=token_merging_ratio_hr,
            pad_cond_uncond=pad_cond_uncond,
            pad_cond_uncond_v0=pad_cond_uncond_v0,
            persistent_cond_cache=persistent_cond_cache,
            batch_cond_uncond=batch_cond_uncond,
            fp8_storage=fp8_storage,
            cache_fp16_weight=cache_fp16_weight,
            forge_try_reproduce=forge_try_reproduce,
            auto_backcompat=auto_backcompat,
            use_old_emphasis_implementation=use_old_emphasis_implementation,
            use_old_karras_scheduler_sigmas=use_old_karras_scheduler_sigmas,
            no_dpmpp_sde_batch_determinism=no_dpmpp_sde_batch_determinism,
            use_old_hires_fix_width_height=use_old_hires_fix_width_height,
            hires_fix_use_firstpass_conds=hires_fix_use_firstpass_conds,
            use_old_scheduling=use_old_scheduling,
            use_downcasted_alpha_bar=use_downcasted_alpha_bar,
            refiner_switch_by_sample_steps=refiner_switch_by_sample_steps,
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
            extra_networks_tree_view_style=extra_networks_tree_view_style,
            extra_networks_tree_view_default_enabled=extra_networks_tree_view_default_enabled,
            extra_networks_tree_view_default_width=extra_networks_tree_view_default_width,
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
            quick_setting_list=quick_setting_list,
            ui_tab_order=ui_tab_order,
            hidden_tabs=hidden_tabs,
            ui_reorder_list=ui_reorder_list,
            gradio_theme=gradio_theme,
            gradio_themes_cache=gradio_themes_cache,
            show_progress_in_title=show_progress_in_title,
            send_seed=send_seed,
            send_size=send_size,
            enable_reloading_ui_scripts=enable_reloading_ui_scripts,
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
            prevent_screen_sleep_during_generation=prevent_screen_sleep_during_generation,
            hide_samplers=hide_samplers,
            eta_ddim=eta_ddim,
            eta_ancestral=eta_ancestral,
            ddim_discretize=ddim_discretize,
            s_churn=s_churn,
            s_tmin=s_tmin,
            s_tmax=s_tmax,
            s_noise=s_noise,
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
            skip_early_cond=skip_early_cond,
            beta_dist_alpha=beta_dist_alpha,
            beta_dist_beta=beta_dist_beta,
            postprocessing_enable_in_main_ui=postprocessing_enable_in_main_ui,
            postprocessing_disable_in_extras=postprocessing_disable_in_extras,
            postprocessing_operation_order=postprocessing_operation_order,
            upscaling_max_images_in_cache=upscaling_max_images_in_cache,
            postprocessing_existing_caption_action=postprocessing_existing_caption_action,
            disabled_extensions=disabled_extensions,
            disable_all_extensions=disable_all_extensions,
            restore_config_state_file=restore_config_state_file,
            sd_checkpoint_hash=sd_checkpoint_hash,
            forge_unet_storage_dtype=forge_unet_storage_dtype,
            forge_inference_memory=forge_inference_memory,
            forge_async_loading=forge_async_loading,
            forge_pin_shared_memory=forge_pin_shared_memory,
            forge_preset=forge_preset,
            forge_additional_modules=forge_additional_modules,
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
