# Telegram Generator Bot

A Telegram bot for Generative interfaces.

## Interfaces

* [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
* [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)

## Getting Started

### Installing

```bash
git clone https://github.com/goodarzi/telegram-generator-bot.git
cd telegram-generator-bot
pip3 install -r requirements.txt
cp config-copy.yaml config.yaml
```

### Configuration

1. create config.yaml
   ```sh
   cp config-copy.yaml config.yaml
   ```
2. Enter your configs in `config.yaml`


### Running
```bash
python3 generator_bot.py
```

<!-- ROADMAP -->
## Roadmap

- [x] Stable Diffusion web UI
    - [x] Text to Image
    - [x] Image to Image
        - [x] Image to Text (CLIP & Deepdanbru)
        - [x] Sketch
        - [ ] Inpaint
        - [ ] Inpaint Sketch
    - [x] PNG info
- [x] Stable Diffusion WebUI Forge
    - [x] Forge preset
    - [x] Text to Image
    - [x] Image to Image
        - [x] Image to Text (CLIP & Deepdanbru)
        - [x] Sketch
        - [ ] Inpaint
        - [ ] Inpaint Sketch
    - [x] PNG info
- [ ] Text generation web UI
- [ ] ComfyUI

## Community
Telegram: [tg_generator_bot](https://t.me/tg_generator_bot)
## Contributing
Any contributions you make are appreciated.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/feature_name`)
3. Commit your Changes (`git commit -m 'Add some feature_name'`)
4. Push to the Branch (`git push origin feature/feature_name`)
5. Open a Pull Request
