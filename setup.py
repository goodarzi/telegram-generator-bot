from distutils.core import setup

# from utils import __version__

setup(
    name="telegram_generator_bot",
    version="0.0.1",
    author="Hamid Goodarzi",
    description="A telegram chatbot to generate text and image using generative interfaces",
    url="https://github.com/goodarzi/telegram-generator-bot",
    download_url="https://github.com/goodarzi/telegram-generator-bot/releases/latest",
    py_modules=["telegram_generator_bot"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Internet",
        "Topic :: Communications",
        "Topic :: Communications :: Chat",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={
        "Tracker": "https://github.com/goodarzi/telegram-generator-bot/issues",
        "Community": "https://t.me/TelegramGeneratorBot",
        "Source": "https://github.com/goodarzi/telegram-generator-bot",
    },
    python_requires="~=3.11",
)
