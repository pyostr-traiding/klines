from setuptools import setup, find_packages

setup(
    name="klines",                  # название пакета
    version="0.3.0",                # версия
    packages=find_packages(),        # автоматически найдёт папку klines
    install_requires=[               # зависимости (если есть)
        # "numpy>=1.25",
        "python-dotenv",
        "redis",
        "pydantic",
        "pandas",
        "setuptools",
    ],
    url="https://github.com/pyostr-traiding/klines",
    author="4erni pidor",
    author_email="pyostr@gmail.com",
    description="Klines module for trading",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.12',
)
