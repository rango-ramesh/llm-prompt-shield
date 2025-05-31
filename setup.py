from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-prompt-shield",
    version="0.1.1",
    author="Rango Ramesh",
    author_email="rango@celestials.ai",
    description="Lightweight prompt injection detection and blocking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rango-ramesh/llm-prompt-shield",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "PyYAML>=5.0.0",
        "numpy>=1.20.0",
        "joblib>=1.0.0",
        "sentence-transformers>=2.0.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
    ],
    extras_require={
        "langchain": [
            "langchain>=0.0.200",
        ],
        "autogen": [
            "pyautogen>=0.2.0",
        ],
        "integrations": [
            "langchain>=0.0.200",
            "pyautogen>=0.2.0",
        ]
    },
    include_package_data=True,
    package_data={
        "llm_prompt_shield": ["data/*.yaml", "*.yaml"],
    },
)