from setuptools import setup, find_packages

setup(
    name="entry-extraction",
    version="0.1.0",
    description="Extract and group semantically related sentences from text documents",
    author="Reporter.ai",
    author_email="reporter.ai@boldcode.io",
    packages=find_packages(),
    install_requires=[
        "setfit",
        "nltk",
        "tqdm",
        "punctuators",
        "torch",
        "pandas",
        "datasets",
        "huggingface_hub",
        "sentence-transformers",
    ],
    python_requires=">=3.7",
)