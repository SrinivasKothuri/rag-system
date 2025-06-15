from setuptools import setup, find_packages

setup(
    name="rag_system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "pyyaml>=6.0.1",
        "requests>=2.31.0",
    ],
    python_requires=">=3.8",
    author="Your Name",
    author_email="lssrinivas.kothuri@gmail.com",
    description="A simple but powerful RAG system",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/SrinivasKothuri/rag-system",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
