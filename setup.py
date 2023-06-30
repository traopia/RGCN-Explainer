from setuptools import setup

setup(
    name="kgbench",
    version="0.2",
    description="A set of benchmark datasets for knowledge graph node classification",
    url="https://kgbench.info",
    author="Peter Bloem (Vrije Universiteit), Xander Wilcke (Vrije Universiteit), Lucas van Berkel, Victor de Boer (Vrije Universiteit)",
    author_email="kgbench@peterbloem.nl",
    packages=["kgbench"],
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "Pillow",
        "scikit-image",
        "rdflib",
        "wget",
        "fire"
        # "dgl",
        # "pyg"
    ],
    zip_safe=False
)
