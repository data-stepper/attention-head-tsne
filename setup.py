from setuptools import setup, find_packages

setup(
    name="ah_tsne",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # List your package dependencies here
    ],
    entry_points={
        "console_scripts": [
            "ah-tsne = ah_tsne.__main__:main",
        ],
    },
)
