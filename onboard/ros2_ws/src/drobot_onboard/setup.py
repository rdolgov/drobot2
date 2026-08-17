from glob import glob

from setuptools import find_packages, setup


PACKAGE_NAME = "drobot_onboard"


setup(
    name=PACKAGE_NAME,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE_NAME}"]),
        (f"share/{PACKAGE_NAME}", ["package.xml"]),
        (f"share/{PACKAGE_NAME}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE_NAME}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Roman",
    maintainer_email="roman@localhost",
    description="ROS 2 onboard motor control and LAN dashboard for Drobot2",
    license="Proprietary",
    entry_points={
        "console_scripts": [
            "onboard_control = drobot_onboard.node:main",
        ],
    },
)
