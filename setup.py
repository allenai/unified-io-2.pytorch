# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)

with open(os.path.join(_PATH_ROOT, "README.md")) as fo:
    readme = fo.read()

with open(os.path.join(_PATH_ROOT, "requirements.txt")) as fo:
    requirements = [x.strip().split()[0] for x in fo.readlines() if x.strip()]


setup(
    name="Unified-IO-2-PyTorch",
    version="0.1.0",
    description="A multi-task multi-modal model",
    author="UnifiedIO Team",
    url="https://github.com/allenai/unified-io-2.pytorch",
    install_requires=requirements,
    packages=find_packages(),
    long_description=readme,
    long_description_content_type="text/markdown",
)