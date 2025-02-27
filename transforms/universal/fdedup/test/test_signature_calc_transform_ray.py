# (C) Copyright IBM Corp. 2024.
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#  http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import os

from data_processing.test_support.launch.transform_test import (
    AbstractTransformLauncherTest,
)
from data_processing.utils import ParamsUtils
from data_processing_ray.runtime.ray import RayTransformLauncher
from dpk_fdedup.signature_calc.transform import (
    num_bands_cli_param,
    num_permutations_cli_param,
    num_segments_cli_param,
)
from dpk_fdedup.signature_calc.ray.transform import SignatureCalculationRayTransformConfiguration


class TestRaySignatureCalcTransform(AbstractTransformLauncherTest):
    """
    Extends the super-class to define the test data for the tests defined there.
    The name of this class MUST begin with the word Test so that pytest recognizes it as a test class.
    """

    def get_test_transform_fixtures(self) -> list[tuple]:
        basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ray/test-data"))
        config = {
            "run_locally": True,
            num_permutations_cli_param: 112,
            num_bands_cli_param: 14,
            num_segments_cli_param: 2,
        }
        launcher = RayTransformLauncher(SignatureCalculationRayTransformConfiguration())
        fixtures = [
            (launcher, config, os.path.join(basedir, "input"), os.path.join(basedir, "expected", "signature_calc"))
        ]
        return fixtures
