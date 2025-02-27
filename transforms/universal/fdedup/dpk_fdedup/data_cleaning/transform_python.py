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
from typing import Any

from dpk_fdedup.data_cleaning.transform import (
    DataCleaningTransformConfiguration,
    dataclean_data_access_key,
    dataclean_data_factory_key,
    duplicate_list_location_default,
    duplicate_list_location_key,
)
from data_processing.data_access import DataAccessFactoryBase
from data_processing.runtime.pure_python import PythonTransformLauncher
from data_processing.runtime.pure_python.runtime_configuration import (
    DefaultPythonTransformRuntime,
    PythonTransformRuntimeConfiguration,
)
from data_processing.transform import TransformStatistics
from data_processing.utils import get_logger


logger = get_logger(__name__)


class DataCleaningPythonRuntime(DefaultPythonTransformRuntime):
    """
    Data cleaning runtime support for Python
    """

    def __init__(self, params: dict[str, Any]):
        super().__init__(params=params)
        self.logger = get_logger(__name__)

    def get_transform_config(
        self, data_access_factory: list[DataAccessFactoryBase], statistics: TransformStatistics, files: list[str]
    ) -> dict[str, Any]:
        """
        Download the table of duplicate document ids that will be provided to the
        filtering/annotation method. This is the opportunity for this runtime to
        create a new set of configuration based on the config/params provided to
        this instance's initializer. This may include the addition of new
        configuration data such as ray shared memory, new actors, etc., that
        might be needed and expected by the transform in its initializer and/or
        transform() methods.
        :param data_access_factory - data access factory class being used by the RayOrchestrator.
        :param statistics - reference to statistics actor
        :param files - list of files to process
        :return: dictionary of transform init params
        """
        data_access_out = data_access_factory[1].create_data_access()
        dc_data_access = self.params.get(dataclean_data_access_key, None)
        if dc_data_access is None:
            dc_daf = self.params.get(dataclean_data_factory_key, None)
            if dc_daf is None:
                raise RuntimeError(f"Missing configuration value for key {dataclean_data_factory_key}")
            dc_data_access = dc_daf.create_data_access()
        if dc_data_access.get_output_folder() is None:
            dc_data_access.output_folder = data_access_out.get_output_folder()
        duplicate_list_location = self.params.get(duplicate_list_location_key, duplicate_list_location_default)
        if not duplicate_list_location.startswith("/"):
            out_paths = dc_data_access.output_folder.rstrip("/").split("/")
            dupl_list_paths = duplicate_list_location.split("/")
            paths = out_paths[:-1] + dupl_list_paths
            duplicate_list_location = "/".join([p.strip("/") for p in paths])
        if duplicate_list_location.startswith("s3://"):
            _, duplicate_list_location = duplicate_list_location.split("://")
        self.duplicate_list, retries = dc_data_access.get_file(duplicate_list_location)
        return self.params | {"df": self.duplicate_list}


class DataCleaningPythonTransformConfiguration(PythonTransformRuntimeConfiguration):
    """
    Implements the PythonTransformConfiguration for fuzzy dedup data cleaning step
    as required by the PythonTransformLauncher.
    """

    def __init__(self):
        """
        Initialization
        :param: transform_configuration - transform configuration class
        :param: runtime_class - name of the runtime configuration class
        """
        super().__init__(
            transform_config=DataCleaningTransformConfiguration(),
            runtime_class=DataCleaningPythonRuntime,
        )


if __name__ == "__main__":
    launcher = PythonTransformLauncher(DataCleaningTransformConfiguration())
    logger.info("Launching fuzzy dedup data cleaning transform")
    launcher.launch()
