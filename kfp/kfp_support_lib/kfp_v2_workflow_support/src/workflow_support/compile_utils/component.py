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

import json
import os
from typing import Dict

import kfp.dsl as dsl
from data_processing.utils import get_logger

from kfp import kubernetes


logger = get_logger(__name__)


RUN_NAME = "KFP_RUN_NAME"

# Default path for KFP component specification files
DEFAULT_KFP_COMPONENT_SPEC_PATH = "../../../../kfp/kfp_ray_components/"

ONE_HOUR_SEC = 60 * 60
ONE_DAY_SEC = ONE_HOUR_SEC * 24
ONE_WEEK_SEC = ONE_DAY_SEC * 7


class ComponentUtils:
    """
    Class containing methods supporting building pipelines
    """

    @staticmethod
    def add_settings_to_component(
        task: dsl.PipelineTask,
        timeout: int,
        image_pull_policy: str = "IfNotPresent",
        image_pull_secrets: list = [],
        cache_strategy: bool = False,
    ) -> None:
        """
        Add settings to kfp task
        :param task: kfp task
        :param timeout: timeout to set to the component in seconds
        :param image_pull_policy: pull policy to set to the component
        :param image_pull_secrets: list of secrets containing the credentials to pull the task image.
        :param cache_strategy: cache strategy
        """

        def _add_tolerations() -> None:
            try:
                tolerations = os.getenv("KFP_TOLERATIONS", "")
                if tolerations != "":
                    # TODO: apply the tolerations defined as env vars to ray pods.
                    # Currently they can be specified in the pipeline params:
                    # ray_head_options and ray_worker_options.

                    tolerations = json.loads(tolerations)
                    for toleration in tolerations:
                        kubernetes.add_toleration(
                            task,
                            key=toleration["key"],
                            operator=toleration["operator"],
                            value=toleration["value"],
                            effect=toleration["effect"],
                        )

            except Exception as e:
                logger.warning(f"Exception while handling tolerations {e}")

        def _add_node_selector() -> None:
            try:
                node_selector = os.getenv("KFP_NODE_SELECTOR", "")
                if node_selector != "":
                    print(f"Note: Applying node_selector {node_selector} to kubeflow pipelines pods")
                    node_selector = json.loads(node_selector)
                    kubernetes.add_node_selector(
                        task,
                        label_key=node_selector["label_key"],
                        label_value=node_selector["label_value"],
                    )
            except Exception as e:
                logger.warning(f"Exception while handling node_selector {e}")

        kubernetes.use_field_path_as_env(
            task, env_name=RUN_NAME, field_path="metadata.annotations['pipelines.kubeflow.org/run_name']"
        )
        # Set cashing
        task.set_caching_options(enable_caching=cache_strategy)
        # image pull policy
        kubernetes.set_image_pull_policy(task, image_pull_policy)
        # image pull secret can only be set at the component level in kfp v2
        # see: https://github.com/kubeflow/pipelines/issues/11498
        if len(image_pull_secrets) != 0:
            kubernetes.set_image_pull_secrets(task, image_pull_secrets)
        # Set the timeout for the task to one day (in seconds)
        kubernetes.set_timeout(task, seconds=timeout)
        # Add tolerations if specified
        _add_tolerations()
        # Add node_selector if specified
        _add_node_selector()

    @staticmethod
    def set_s3_env_vars_to_component(
        task: dsl.PipelineTask,
        secret: str = "",
        env2key: Dict[str, str] = None,
        prefix: str = None,
    ) -> None:
        """
        Set S3 env variables to KFP component
        :param task: kfp task
        :param secret: secret name with the S3 credentials
        :param env2key: dict with mapping each env variable to a key in the secret
        :param prefix: prefix to add to env name
        """
        if env2key is None:
            env2key = {"s3-key": "S3_KEY", "s3-secret": "S3_SECRET", "s3-endpoint": "ENDPOINT"}

        if prefix is not None:
            for secret_key, _ in env2key.items():
                env_name = env2key.pop(secret_key)
                env_name = f"{prefix}_{env_name}"
                env2key[secret_key] = env_name
        # FIXME: see https://github.com/kubeflow/pipelines/issues/10914
        kubernetes.use_secret_as_env(task=task, secret_name="s3-secret", secret_key_to_env=env2key)
