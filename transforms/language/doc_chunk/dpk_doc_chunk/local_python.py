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

import ast
import os
import sys

from data_processing.runtime.pure_python import PythonTransformLauncher
from data_processing.utils import ParamsUtils
from dpk_doc_chunk.transform_python import DocChunkPythonTransformConfiguration
from dpk_doc_chunk.transform import chunking_types

# create parameters
input_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test-data", "input"))
# input_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test-data", "input_md"))
# input_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test-data", "input_token_text"))
output_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
local_conf = {
    "input_folder": input_folder,
    "output_folder": output_folder,
}
code_location = {"github": "github", "commit_hash": "12345", "path": "path"}
params = {
    # Data access. Only required parameters are specified
    "data_local_config": ParamsUtils.convert_to_ast(local_conf),
    "data_files_to_use": ast.literal_eval("['.parquet']"),
    # execution info
    "runtime_pipeline_id": "pipeline_id",
    "runtime_job_id": "job_id",
    "runtime_code_location": ParamsUtils.convert_to_ast(code_location),
    # doc_chunk params
    # "doc_chunk_dl_min_chunk_len": 10,  # for testing the usage of the deprecated argument
    # "doc_chunk_chunking_type": "li_markdown",
    "doc_chunk_chunking_type": "dl_json",
    # "doc_chunk_chunking_type": chunking_types.LI_TOKEN_TEXT, 
    # fixed-size params
    # "doc_chunk_output_chunk_column_name": "chunk_text",
    # "doc_chunk_chunk_size_tokens": 128,
    # "doc_chunk_chunk_overlap_tokens": 30
}
if __name__ == "__main__":
    # Set the simulated command line args
    sys.argv = ParamsUtils.dict_to_req(d=params)
    # create launcher
    launcher = PythonTransformLauncher(runtime_config=DocChunkPythonTransformConfiguration())
    # Launch the ray actor(s) to process the input
    launcher.launch()
