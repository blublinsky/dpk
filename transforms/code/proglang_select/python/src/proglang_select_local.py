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

from data_processing.data_access import DataAccessFactory, DataAccessLocal
from proglang_select_transform import (
    ProgLangSelectTransform,
    lang_allowed_langs_file_key,
    lang_data_factory_key,
    lang_lang_column_key,
    lang_output_column_key,
)


# create parameters
language_column_name = "language"
annotated_column_name = "lang_selected"

selected_languages_file = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../test-data/languages/allowed-code-languages.txt")
)
input_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../test-data/input"))

params = {
    lang_allowed_langs_file_key: selected_languages_file,
    lang_lang_column_key: language_column_name,
    lang_output_column_key: annotated_column_name,
    lang_data_factory_key: DataAccessFactory(),  # Expect to create DataAccessLocal
}
if __name__ == "__main__":
    # Here we show how to run outside of ray
    # Create and configure the transform.
    transform = ProgLangSelectTransform(params)
    # Use the local data access to read a parquet table.
    data_access = DataAccessLocal()
    table, _ = data_access.get_table(os.path.join(input_folder, "test1.parquet"))
    print(f"input table: {table}")
    # Transform the table
    table_list, metadata = transform.transform(table)
    print(f"\noutput table: {table_list}")
    print(f"output metadata : {metadata}")
