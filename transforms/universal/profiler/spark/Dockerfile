ARG BASE_IMAGE=quay.io/dataprep1/data-prep-kit/data-prep-kit-spark-3.5.2:latest
FROM ${BASE_IMAGE}

USER root
# install pytest
RUN pip install --no-cache-dir pytest

WORKDIR ${SPARK_HOME}/work-dir
ARG DPK_WHEEL_FILE_NAME

# Copy and install data processing libraries 
# These are expected to be placed in the docker context before this is run (see the make image).
COPY --chown=dpk:root data-processing-dist data-processing-dist
RUN  pip install data-processing-dist/${DPK_WHEEL_FILE_NAME}[spark]

COPY --chown=spark:root python-transform/  python-transform/
RUN cd python-transform && pip install --no-cache-dir -e .

COPY --chown=root:root src/ src/
COPY --chown=root:root pyproject.toml pyproject.toml 
RUN pip install --no-cache-dir -e .

# copy in the main() entry point to the image 
COPY ./src/profiler_transform_spark.py .

# Copy in some samples
COPY ./src/profiler_local_spark.py local/

# copy test
COPY test/ test/
COPY test-data/ test-data/

USER spark

# Set environment
ENV PYTHONPATH=${SPARK_HOME}/work-dir/:${PYTHONPATH}

# Put these at the end since they seem to upset the docker cache.
ARG BUILD_DATE
ARG GIT_COMMIT
LABEL build-date=$BUILD_DATE
LABEL git-commit=$GIT_COMMIT


