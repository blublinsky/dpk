# Define the root of the local git clone for the common rules to be able 
# know where they are running from.
REPOROOT=../../../..

# Set this, before including .make.defaults, to 
#   1 if requirements reference the latest code in the data processing library 
#     in this repo (that is not yet published to pypi).	 This is the default setting.
#   0 if the transforms DPK dependencies are on wheels published to 
#     pypi (e.g. data-prep-toolkit=0.2.1)
#USE_REPO_LIB_SRC=1

# Include a library of common .transform.* targets which most
# transforms should be able to reuse.  However, feel free
# to override/redefine the rules below. 
include $(REPOROOT)/transforms/.make.transforms

# Include the common configuration for this transform
include ../transform.config

BASE_IMAGE=$(RAY_BASE_IMAGE)

venv::	.transforms.ray-venv

test::	.transforms.ray-test

clean:: .transforms.clean

image:: .transforms.ray-image

test-src:: .transforms.test-src

setup:: .transforms.setup

#test-image:: .transforms.ray-test-image
test-image:: .transforms.ray-image .transforms.test-image-help .transforms.clean

build:: build-dist image

publish:: publish-image

publish-image:: .transforms.publish-image-ray

setup:: .transforms.setup

# TRANSFORM_PYTHON_VERSION has no effect since requirements do not specify a python transform implementation
set-versions: 
	$(MAKE) TRANSFORM_PYTHON_VERSION=dummy TOML_VERSION=$(REPO_LVL_ORDER_RAY_VERSION) .transforms.set-versions 
        
build-dist:: .defaults.build-dist 

publish-dist:: .defaults.publish-dist

run-cli-sample: .transforms.run-cli-ray-sample

run-local-sample: .transforms.run-local-ray-sample

run-s3-sample: .transforms.run-s3-ray-sample

minio-start:	.minio-start

load-image:: .transforms.load-image

kind-load-image:: .transforms.kind-load-image

docker-load-image: .defaults.docker-load-image

docker-save-image: .defaults.docker-save-image

