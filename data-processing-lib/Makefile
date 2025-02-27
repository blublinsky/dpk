#################################################################################################################
#
# This is the top level makefile, which is intended to be able to process a common set of rules on all 
# sub-projects underneath this directory.  Currently, the common/standardized set of rules are as follows
# and supported by .make.defaults 
#
# setup: 
# clean: 
# build:
# test:
#
# When finally getting to a makefile that requires a rule implementation, for example to test the build,
# that makefile should override/implement the rule to meet its needs.  Such a rule may continue to recurse
# using "$(MAKE) <rule>-recurse", for example "$(MAKE) test-recurse". 
#
# Each rule is called recursively on sub-directories and if a similar inclusion is done in the sub-Makefiles,
# the rules will be applied/executed recursively in their sub-directories.
#
#################################################################################################################

REPOROOT=..

# Get some common rules for the whole repo
include $(REPOROOT)/.make.defaults
include $(REPOROOT)/.make.versions

########## ########## ########## ########## ########## ########## ########## ########## 
# Global rules that are generally to be implemented in the sub-directories and can
# be overridden there (the double colon on the rule makes the overridable). 

clean:: 
	@# Help: Recursively $@ in all subdirs 
	$(MAKE) RULE=$@ .recurse

setup::
	@# Help: Recursively $@ in all subdirs
	@$(MAKE) RULE=$@ .recurse

build:: 
	@# Help: Recursively $@ in all subdirs 
	$(MAKE) RULE=$@ .recurse

test::  
	@# Help: Recursively $@ in all subdirs 
	@$(MAKE) RULE=$@ .recurse

image::  
	@# Help: Recursively $@ in all subdirs 
	@$(MAKE) RULE=$@ .recurse

publish::  
	@# Help: Recursively $@ in all subdirs 
	@$(MAKE) RULE=$@ .recurse

set-versions:  
	@# Help: Recursively $@ in all subdirs 
	$(MAKE) TOML_VERSION=$(DPK_LIB_VERSION) .defaults.update-toml
	@$(MAKE) RULE=$@ .recurse


build-pkg-dist:: 
	$(MAKE) .defaults.build-dist BUILD_WHEEL_EXTRA_ARG=-w

publish-dist :: .defaults.publish-dist

