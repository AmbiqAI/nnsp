local_src := $(wildcard $(subdirectory)/src/*.c)
includes_api += $(subdirectory)/includes-api
includes_api += extern/CMSIS/CMSIS_5-5.9.0/CMSIS/Core/Include
includes_api += extern/CMSIS/CMSIS_5-5.9.0/CMSIS/NN/Include
includes_api += extern/CMSIS/CMSIS_5-5.9.0/CMSIS/DSP/Include
includes_api += extern/CMSIS/CMSIS_5-5.9.0/CMSIS/DSP/PrivateInclude

local_bin := $(BINDIR)/$(subdirectory)
bindirs   += $(local_bin)
$(eval $(call make-library, $(local_bin)/ns-nnsp.a, $(local_src)))
