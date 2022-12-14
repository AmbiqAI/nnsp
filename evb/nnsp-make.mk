INCLUDES += neuralspot/ns-nnsp/includes-api
libraries += libs/ns-nnsp.a

ACC32_OPT:=0		# 1: accumulator 32bit, 0: acc 64bit
NNSP_MODE:=0        # 1: run vad+kws+s2i, 0: s2i only
ifeq ($(NNSP_MODE),1)
local_app_name:=main_nnsp
else
local_app_name:=main_s2i
endif
TARGET:=$(local_app_name)