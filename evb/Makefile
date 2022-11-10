include make/helpers.mk
include make/neuralspot_config.mk
include make/neuralspot_toolchain.mk
include make/jlink.mk
include autogen.mk

sources:=src/def_nn0_s2i.c
sources+=src/def_nn1_vad.c
sources+=src/def_nn2_kws_galaxy.c
sources+=src/PcmBufClass.c
sources+=src/arm_intrinsic_test.c
sources+=src/$(local_app_name).cc
ifeq ($(NNSP_MODE),1)
sources+=src/nnCntrlClass.c
else
sources+=src/s2iCntrlClass.c
endif
targets  := $(BINDIR)/$(local_app_name).axf
targets  += $(BINDIR)/$(local_app_name).bin
mains    += $(BINDIR)/src/$(local_app_name).o

objects      = $(filter-out $(mains),$(call source-to-object,$(sources)))
dependencies = $(subst .o,.d,$(objects))

CFLAGS     += $(addprefix -D,$(DEFINES))
CFLAGS     += $(addprefix -I includes/,$(INCLUDES))
LINKER_FILE := libs/linker_script.ld

all: $(BINDIR) $(objects) $(targets)

.PHONY: clean
clean:
ifeq ($(OS),Windows_NT)
	@echo "Windows_NT"
	@echo $(Q) $(RM) -rf $(BINDIR)/*
	$(Q) $(RM) -rf $(BINDIR)/*
else
	$(Q) $(RM) -rf $(BINDIR) $(JLINK_CF)
endif

ifneq "$(MAKECMDGOALS)" "clean"
  include $(dependencies)
endif

$(BINDIR):
	@mkdir -p $@

$(BINDIR)/%.o: %.cc
	@echo " Compiling $(COMPILERNAME) $< to make $@"
	@mkdir -p $(@D)
	$(Q) $(CC) -c $(CFLAGS) $(CCFLAGS) $< -o $@

$(BINDIR)/%.o: %.cpp
	@echo " Compiling $(COMPILERNAME) $< to make $@"
	@mkdir -p $(@D)
	$(Q) $(CC) -c $(CFLAGS) $(CCFLAGS) $< -o $@

$(BINDIR)/%.o: %.c
	@echo " Compiling $(COMPILERNAME) $< to make $@"
	@mkdir -p $(@D)
	$(Q) $(CC) -c $(CFLAGS) $(CONLY_FLAGS) $< -o $@

$(BINDIR)/%.o: %.s
	@echo " Assembling $(COMPILERNAME) $<"
	@mkdir -p $(@D)
	$(Q) $(CC) -c $(CFLAGS) $< -o $@

%.axf: src/%.o
	@echo " Linking $(COMPILERNAME) $@"
	@mkdir -p $(@D)
	$(Q) $(CC) -Wl,-T,$(LINKER_FILE) -o $@ $< $(objects) $(LFLAGS)

%.bin: %.axf 
	@echo " Copying $(COMPILERNAME) $@..."
	@mkdir -p $(@D)
	$(Q) $(CP) $(CPFLAGS) $< $@
	$(Q) $(OD) $(ODFLAGS) $< > $*.lst
	$(Q) $(SIZE) $(objects) $(lib_prebuilt) $< > $*.size

$(JLINK_CF):
	@echo " Creating JLink command sequence input file..."
	$(Q) echo "ExitOnError 1" > $@
	$(Q) echo "Reset" >> $@
	$(Q) echo "LoadFile $(BINDIR)/$(TARGET).bin, $(JLINK_PF_ADDR)" >> $@
	$(Q) echo "Exit" >> $@

.PHONY: deploy
deploy: $(JLINK_CF)
	@echo " Deploying $< to device (ensure JLink USB connected and powered on)..."
	$(Q) $(JLINK) $(JLINK_CMD)

.PHONY: view
view:
	@echo " Printing SWO output (ensure JLink USB connected and powered on)..."
	$(Q) $(JLINK_SWO) $(JLINK_SWO_CMD)

%.d: ;
.PHONY: print
print: 
	@echo $(sources)
	@echo $(mains)
	@echo $(objects)
