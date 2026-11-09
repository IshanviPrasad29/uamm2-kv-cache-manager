# Hardware regression. Needs Verilator >= 5 (pip install verilator, or your distro package)
# and a C++20 compiler. Usage:  make -C hw test
VERILATOR ?= verilator
VFLAGS    ?= --binary --timing -Wno-fatal -Wno-lint -CFLAGS "-std=c++20 -fcoroutines"
PY        ?= python3
BUILD     ?= build

.PHONY: all vectors test lint clean
all: test

vectors:
	cd .. && $(PY) -c "from uamm2.hwmodel import write_vectors; write_vectors('hw/vectors')"

define RUN
	$(VERILATOR) $(VFLAGS) --top-module tb_$(1) -Mdir $(BUILD)/$(1) $(2) -o sim >$(BUILD)/$(1).log 2>&1 || (cat $(BUILD)/$(1).log; exit 1)
	$(BUILD)/$(1)/sim | tee $(BUILD)/$(1).out | grep -E "PASS|FAIL"
	@grep -q PASS $(BUILD)/$(1).out
endef

test: vectors
	@mkdir -p $(BUILD)
	$(call RUN,sweep_scorer,rtl/uamm_sweep_scorer.v tb/tb_sweep_scorer.v)
	$(call RUN,kv_codec,rtl/uamm_kv_codec.v tb/tb_kv_codec.v)
	$(call RUN,block_table,rtl/uamm_block_table.v tb/tb_block_table.v)

lint:
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module uamm_sweep_scorer rtl/uamm_sweep_scorer.v
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module uamm_kv_quant4 rtl/uamm_kv_codec.v
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module uamm_kv_dequant4 rtl/uamm_kv_codec.v
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME --top-module uamm_block_table rtl/uamm_block_table.v

clean:
	rm -rf $(BUILD)
