# Every result in results/ is produced by one of these targets.
PY ?= python3

.PHONY: all test roms golden units rtl stats perf baseline llm replay mutate synth-vivado synth-yosys clean

all: test

roms:            ## ROM hex files the RTL loads, from the golden model
	$(PY) scripts/gen_roms.py

golden:          ## golden model self-test
	$(PY) src/golden/wm_ref.py

units: roms      ## arithmetic blocks vs golden          -> results/unit_tests.md
	$(PY) scripts/test_units.py

rtl: roms        ## scenarios + errors, twice            -> results/rtl_tests.md
	$(PY) scripts/test_rtl.py

stats: roms      ## sampling distribution through RTL    -> results/stats_rtl.md
	$(PY) scripts/stats_rtl.py

perf: roms       ## latency and throughput               -> results/perf.md
	$(PY) scripts/perf.py

baseline: perf   ## C and NumPy baselines                -> results/baseline.md
	$(PY) scripts/baseline.py

test: golden units rtl stats perf

llm:             ## real-model experiment, needs the local llama.cpp server on :8081 -> results/llm_eval.md
	$(PY) scripts/llm_eval.py

replay: roms     ## recorded model steps through the RTL -> results/llm_rtl_replay.md
	$(PY) scripts/replay_llm_rtl.py

mutate: roms     ## planted-bug sweep (slow)             -> results/mutations.md
	$(PY) scripts/mutation_test.py

synth-vivado: roms   ## needs Vivado on PATH              -> results/synth/vivado/
	vivado -mode batch -nojournal -nolog -source syn/vivado_synth.tcl

synth-yosys: roms    ## needs yosys on PATH               -> results/synth/yosys/stat.txt
	mkdir -p results/synth/yosys
	yosys -q -s syn/yosys_synth.ys

clean:
	rm -rf build .Xil
