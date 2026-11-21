# Out-of-context synthesis, placement and routing of wm_top on the PYNQ-Z2 part,
# with the utilization and timing reports written to results/synth/vivado/.
#   make synth-vivado        (runs: vivado -mode batch -source syn/vivado_synth.tcl, from the repo root)
# The ROM files are $readmemh'd by relative path, so run from the repo root.
set part xc7z020clg400-1
set out  results/synth/vivado
file mkdir $out
read_verilog [glob src/rtl/*.v]
read_xdc syn/wm_top.xdc
synth_design -top wm_top -part $part -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -hierarchical -file $out/utilization_synth_hier.rpt
report_utilization -file $out/utilization_synth.rpt
opt_design
place_design
route_design
report_utilization -file $out/utilization_routed.rpt
report_timing_summary -max_paths 10 -file $out/timing_routed.rpt
report_power -file $out/power_routed.rpt
