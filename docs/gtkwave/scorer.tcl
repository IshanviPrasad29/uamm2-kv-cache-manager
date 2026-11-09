gtkwave::addSignalsFromList [list \
 {tb_sweep_scorer.dut.clk} {tb_sweep_scorer.dut.rst_n} {tb_sweep_scorer.dut.start} {tb_sweep_scorer.dut.busy} \
 {tb_sweep_scorer.dut.rd_cnt[6:0]} {tb_sweep_scorer.dut.out_valid} {tb_sweep_scorer.dut.out_idx[5:0]} \
 {tb_sweep_scorer.dut.out_score[7:0]} {tb_sweep_scorer.dut.out_tier[1:0]} {tb_sweep_scorer.dut.out_evict} {tb_sweep_scorer.dut.done}]
foreach s {{tb_sweep_scorer.dut.rd_cnt[6:0]} {tb_sweep_scorer.dut.out_idx[5:0]} {tb_sweep_scorer.dut.out_score[7:0]} {tb_sweep_scorer.dut.out_tier[1:0]}} {
  gtkwave::/Edit/UnHighlight_All
  gtkwave::highlightSignalsFromList $s
  gtkwave::/Edit/Data_Format/Decimal
}
gtkwave::/Edit/UnHighlight_All
gtkwave::setZoomRangeTimes 600000 1000000
gtkwave::setMarker 700000
