set sigs [list {tb_sweep_scorer.dut.clk} {tb_sweep_scorer.dut.rst_n} {tb_sweep_scorer.dut.busy} {tb_sweep_scorer.dut.rd_cnt[6:0]} {tb_sweep_scorer.dut.s0_v} {tb_sweep_scorer.dut.out_valid} {tb_sweep_scorer.dut.done}]
gtkwave::addSignalsFromList $sigs
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_sweep_scorer.dut.rd_cnt[6:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::setZoomRangeTimes 13680000 13870000
gtkwave::setMarker 13780000
