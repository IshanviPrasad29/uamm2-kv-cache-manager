set sigs [list {tb_sweep_scorer.dut.clk} {tb_sweep_scorer.dut.rst_n} {tb_sweep_scorer.dut.start} {tb_sweep_scorer.dut.busy} {tb_sweep_scorer.dut.rd_cnt[6:0]} {tb_sweep_scorer.dut.s0_v} {tb_sweep_scorer.dut.s0_data[24:0]} {tb_sweep_scorer.dut.out_valid} {tb_sweep_scorer.dut.out_idx[5:0]} {tb_sweep_scorer.dut.out_score[7:0]} {tb_sweep_scorer.dut.out_tier[1:0]} {tb_sweep_scorer.dut.out_evict} {tb_sweep_scorer.dut.done}]
gtkwave::addSignalsFromList $sigs
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_sweep_scorer.dut.rd_cnt[6:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_sweep_scorer.dut.s0_data[24:0]}
gtkwave::/Edit/Data_Format/Hex
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_sweep_scorer.dut.out_idx[5:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_sweep_scorer.dut.out_score[7:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_sweep_scorer.dut.out_tier[1:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::setZoomRangeTimes 655000 815000
gtkwave::setMarker 700000
