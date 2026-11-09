set sigs [list {tb_sweep_scorer.dut.clk} {tb_sweep_scorer.dut.pres[7:0]} {tb_sweep_scorer.dut.out_valid} {tb_sweep_scorer.dut.out_idx[5:0]} {tb_sweep_scorer.dut.out_score[7:0]} {tb_sweep_scorer.dut.out_tier[1:0]} {tb_sweep_scorer.dut.out_evict}]
gtkwave::addSignalsFromList $sigs
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_sweep_scorer.dut.pres[7:0]}
gtkwave::/Edit/Data_Format/Decimal
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
gtkwave::setZoomRangeTimes 2025000 2185000
gtkwave::setMarker 2085000
