set sigs [list {tb_block_table.dut.clk} {tb_block_table.dut.we} {tb_block_table.dut.waddr[7:0]} {tb_block_table.dut.w_valid} {tb_block_table.dut.w_tier[1:0]} {tb_block_table.dut.w_codec[1:0]} {tb_block_table.dut.w_phys[11:0]} {tb_block_table.dut.rd_en} {tb_block_table.dut.raddr[7:0]} {tb_block_table.dut.rd_valid} {tb_block_table.dut.hit} {tb_block_table.dut.tier[1:0]} {tb_block_table.dut.codec[1:0]} {tb_block_table.dut.phys[11:0]}]
gtkwave::addSignalsFromList $sigs
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_block_table.dut.waddr[7:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_block_table.dut.w_tier[1:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_block_table.dut.w_codec[1:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_block_table.dut.w_phys[11:0]}
gtkwave::/Edit/Data_Format/Hex
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_block_table.dut.raddr[7:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_block_table.dut.tier[1:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_block_table.dut.codec[1:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_block_table.dut.phys[11:0]}
gtkwave::/Edit/Data_Format/Hex
gtkwave::/Edit/UnHighlight_All
gtkwave::setZoomRangeTimes 1105000 1285000
gtkwave::setMarker 1185000
