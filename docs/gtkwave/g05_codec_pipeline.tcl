set sigs [list {tb_kv_codec.uq.clk} {tb_kv_codec.uq.in_valid} {tb_kv_codec.uq.zp[7:0]} {tb_kv_codec.uq.inv_q12[15:0]} {tb_kv_codec.uq.x_flat[127:0]} {tb_kv_codec.uq.out_valid} {tb_kv_codec.uq.q_flat[63:0]} {tb_kv_codec.ud.step_q8[15:0]} {tb_kv_codec.ud.out_valid} {tb_kv_codec.ud.x_flat[127:0]}]
gtkwave::addSignalsFromList $sigs
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_kv_codec.uq.zp[7:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_kv_codec.uq.inv_q12[15:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_kv_codec.uq.x_flat[127:0]}
gtkwave::/Edit/Data_Format/Hex
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_kv_codec.uq.q_flat[63:0]}
gtkwave::/Edit/Data_Format/Hex
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_kv_codec.ud.step_q8[15:0]}
gtkwave::/Edit/Data_Format/Decimal
gtkwave::/Edit/UnHighlight_All
gtkwave::highlightSignalsFromList {tb_kv_codec.ud.x_flat[127:0]}
gtkwave::/Edit/Data_Format/Hex
gtkwave::/Edit/UnHighlight_All
gtkwave::setZoomRangeTimes 15000 115000
gtkwave::setMarker 50000
