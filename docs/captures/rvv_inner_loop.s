kv_qk_dot_kivi4_rvv:                    # @kv_qk_dot_kivi4_rvv
# %bb.0:
	vsetvli	a6, zero, e32, m1, ta, ma
	blez	a3, .LBB2_6
# %bb.1:
	li	a6, 0
	srliw	a7, a4, 31
	add	a7, a7, a4
	sraiw	a7, a7, 1
	vmv.v.i	v8, 0
	li	t0, 2
	vmv.s.x	v9, zero
	j	.LBB2_3
.LBB2_2:                                #   in Loop: Header=BB2_3 Depth=1
	vsetvli	t1, zero, e32, m1, ta, ma
	vfredusum.vs	v10, v10, v9
	vfmv.f.s	fa5, v10
	fadd.s	fa5, fa5, fa0
	slli	t1, a6, 2
	add	t1, t1, a5
	addi	a6, a6, 1
	fsw	fa5, 0(t1)
	beq	a6, a3, .LBB2_6
.LBB2_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_5 Depth 2
	vmv1r.v	v10, v8
	blt	a4, t0, .LBB2_2
# %bb.4:                                #   in Loop: Header=BB2_3 Depth=1
	li	t1, 0
	mul	t2, a6, a7
	add	t2, t2, a2
	vmv1r.v	v10, v8
.LBB2_5:                                #   Parent Loop BB2_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	subw	t3, a7, t1
	vsetvli	t3, t3, e8, mf4, ta, ma
	add	t4, t2, t1
	vle8.v	v11, (t4)
	vand.vi	v12, v11, 15
	vsrl.vi	v11, v11, 4
	vsetvli	zero, zero, e32, m1, ta, ma
	vzext.vf4	v13, v12
	vfcvt.f.xu.v	v12, v13
	vzext.vf4	v13, v11
	vfcvt.f.xu.v	v11, v13
	slli	t4, t1, 2
	add	t5, a0, t4
	vle32.v	v13, (t5)
	add	t4, t4, a1
	vle32.v	v14, (t4)
	vsetvli	zero, zero, e32, m1, tu, ma
	vfmacc.vv	v10, v12, v13
	addw	t1, t1, t3
	vfmacc.vv	v10, v11, v14
	blt	t1, a7, .LBB2_5
	j	.LBB2_2
.LBB2_6:
	ret
