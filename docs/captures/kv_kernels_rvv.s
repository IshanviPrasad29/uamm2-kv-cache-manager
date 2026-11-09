	.text
	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"kv_kernels.c"
	.globl	kv_qk_prepare                   # -- Begin function kv_qk_prepare
	.p2align	1
	.type	kv_qk_prepare,@function
kv_qk_prepare:                          # @kv_qk_prepare
# %bb.0:
	blez	a3, .LBB0_5
# %bb.1:
	li	a6, 0
	fmv.w.x	fa0, zero
	j	.LBB0_3
.LBB0_2:                                #   in Loop: Header=BB0_3 Depth=1
	flw	fa5, 0(a0)
	flw	fa4, 0(a2)
	flw	fa3, 0(a1)
	fmadd.s	fa0, fa5, fa4, fa0
	fmul.s	fa5, fa5, fa3
	srliw	t0, a6, 1
	slli	t0, t0, 2
	add	a7, a7, t0
	fsw	fa5, 0(a7)
	addi	a6, a6, 1
	addi	a1, a1, 4
	addi	a2, a2, 4
	addi	a0, a0, 4
	beq	a3, a6, .LBB0_6
.LBB0_3:                                # =>This Inner Loop Header: Depth=1
	andi	t0, a6, 1
	mv	a7, a4
	beqz	t0, .LBB0_2
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=1
	mv	a7, a5
	j	.LBB0_2
.LBB0_5:
	fmv.w.x	fa0, zero
.LBB0_6:
	ret
.Lfunc_end0:
	.size	kv_qk_prepare, .Lfunc_end0-kv_qk_prepare
                                        # -- End function
	.globl	kv_qk_dot_kivi4_scalar          # -- Begin function kv_qk_dot_kivi4_scalar
	.p2align	1
	.type	kv_qk_dot_kivi4_scalar,@function
kv_qk_dot_kivi4_scalar:                 # @kv_qk_dot_kivi4_scalar
# %bb.0:
	blez	a3, .LBB1_6
# %bb.1:
	li	a6, 0
	srliw	a7, a4, 31
	add	a7, a7, a4
	sraiw	a7, a7, 1
	slli	t0, a7, 32
	srli	t0, t0, 30
	add	t0, t0, a1
	fmv.w.x	fa5, zero
	li	t1, 2
	j	.LBB1_3
.LBB1_2:                                #   in Loop: Header=BB1_3 Depth=1
	fadd.s	fa4, fa4, fa0
	slli	t2, a6, 2
	add	t2, t2, a5
	fsw	fa4, 0(t2)
	addi	a6, a6, 1
	add	a2, a2, a7
	beq	a6, a3, .LBB1_6
.LBB1_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_5 Depth 2
	fmv.s	fa4, fa5
	blt	a4, t1, .LBB1_2
# %bb.4:                                #   in Loop: Header=BB1_3 Depth=1
	mv	t2, a2
	mv	t3, a0
	mv	t4, a1
	fmv.s	fa4, fa5
.LBB1_5:                                #   Parent Loop BB1_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	lbu	t5, 0(t2)
	flw	fa3, 0(t3)
	andi	t6, t5, 15
	fcvt.s.w	fa2, t6
	flw	fa1, 0(t4)
	fmadd.s	fa4, fa2, fa3, fa4
	srli	t5, t5, 4
	fcvt.s.w	fa3, t5
	fmadd.s	fa4, fa3, fa1, fa4
	addi	t4, t4, 4
	addi	t3, t3, 4
	addi	t2, t2, 1
	bne	t4, t0, .LBB1_5
	j	.LBB1_2
.LBB1_6:
	ret
.Lfunc_end1:
	.size	kv_qk_dot_kivi4_scalar, .Lfunc_end1-kv_qk_dot_kivi4_scalar
                                        # -- End function
	.globl	kv_qk_dot_kivi4_rvv             # -- Begin function kv_qk_dot_kivi4_rvv
	.p2align	1
	.type	kv_qk_dot_kivi4_rvv,@function
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
.Lfunc_end2:
	.size	kv_qk_dot_kivi4_rvv, .Lfunc_end2-kv_qk_dot_kivi4_rvv
                                        # -- End function
	.globl	kv_qk_dot_kivi4                 # -- Begin function kv_qk_dot_kivi4
	.p2align	1
	.type	kv_qk_dot_kivi4,@function
kv_qk_dot_kivi4:                        # @kv_qk_dot_kivi4
# %bb.0:
	vsetvli	a6, zero, e32, m1, ta, ma
	blez	a3, .LBB3_6
# %bb.1:
	li	a6, 0
	srliw	a7, a4, 31
	add	a7, a7, a4
	sraiw	a7, a7, 1
	vmv.v.i	v8, 0
	li	t0, 2
	vmv.s.x	v9, zero
	j	.LBB3_3
.LBB3_2:                                #   in Loop: Header=BB3_3 Depth=1
	vsetvli	t1, zero, e32, m1, ta, ma
	vfredusum.vs	v10, v10, v9
	vfmv.f.s	fa5, v10
	fadd.s	fa5, fa5, fa0
	slli	t1, a6, 2
	add	t1, t1, a5
	addi	a6, a6, 1
	fsw	fa5, 0(t1)
	beq	a6, a3, .LBB3_6
.LBB3_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_5 Depth 2
	vmv1r.v	v10, v8
	blt	a4, t0, .LBB3_2
# %bb.4:                                #   in Loop: Header=BB3_3 Depth=1
	li	t1, 0
	mul	t2, a6, a7
	add	t2, t2, a2
	vmv1r.v	v10, v8
.LBB3_5:                                #   Parent Loop BB3_3 Depth=1
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
	blt	t1, a7, .LBB3_5
	j	.LBB3_2
.LBB3_6:
	ret
.Lfunc_end3:
	.size	kv_qk_dot_kivi4, .Lfunc_end3-kv_qk_dot_kivi4
                                        # -- End function
	.ident	"Ubuntu clang version 18.1.3 (1ubuntu1)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
