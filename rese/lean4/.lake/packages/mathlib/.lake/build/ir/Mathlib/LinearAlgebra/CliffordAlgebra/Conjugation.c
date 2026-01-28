// Lean compiler output
// Module: Mathlib.LinearAlgebra.CliffordAlgebra.Conjugation
// Imports: public import Init public import Mathlib.LinearAlgebra.CliffordAlgebra.Grading public import Mathlib.Algebra.Module.Opposite
#include <lean/lean.h>
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wunused-label"
#elif defined(__GNUC__) && !defined(__CLANG__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-label"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif
#ifdef __cplusplus
extern "C" {
#endif
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involuteEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_opLinearEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_toLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involuteEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOp___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddGroupWithOne___redArg(lean_object*);
lean_object* lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseEquiv___redArg(lean_object*);
lean_object* lp_mathlib_AlgEquiv_ofAlgHom___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involuteEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involute___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_instRingCliffordAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involute___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CliffordAlgebra_00_u03b9___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverse___redArg(lean_object*);
lean_object* lp_mathlib_CliffordAlgebra_lift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Ring_toNonAssocRing___redArg(lean_object*);
lean_object* lp_mathlib_AlgHom_opComm___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_ofInvolutive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOpEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involute(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involute___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instAlgebraCliffordAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOpEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOpEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involute___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; uint8_t x_5; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_Ring_toAddCommGroup___redArg(x_1);
x_4 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_3);
lean_dec_ref(x_3);
x_5 = !lean_is_exclusive(x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_6 = lean_ctor_get(x_4, 1);
x_7 = lean_ctor_get(x_4, 0);
lean_dec(x_7);
lean_inc_ref(x_1);
x_8 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_9 = lean_ctor_get(x_8, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_9, 2);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lp_mathlib_CliffordAlgebra_00_u03b9___redArg(x_1);
x_12 = lean_apply_1(x_11, x_2);
x_13 = lean_apply_1(x_6, x_10);
x_14 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set_tag(x_4, 3);
lean_ctor_set(x_4, 1, x_12);
lean_ctor_set(x_4, 0, x_14);
return x_4;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_15 = lean_ctor_get(x_4, 1);
lean_inc(x_15);
lean_dec(x_4);
lean_inc_ref(x_1);
x_16 = lp_mathlib_Ring_toAddGroupWithOne___redArg(x_1);
x_17 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
x_18 = lean_ctor_get(x_17, 2);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lp_mathlib_CliffordAlgebra_00_u03b9___redArg(x_1);
x_20 = lean_apply_1(x_19, x_2);
x_21 = lean_apply_1(x_15, x_18);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_21);
x_23 = lean_alloc_ctor(3, 2, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_20);
return x_23;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involute___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
lean_inc_ref(x_1);
x_4 = lp_mathlib_instAlgebraCliffordAlgebra___redArg(x_1);
lean_inc_ref(x_1);
x_5 = lp_mathlib_CliffordAlgebra_lift___redArg(x_1, x_3, x_4);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CliffordAlgebra_involute___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_1);
x_8 = lean_apply_1(x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involute(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_involute___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involute___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_involute(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involuteEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_CliffordAlgebra_involute___redArg(x_1);
lean_inc(x_2);
x_3 = lp_mathlib_AlgEquiv_ofAlgHom___redArg(x_2, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involuteEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_involuteEquiv___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_involuteEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_involuteEquiv(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOp___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
lean_inc_ref(x_3);
x_4 = lp_mathlib_MulOpposite_instSemiring___redArg(x_3);
x_5 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_1);
x_6 = lp_mathlib_instAlgebraCliffordAlgebra___redArg(x_1);
lean_inc_ref(x_6);
x_7 = lp_mathlib_MulOpposite_instAlgebra___redArg(x_6);
x_8 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_8);
lean_dec_ref(x_3);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_6, 0);
lean_inc(x_10);
lean_dec_ref(x_6);
x_11 = lp_mathlib_MulOpposite_opLinearEquiv(lean_box(0), lean_box(0), x_5, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
lean_inc_ref(x_1);
x_13 = lp_mathlib_CliffordAlgebra_lift___redArg(x_1, x_4, x_7);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_CliffordAlgebra_00_u03b9___redArg(x_1);
x_16 = lp_mathlib_LinearMap_comp___redArg(x_12, x_15);
x_17 = lean_apply_1(x_14, x_16);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_reverseOp___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_reverseOp(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOpEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
lean_dec_ref(x_2);
lean_inc_ref(x_3);
x_4 = lp_mathlib_AlgHom_opComm___redArg(x_3, x_3);
lean_dec_ref(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_CliffordAlgebra_reverseOp___redArg(x_1);
lean_inc(x_6);
x_7 = lean_apply_1(x_5, x_6);
x_8 = lp_mathlib_AlgEquiv_ofAlgHom___redArg(x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOpEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_reverseOpEquiv___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseOpEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_reverseOpEquiv(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverse___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_1);
x_3 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_4 = lp_mathlib_Ring_toNonAssocRing___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_NonUnitalNonAssocRing_toNonUnitalNonAssocSemiring___redArg(x_5);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec_ref(x_6);
lean_inc_ref(x_1);
x_8 = lp_mathlib_instAlgebraCliffordAlgebra___redArg(x_1);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_MulOpposite_opLinearEquiv(lean_box(0), lean_box(0), x_2, x_7, x_9);
lean_dec(x_9);
lean_dec_ref(x_7);
x_11 = lp_mathlib_LinearEquiv_symm___redArg(x_10);
x_12 = lean_ctor_get(x_11, 0);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib_CliffordAlgebra_reverseOp___redArg(x_1);
x_14 = lp_mathlib_AlgHom_toLinearMap___redArg(x_13);
x_15 = lp_mathlib_LinearMap_comp___redArg(x_12, x_14);
return x_15;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverse(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_reverse___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverse___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_reverse(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_CliffordAlgebra_reverse___redArg(x_1);
x_3 = lp_mathlib_LinearEquiv_ofInvolutive___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_reverseEquiv___redArg(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_reverseEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CliffordAlgebra_reverseEquiv(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Grading(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Conjugation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Grading(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
