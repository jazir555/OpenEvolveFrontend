// Lean compiler output
// Module: Mathlib.Algebra.Ring.Subsemiring.MulOpposite
// Imports: public import Init public import Mathlib.Algebra.Group.Submonoid.MulOpposite public import Mathlib.Algebra.Ring.Subsemiring.Basic public import Mathlib.Algebra.Ring.Opposite
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
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_addEquivOp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_opEquiv(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_ringEquivOpMop(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submonoid_equivOp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mopRingEquivOp___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Subsemiring_mopRingEquivOp___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_op___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mopRingEquivOp(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
static lean_object* lp_mathlib_Subsemiring_ringEquivOpMop___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_addEquivOp(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_opEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_op(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_ringEquivOpMop___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_unop(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_unop___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_opEquiv(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_op(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_op___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subsemiring_op(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_unop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_unop___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subsemiring_unop(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_opEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Subsemiring_op___boxed), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subsemiring_unop___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_opEquiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subsemiring_opEquiv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_addEquivOp___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_NonAssocSemiring_toMulZeroOneClass___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_Submonoid_equivOp(lean_box(0), x_4, x_2);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_addEquivOp(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subsemiring_addEquivOp___redArg(x_2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Subsemiring_ringEquivOpMop___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_MulOpposite_opEquiv(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_ringEquivOpMop___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Subsemiring_addEquivOp___redArg(x_1, x_2);
x_4 = lp_mathlib_Subsemiring_ringEquivOpMop___redArg___closed__0;
x_5 = lp_mathlib_Equiv_trans___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_ringEquivOpMop(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subsemiring_ringEquivOpMop___redArg(x_2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Subsemiring_mopRingEquivOp___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Subsemiring_ringEquivOpMop___redArg___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mopRingEquivOp___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Subsemiring_mopRingEquivOp___redArg___closed__0;
x_4 = lp_mathlib_Subsemiring_addEquivOp___redArg(x_1, x_2);
x_5 = lp_mathlib_Equiv_trans___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subsemiring_mopRingEquivOp(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subsemiring_mopRingEquivOp___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Opposite(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_MulOpposite(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_MulOpposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Subsemiring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Subsemiring_ringEquivOpMop___redArg___closed__0 = _init_lp_mathlib_Subsemiring_ringEquivOpMop___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Subsemiring_ringEquivOpMop___redArg___closed__0);
lp_mathlib_Subsemiring_mopRingEquivOp___redArg___closed__0 = _init_lp_mathlib_Subsemiring_mopRingEquivOp___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Subsemiring_mopRingEquivOp___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
