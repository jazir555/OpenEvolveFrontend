// Lean compiler output
// Module: Mathlib.Algebra.NonAssoc.PreLie.Basic
// Imports: public import Init public import Mathlib.Algebra.Module.Opposite public import Mathlib.Algebra.Ring.Associator public import Mathlib.GroupTheory.GroupAction.Ring
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
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instRightPreLieAlgebraMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLeftPreLieAlgebraMulOpposite___redArg(lean_object*);
lean_object* lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLeftPreLieAlgebraMulOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLeftPreLieRingMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLeftPreLieAlgebraMulOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLeftPreLieRingMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instRightPreLieRingMulOpposite___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instRightPreLieRingMulOpposite(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instRightPreLieAlgebraMulOpposite(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instRightPreLieAlgebraMulOpposite___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulOpposite_instSMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instRightPreLieRingMulOpposite(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieRing_instRightPreLieRingMulOpposite___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instRightPreLieAlgebraMulOpposite(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MulOpposite_instSMul___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instRightPreLieAlgebraMulOpposite___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulOpposite_instSMul___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LeftPreLieAlgebra_instRightPreLieAlgebraMulOpposite___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_LeftPreLieAlgebra_instRightPreLieAlgebraMulOpposite(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLeftPreLieRingMulOpposite(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieRing_instLeftPreLieRingMulOpposite___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulOpposite_instNonUnitalNonAssocRing___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLeftPreLieAlgebraMulOpposite(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MulOpposite_instSMul___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLeftPreLieAlgebraMulOpposite___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulOpposite_instSMul___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_RightPreLieAlgebra_instLeftPreLieAlgebraMulOpposite___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_RightPreLieAlgebra_instLeftPreLieAlgebraMulOpposite(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Opposite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Associator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_GroupAction_Ring(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_NonAssoc_PreLie_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Opposite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Associator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_GroupAction_Ring(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
