// Lean compiler output
// Module: Mathlib.RingTheory.Localization.FractionRing
// Imports: public import Init public import Mathlib.Algebra.Ring.Hom.InjSurj public import Mathlib.Algebra.Field.Equiv public import Mathlib.Algebra.Field.Subfield.Basic public import Mathlib.Algebra.Order.GroupWithZero.Submonoid public import Mathlib.Algebra.Order.Ring.Int public import Mathlib.RingTheory.Localization.Basic public import Mathlib.RingTheory.SimpleRing.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Localization_instUniqueLocalization___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lp_mathlib_Localization_instUniqueLocalization___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FractionRing_unique___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_FractionRing_unique(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_FractionRing_unique___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_FractionRing_unique___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Hom_InjSurj(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Equiv(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Submonoid(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RingTheory_Localization_FractionRing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Hom_InjSurj(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Field_Subfield_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_GroupWithZero_Submonoid(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Order_Ring_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_Localization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_SimpleRing_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
