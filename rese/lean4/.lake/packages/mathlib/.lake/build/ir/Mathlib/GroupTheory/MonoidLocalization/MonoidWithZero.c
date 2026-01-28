// Lean compiler output
// Module: Mathlib.GroupTheory.MonoidLocalization.MonoidWithZero
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Hom public import Mathlib.Algebra.GroupWithZero.NonZeroDivisors public import Mathlib.Algebra.GroupWithZero.Units.Basic public import Mathlib.GroupTheory.MonoidLocalization.Basic public import Mathlib.RingTheory.OreLocalization.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidWithZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidWithZeroHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instCommMonoidWithZero(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_instMonoid___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationWithZeroMap_toMonoidWithZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationWithZeroMap_toMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidWithZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationWithZeroMap_toMonoidWithZeroHom___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_oreSetComm(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Localization_instCommMonoidWithZero___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_OreLocalization_zero___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidWithZeroHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_apply_1(x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidWithZeroHom___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationMap_toMonoidWithZeroHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Submonoid_LocalizationMap_toMonoidWithZeroHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationWithZeroMap_toMonoidWithZeroHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_apply_1(x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationWithZeroMap_toMonoidWithZeroHom___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submonoid_LocalizationWithZeroMap_toMonoidWithZeroHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Submonoid_LocalizationWithZeroMap_toMonoidWithZeroHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_instCommMonoidWithZero___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_OreLocalization_oreSetComm(lean_box(0), x_3, x_2);
lean_inc_ref(x_3);
x_5 = lp_mathlib_OreLocalization_instMonoid___redArg(x_3, x_2, x_4);
x_6 = lp_mathlib_CommMonoidWithZero_toMonoidWithZero___redArg(x_1);
x_7 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_6);
x_8 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_7);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lean_ctor_get(x_8, 1);
x_11 = lean_ctor_get(x_8, 0);
lean_dec(x_11);
x_12 = lp_mathlib_OreLocalization_zero___redArg(x_3, x_10);
lean_dec_ref(x_3);
lean_ctor_set(x_8, 1, x_12);
lean_ctor_set(x_8, 0, x_5);
return x_8;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_8, 1);
lean_inc(x_13);
lean_dec(x_8);
x_14 = lp_mathlib_OreLocalization_zero___redArg(x_3, x_13);
lean_dec_ref(x_3);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_5);
lean_ctor_set(x_15, 1, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Localization_instCommMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Localization_instCommMonoidWithZero___redArg(x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_MonoidWithZero(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_NonZeroDivisors(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_MonoidLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RingTheory_OreLocalization_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
