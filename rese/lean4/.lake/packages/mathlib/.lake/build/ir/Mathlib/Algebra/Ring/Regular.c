// Lean compiler output
// Module: Mathlib.Algebra.Ring.Regular
// Imports: public import Init public import Mathlib.Algebra.Group.Basic public import Mathlib.Algebra.GroupWithZero.Regular public import Mathlib.Algebra.Ring.Defs
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
lean_object* lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NoZeroDivisors_toCancelCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NoZeroDivisors_toCancelMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDomain_toCancelCommMonoidWithZero(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDomain_toCancelMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NoZeroDivisors_toCancelCommMonoidWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDomain_toCancelMonoidWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NoZeroDivisors_toCancelMonoidWithZero(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_IsDomain_toCancelCommMonoidWithZero___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_NoZeroDivisors_toCancelMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_2);
x_5 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NoZeroDivisors_toCancelMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_NoZeroDivisors_toCancelCommMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_2);
x_5 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_4);
x_6 = !lean_is_exclusive(x_5);
if (x_6 == 0)
{
return x_5;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_5, 0);
x_8 = lean_ctor_get(x_5, 1);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_NoZeroDivisors_toCancelCommMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_2);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_3);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDomain_toCancelMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDomain_toCancelMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Semiring_toMonoidWithZero___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDomain_toCancelCommMonoidWithZero(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_IsDomain_toCancelCommMonoidWithZero___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CommSemiring_toCommMonoidWithZero___redArg(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Regular(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Regular(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
