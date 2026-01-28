// Lean compiler output
// Module: Mathlib.Algebra.Group.Int.Even
// Imports: public import Init public import Mathlib.Algebra.Group.Int.Defs public import Mathlib.Algebra.Group.Nat.Even public import Mathlib.Data.Int.Sqrt
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
lean_object* lean_int_emod(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredEven___boxed(lean_object*);
static lean_object* lp_mathlib_Int_instDecidablePredEven___closed__1;
static lean_object* lp_mathlib_Int_instDecidablePredEven___closed__0;
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredEven(lean_object*);
lean_object* lean_nat_to_int(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredIsSquare___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredIsSquare(lean_object*);
lean_object* lean_int_mul(lean_object*, lean_object*);
lean_object* lp_mathlib_Int_sqrt(lean_object*);
uint8_t lean_int_dec_eq(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Int_instDecidablePredEven___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(2u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Int_instDecidablePredEven___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredEven(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_mathlib_Int_instDecidablePredEven___closed__0;
x_3 = lean_int_emod(x_1, x_2);
x_4 = lp_mathlib_Int_instDecidablePredEven___closed__1;
x_5 = lean_int_dec_eq(x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredEven___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Int_instDecidablePredEven(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Int_instDecidablePredIsSquare(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lp_mathlib_Int_sqrt(x_1);
x_3 = lean_int_mul(x_2, x_2);
lean_dec(x_2);
x_4 = lean_int_dec_eq(x_3, x_1);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_instDecidablePredIsSquare___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Int_instDecidablePredIsSquare(x_1);
lean_dec(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Even(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Int_Sqrt(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Int_Even(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Int_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Even(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Int_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instDecidablePredEven___closed__0 = _init_lp_mathlib_Int_instDecidablePredEven___closed__0();
lean_mark_persistent(lp_mathlib_Int_instDecidablePredEven___closed__0);
lp_mathlib_Int_instDecidablePredEven___closed__1 = _init_lp_mathlib_Int_instDecidablePredEven___closed__1();
lean_mark_persistent(lp_mathlib_Int_instDecidablePredEven___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
