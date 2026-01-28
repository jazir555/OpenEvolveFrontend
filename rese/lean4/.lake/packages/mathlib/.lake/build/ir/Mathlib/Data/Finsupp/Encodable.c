// Lean compiler output
// Module: Mathlib.Data.Finsupp.Encodable
// Imports: public import Init public import Mathlib.Data.Finsupp.ToDFinsupp public import Mathlib.Data.DFinsupp.Encodable
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
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_finsuppEquivDFinsupp___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Encodable_ofEquiv___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Encodable_decidableEqOfEncodable___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__2(x_1, x_2, x_3);
lean_dec(x_2);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_6, 0, x_2);
lean_inc_ref(x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg___lam__2___boxed), 3, 1);
lean_closure_set(x_7, 0, x_4);
lean_inc_ref(x_1);
x_8 = lp_mathlib_instEncodableDFinsuppOfDecidableNeOfNat___redArg(x_5, x_1, x_6, x_7);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Encodable_decidableEqOfEncodable___boxed), 4, 2);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_1);
x_10 = lp_mathlib_finsuppEquivDFinsupp___redArg(x_9, x_3, x_4);
x_11 = lp_mathlib_Encodable_ofEquiv___redArg(x_8, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instEncodableFinsuppOfDecidableNeOfNat___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_DFinsupp_Encodable(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finsupp_Encodable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finsupp_ToDFinsupp(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_DFinsupp_Encodable(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
