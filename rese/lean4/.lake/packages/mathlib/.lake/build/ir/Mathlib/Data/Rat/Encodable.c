// Lean compiler output
// Module: Mathlib.Data.Rat.Encodable
// Imports: public import Init public import Mathlib.Logic.Encodable.Basic public import Mathlib.Data.Rat.Init
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
lean_object* lean_nat_gcd(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_encodable;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable___lam__1(lean_object*);
lean_object* lp_mathlib_Subtype_encodable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable___lam__3(lean_object*, lean_object*);
lean_object* lp_mathlib_Sigma_encodable___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable___lam__0(lean_object*);
lean_object* lp_mathlib_Encodable_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lean_nat_abs(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Rat_instEncodable___lam__2(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
extern lean_object* lp_mathlib_Int_encodable;
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Rat_instEncodable___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_nat_dec_lt(x_3, x_2);
if (x_4 == 0)
{
return x_4;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_5 = lean_nat_abs(x_1);
x_6 = lean_nat_gcd(x_5, x_2);
lean_dec(x_5);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_dec_eq(x_6, x_7);
lean_dec(x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable___lam__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_Rat_instEncodable___lam__2(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable___lam__3(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Rat_instEncodable___lam__2___boxed), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_mathlib_Subtype_encodable___redArg(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable___lam__0(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_inc(x_3);
lean_dec(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Rat_instEncodable___lam__1(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_inc(x_3);
lean_dec(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
}
static lean_object* _init_lp_mathlib_Rat_instEncodable() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Rat_instEncodable___lam__0), 1, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Rat_instEncodable___lam__1), 1, 0);
x_3 = lp_mathlib_Int_encodable;
x_4 = lp_mathlib_Nat_encodable;
x_5 = lean_alloc_closure((void*)(lp_mathlib_Rat_instEncodable___lam__3), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_mathlib_Sigma_encodable___redArg(x_3, x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_1);
lean_ctor_set(x_7, 1, x_2);
x_8 = lp_mathlib_Encodable_ofEquiv___redArg(x_6, x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Rat_Init(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Rat_Encodable(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Rat_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Rat_instEncodable = _init_lp_mathlib_Rat_instEncodable();
lean_mark_persistent(lp_mathlib_Rat_instEncodable);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
