// Lean compiler output
// Module: Mathlib.Data.Nat.Pairing
// Imports: public import Init public import Mathlib.Algebra.Notation.Prod public import Mathlib.Data.Nat.Sqrt public import Mathlib.Data.Set.Lattice.Image
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
static lean_object* lp_mathlib_Nat_pairEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Nat_unpair(lean_object*);
lean_object* lp_batteries_Nat_sqrt(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pair(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_unpair___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv___lam__0(lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lean_nat_mul(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv;
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pair___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Nat_pair(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = lean_nat_dec_lt(x_1, x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_nat_mul(x_1, x_1);
x_5 = lean_nat_add(x_4, x_1);
lean_dec(x_4);
x_6 = lean_nat_add(x_5, x_2);
lean_dec(x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_nat_mul(x_2, x_2);
x_8 = lean_nat_add(x_7, x_1);
lean_dec(x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_pair___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Nat_pair(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_unpair(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lp_batteries_Nat_sqrt(x_1);
x_3 = lean_nat_mul(x_2, x_2);
x_4 = lean_nat_sub(x_1, x_3);
lean_dec(x_3);
x_5 = lean_nat_dec_lt(x_4, x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_nat_sub(x_4, x_2);
lean_dec(x_4);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_2);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
else
{
lean_object* x_8; 
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_4);
lean_ctor_set(x_8, 1, x_2);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_unpair___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nat_unpair(x_1);
lean_dec(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_pairEquiv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_unpair___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lp_mathlib_Nat_pair(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Nat_pairEquiv___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Nat_pairEquiv___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Nat_pairEquiv() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Nat_pairEquiv___lam__0___boxed), 1, 0);
x_2 = lp_mathlib_Nat_pairEquiv___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Notation_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Sqrt(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Nat_Pairing(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Notation_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Sqrt(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Nat_pairEquiv___closed__0 = _init_lp_mathlib_Nat_pairEquiv___closed__0();
lean_mark_persistent(lp_mathlib_Nat_pairEquiv___closed__0);
lp_mathlib_Nat_pairEquiv = _init_lp_mathlib_Nat_pairEquiv();
lean_mark_persistent(lp_mathlib_Nat_pairEquiv);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
