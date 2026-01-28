// Lean compiler output
// Module: Mathlib.Data.Sym.Sym2.Order
// Imports: public import Init public import Mathlib.Data.Sym.Sym2 public import Mathlib.Order.Lattice
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
lean_object* lp_mathlib_Sym2_lift(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_inf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sup___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_inf(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_inf___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sup(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Sym2_sup___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv(lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg___lam__0(lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
static lean_object* _init_lp_mathlib_Sym2_sup___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Sym2_lift(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sup___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Sym2_sup___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Sym2_sup___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_apply_2(x_4, x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sup(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Sym2_sup___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_inf___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_inf___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Sym2_sup___redArg___closed__0;
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Sym2_inf___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_apply_2(x_4, x_5, x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_inf(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Sym2_inf___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Sym2_sortEquiv___redArg___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_4 = lp_mathlib_Sym2_inf___redArg(x_1, x_3);
x_5 = lp_mathlib_Sym2_sup___redArg(x_2, x_3);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
lean_inc_ref(x_2);
x_3 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_2);
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
lean_dec(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Sym2_sortEquiv___redArg___lam__0___boxed), 1, 0);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Sym2_sortEquiv___redArg___lam__1), 3, 2);
lean_closure_set(x_8, 0, x_3);
lean_closure_set(x_8, 1, x_5);
lean_ctor_set(x_2, 1, x_7);
lean_ctor_set(x_2, 0, x_8);
return x_2;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_ctor_get(x_2, 0);
lean_inc(x_9);
lean_dec(x_2);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Sym2_sortEquiv___redArg___lam__0___boxed), 1, 0);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Sym2_sortEquiv___redArg___lam__1), 3, 2);
lean_closure_set(x_11, 0, x_3);
lean_closure_set(x_11, 1, x_9);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_10);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Sym2_sortEquiv___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Sym2_sortEquiv(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sym2_sortEquiv___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Sym2_sortEquiv___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sym_Sym2(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Lattice(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Sym_Sym2_Order(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sym_Sym2(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Sym2_sup___redArg___closed__0 = _init_lp_mathlib_Sym2_sup___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Sym2_sup___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
