// Lean compiler output
// Module: Mathlib.Order.SupClosed
// Imports: public import Init public import Mathlib.Data.Finset.Lattice.Prod public import Mathlib.Data.Finset.Powerset public import Mathlib.Data.Set.Finite.Basic public import Mathlib.Order.Closure public import Mathlib.Order.ConditionallyCompleteLattice.Finset
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
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_latticeClosure___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infClosure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_supClosure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_infClosure___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_supClosure___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_latticeClosure(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_supClosure(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_supClosure___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_supClosure(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_infClosure(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_infClosure___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_infClosure(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_latticeClosure(lean_object* x_1, lean_object* x_2) {
_start:
{
return lean_box(0);
}
}
LEAN_EXPORT lean_object* lp_mathlib_latticeClosure___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_latticeClosure(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_dec(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_2);
lean_ctor_set(x_1, 1, x_5);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_2);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SemilatticeSup_toCompleteSemilatticeSup___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_dec(x_4);
x_5 = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_2);
lean_ctor_set(x_1, 1, x_5);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_2);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SemilatticeInf_toCompleteSemilatticeInf___redArg(x_2, x_3);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Powerset(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Finite_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Closure(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_SupClosed(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Powerset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Finite_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Closure(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Finset(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
