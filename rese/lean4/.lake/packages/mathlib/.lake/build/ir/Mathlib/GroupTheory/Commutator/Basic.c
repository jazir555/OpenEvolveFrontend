// Lean compiler output
// Module: Mathlib.GroupTheory.Commutator.Basic
// Imports: public import Init public import Mathlib.Algebra.Group.Commutator public import Mathlib.GroupTheory.Subgroup.Centralizer public import Mathlib.GroupTheory.QuotientGroup.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_closureCommutatorRepresentatives___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commutator___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_closureCommutatorRepresentatives(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commutator___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_commutator(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commutator___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commutator(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commutator___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commutator(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_commutator___lam__0), 2, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_commutator___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subgroup_commutator(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commutator(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_commutator___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_commutator(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_closureCommutatorRepresentatives(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_closureCommutatorRepresentatives___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_closureCommutatorRepresentatives(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Commutator(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Commutator_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Commutator(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subgroup_Centralizer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_QuotientGroup_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
