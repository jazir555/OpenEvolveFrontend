// Lean compiler output
// Module: Mathlib.RepresentationTheory.Submodule
// Imports: public import Init public import Mathlib.Algebra.Module.Submodule.Invariant public import Mathlib.RepresentationTheory.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Representation_invtSubmodule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Representation_invtSubmodule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Sublattice_instInfSet___lam__0(lean_object*);
static lean_object* lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice___closed__0;
static lean_object* lp_mathlib_Representation_invtSubmodule___closed__0;
static lean_object* _init_lp_mathlib_Representation_invtSubmodule___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Sublattice_instInfSet___lam__0(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Representation_invtSubmodule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Representation_invtSubmodule___closed__0;
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Representation_invtSubmodule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Representation_invtSubmodule(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
static lean_object* _init_lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice___closed__0;
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Invariant(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_RepresentationTheory_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_RepresentationTheory_Submodule(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Invariant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_RepresentationTheory_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Representation_invtSubmodule___closed__0 = _init_lp_mathlib_Representation_invtSubmodule___closed__0();
lean_mark_persistent(lp_mathlib_Representation_invtSubmodule___closed__0);
lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice___closed__0 = _init_lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice___closed__0();
lean_mark_persistent(lp_mathlib_Representation_invtSubmodule_instBoundedOrderSubtypeSubmoduleMemSublattice___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
