// Lean compiler output
// Module: Mathlib.Algebra.Category.Ring.Topology
// Imports: public import Init public import Mathlib.Algebra.Category.Ring.Colimits public import Mathlib.Algebra.Category.Ring.Constructions public import Mathlib.Algebra.MvPolynomial.CommRing public import Mathlib.Topology.Algebra.Ring.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_instTopologicalSpaceHom___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_RingHom_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_instTopologicalSpaceHom(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_instTopologicalSpaceHom(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_instTopologicalSpaceHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CommRingCat_HomTopology_instTopologicalSpaceHom(x_1, x_2, x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_RingHom_comp___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_RingHom_comp___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommRingCat_HomTopology_precompHomeomorph___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CommRingCat_HomTopology_precompHomeomorph(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Ring_Colimits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Ring_Constructions(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_MvPolynomial_CommRing(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Ring_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_Ring_Topology(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Ring_Colimits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Ring_Constructions(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_MvPolynomial_CommRing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Ring_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
