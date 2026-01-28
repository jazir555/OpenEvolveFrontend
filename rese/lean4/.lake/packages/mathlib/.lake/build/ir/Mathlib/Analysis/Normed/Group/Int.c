// Lean compiler output
// Module: Mathlib.Analysis.Normed.Group.Int
// Imports: public import Init public import Mathlib.Analysis.Normed.Group.Basic public import Mathlib.Topology.Instances.Int
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
extern lean_object* lp_mathlib_Int_instAddCommGroup;
extern lean_object* lp_mathlib_Real_instAddGroup;
LEAN_EXPORT lean_object* lp_mathlib_Int_instNormedAddCommGroup;
lean_object* lp_mathlib_abs___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Real_instIntCast;
extern lean_object* lp_mathlib_Real_instDistribLattice;
extern lean_object* lp_mathlib_Int_instMetricSpace;
LEAN_EXPORT lean_object* lp_mathlib_Int_instNormedAddCommGroup___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_instNormedAddCommGroup___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_Real_instDistribLattice;
x_3 = lp_mathlib_Real_instAddGroup;
x_4 = lp_mathlib_Real_instIntCast;
x_5 = lean_apply_1(x_4, x_1);
x_6 = lp_mathlib_abs___redArg(x_2, x_3, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_Int_instNormedAddCommGroup() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Int_instNormedAddCommGroup___lam__0), 1, 0);
x_2 = lp_mathlib_Int_instAddCommGroup;
x_3 = lp_mathlib_Int_instMetricSpace;
x_4 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_2);
lean_ctor_set(x_4, 2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Instances_Int(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Int(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Instances_Int(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Int_instNormedAddCommGroup = _init_lp_mathlib_Int_instNormedAddCommGroup();
lean_mark_persistent(lp_mathlib_Int_instNormedAddCommGroup);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
