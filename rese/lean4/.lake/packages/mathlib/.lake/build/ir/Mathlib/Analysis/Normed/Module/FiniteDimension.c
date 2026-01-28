// Lean compiler output
// Module: Mathlib.Analysis.Normed.Module.FiniteDimension
// Imports: public import Init public import Mathlib.Analysis.Asymptotics.AsymptoticEquivalent public import Mathlib.Analysis.Normed.Group.Lemmas public import Mathlib.Analysis.Normed.Affine.Isometry public import Mathlib.Analysis.Normed.Operator.NormedSpace public import Mathlib.Analysis.Normed.Module.RieszLemma public import Mathlib.Analysis.Normed.Module.Ball.Pointwise public import Mathlib.Analysis.SpecificLimits.Normed public import Mathlib.Logic.Encodable.Pi public import Mathlib.Topology.Algebra.AffineSubspace public import Mathlib.Topology.Algebra.Module.FiniteDimension public import Mathlib.Topology.Algebra.InfiniteSum.Module public import Mathlib.Topology.Instances.Matrix public import Mathlib.LinearAlgebra.Dimension.LinearMap
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
lean_object* lp_mathlib_Semifield_toDivisionSemiring___redArg(lean_object*);
lean_object* lp_mathlib_NormedCommRing_toSeminormedCommRing___redArg(lean_object*);
lean_object* lp_mathlib_NormedAddCommGroup_toENormedAddCommMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Pi_topologicalSpace(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_Function_module___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__1___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearEquiv_piRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NormedField_toNormedCommRing___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Semiring_toModule___redArg(lean_object*);
lean_object* lp_mathlib_LinearEquiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional___redArg___boxed(lean_object*);
lean_object* lp_mathlib_NormedCommRing_toNonUnitalNormedCommRing___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__0___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_NormedAddCommGroup_toSeminormedAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Field_toSemifield___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_toContinuousLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_addCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; 
x_18 = lean_ctor_get(x_17, 0);
lean_inc_ref(x_18);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec_ref(x_17);
lean_dec_ref(x_15);
lean_dec_ref(x_14);
lean_dec_ref(x_13);
lean_dec_ref(x_12);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AffineEquiv_toHomeomorphOfFiniteDimensional___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__1(x_1, x_2);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; 
x_6 = lean_ctor_get(x_1, 1);
x_7 = lp_mathlib_Field_toSemifield___redArg(x_6);
x_8 = lp_mathlib_Semifield_toDivisionSemiring___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
lean_inc_ref(x_1);
x_10 = lp_mathlib_NormedField_toNormedCommRing___redArg(x_1);
lean_inc_ref(x_10);
x_11 = lp_mathlib_NormedCommRing_toSeminormedCommRing___redArg(x_10);
x_12 = lean_ctor_get(x_11, 2);
lean_inc_ref(x_12);
lean_dec_ref(x_11);
x_13 = lean_ctor_get(x_12, 2);
lean_inc_ref(x_13);
lean_dec_ref(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lp_mathlib_NormedCommRing_toNonUnitalNormedCommRing___redArg(x_10);
x_16 = lp_mathlib_NonUnitalNormedRing_toNormedAddCommGroup___redArg(x_15);
x_17 = lean_ctor_get(x_16, 1);
lean_inc_ref(x_17);
lean_dec_ref(x_16);
lean_inc_ref(x_2);
x_18 = lp_mathlib_NormedAddCommGroup_toSeminormedAddCommGroup___redArg(x_2);
x_19 = lean_ctor_get(x_18, 2);
lean_inc_ref(x_19);
x_20 = lean_ctor_get(x_19, 2);
lean_inc_ref(x_20);
lean_dec_ref(x_19);
x_21 = lean_ctor_get(x_18, 1);
lean_inc_ref(x_21);
lean_dec_ref(x_18);
x_22 = lean_ctor_get(x_20, 0);
lean_inc(x_22);
lean_dec_ref(x_20);
x_23 = lp_mathlib_NormedAddCommGroup_toENormedAddCommMonoid___redArg(x_2);
x_24 = lean_ctor_get(x_23, 1);
lean_inc_ref(x_24);
lean_dec_ref(x_23);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_25, 0, x_14);
x_26 = lean_alloc_closure((void*)(lp_mathlib_ContinuousLinearEquiv_piRing___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_26, 0, x_17);
lean_inc_ref(x_9);
x_27 = lp_mathlib_Semiring_toModule___redArg(x_9);
x_28 = lp_mathlib_Pi_Function_module___redArg(x_27);
x_29 = lp_mathlib_Pi_topologicalSpace(lean_box(0), lean_box(0), x_25);
lean_dec_ref(x_25);
x_30 = lp_mathlib_Pi_addCommGroup___redArg(x_26);
x_31 = lp_mathlib_LinearMap_toContinuousLinearMap(lean_box(0), x_1, lean_box(0), x_30, x_28, x_29, lean_box(0), lean_box(0), lean_box(0), x_21, x_3, x_22, lean_box(0), lean_box(0), lean_box(0), lean_box(0), lean_box(0));
lean_dec_ref(x_21);
lean_dec(x_28);
lean_dec_ref(x_30);
lean_dec_ref(x_1);
x_32 = lp_mathlib_LinearEquiv_symm___redArg(x_31);
x_33 = lp_mathlib_LinearEquiv_piRing___redArg(x_9, x_4, x_5, x_24, x_3);
x_34 = lp_mathlib_LinearEquiv_trans___redArg(x_32, x_33);
return x_34;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ContinuousLinearEquiv_piRing(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_ContinuousLinearEquiv_piRing___redArg(x_2, x_4, x_5, x_8, x_9);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Asymptotics_AsymptoticEquivalent(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Affine_Isometry(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Operator_NormedSpace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_RieszLemma(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_Ball_Pointwise(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_SpecificLimits_Normed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Encodable_Pi(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_AffineSubspace(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_Module_FiniteDimension(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Algebra_InfiniteSum_Module(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Topology_Instances_Matrix(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Dimension_LinearMap(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Module_FiniteDimension(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Asymptotics_AsymptoticEquivalent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Affine_Isometry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Operator_NormedSpace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Module_RieszLemma(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Module_Ball_Pointwise(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_SpecificLimits_Normed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Encodable_Pi(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_AffineSubspace(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_Module_FiniteDimension(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Algebra_InfiniteSum_Module(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Topology_Instances_Matrix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Dimension_LinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
