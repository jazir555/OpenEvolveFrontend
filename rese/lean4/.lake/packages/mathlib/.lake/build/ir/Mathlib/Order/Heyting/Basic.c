// Lean compiler output
// Module: Mathlib.Order.Heyting.Basic
// Imports: public import Init public import Mathlib.Order.PropInstances public import Mathlib.Order.GaloisConnection.Defs
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
lean_object* lp_mathlib_Pi_instOrderTop___redArg(lean_object*);
static lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedHeytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNotForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNotForall___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBiheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHeytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHasCompl___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instHeytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prop_instHeytingAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHeytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_sdiff___redArg(lean_object*);
static lean_object* lp_mathlib_Prop_instHeytingAlgebra___closed__1;
static lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
lean_object* lp_mathlib_Pi_instOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHasCompl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toDistribLattice___redArg___boxed(lean_object*);
extern lean_object* lp_mathlib_Prop_instDistribLattice;
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_hasCompl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImpForall___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedCoheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBiheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBiheytingAlgebra___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instHeytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNotForall(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImpForall(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Prop_instHeytingAlgebra___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedCoheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_OrderDual_instHeytingAlgebra___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toDistribLattice___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_PUnit_instLinearOrder;
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toDistribLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImpForall___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBiheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(lean_object*);
lean_object* lp_mathlib_LinearOrder_toLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = !lean_is_exclusive(x_4);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_4, 0);
x_9 = lean_ctor_get(x_4, 1);
x_10 = lean_apply_2(x_1, x_5, x_8);
x_11 = lean_apply_2(x_2, x_6, x_9);
lean_ctor_set(x_4, 1, x_11);
lean_ctor_set(x_4, 0, x_10);
return x_4;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_ctor_get(x_4, 0);
x_13 = lean_ctor_get(x_4, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_4);
x_14 = lean_apply_2(x_1, x_5, x_12);
x_15 = lean_apply_2(x_2, x_6, x_13);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_14);
lean_ctor_set(x_16, 1, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHImp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instHImp___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
x_7 = lean_apply_1(x_1, x_5);
x_8 = lean_apply_1(x_2, x_6);
lean_ctor_set(x_3, 1, x_8);
lean_ctor_set(x_3, 0, x_7);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_3, 0);
x_10 = lean_ctor_get(x_3, 1);
lean_inc(x_10);
lean_inc(x_9);
lean_dec(x_3);
x_11 = lean_apply_1(x_1, x_9);
x_12 = lean_apply_1(x_2, x_10);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHNot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instHNot___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instSDiff(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instSDiff___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHasCompl___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHasCompl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instHasCompl___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImpForall___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_5 = lean_apply_1(x_2, x_4);
lean_inc(x_4);
x_6 = lean_apply_1(x_3, x_4);
x_7 = lean_apply_3(x_1, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImpForall___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_instHImpForall___redArg___lam__0), 4, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHImpForall(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instHImpForall___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNotForall___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
lean_inc(x_3);
x_4 = lean_apply_1(x_2, x_3);
x_5 = lean_apply_2(x_1, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNotForall___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_instHNotForall___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHNotForall(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instHNotForall___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_inc(x_4);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_HeytingAlgebra_toBoundedOrder(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_inc(x_3);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CoheytingAlgebra_toBoundedOrder(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_2, 2);
lean_dec(x_8);
x_9 = lean_ctor_get(x_2, 0);
lean_dec(x_9);
x_10 = !lean_is_exclusive(x_3);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_ctor_get(x_3, 1);
x_12 = lean_ctor_get(x_3, 2);
lean_dec(x_12);
lean_ctor_set(x_3, 2, x_4);
lean_ctor_set(x_3, 1, x_7);
lean_ctor_set(x_2, 2, x_5);
lean_ctor_set(x_2, 1, x_11);
return x_2;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_3, 0);
x_14 = lean_ctor_get(x_3, 1);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_3);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_13);
lean_ctor_set(x_15, 1, x_7);
lean_ctor_set(x_15, 2, x_4);
lean_ctor_set(x_2, 2, x_5);
lean_ctor_set(x_2, 1, x_14);
lean_ctor_set(x_2, 0, x_15);
return x_2;
}
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_16 = lean_ctor_get(x_2, 1);
lean_inc(x_16);
lean_dec(x_2);
x_17 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_17);
x_18 = lean_ctor_get(x_3, 1);
lean_inc(x_18);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 x_19 = x_3;
} else {
 lean_dec_ref(x_3);
 x_19 = lean_box(0);
}
if (lean_is_scalar(x_19)) {
 x_20 = lean_alloc_ctor(0, 3, 0);
} else {
 x_20 = x_19;
}
lean_ctor_set(x_20, 0, x_17);
lean_ctor_set(x_20, 1, x_16);
lean_ctor_set(x_20, 2, x_4);
x_21 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_18);
lean_ctor_set(x_21, 2, x_5);
return x_21;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 1);
lean_inc(x_7);
lean_dec_ref(x_3);
lean_inc(x_7);
lean_inc(x_4);
x_8 = lean_alloc_closure((void*)(lp_mathlib_HeytingAlgebra_ofHImp___redArg___lam__0), 3, 2);
lean_closure_set(x_8, 0, x_4);
lean_closure_set(x_8, 1, x_7);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_6);
lean_ctor_set(x_9, 2, x_4);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_7);
lean_ctor_set(x_10, 2, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofHImp___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec_ref(x_2);
lean_inc(x_5);
lean_inc(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_HeytingAlgebra_ofHImp___redArg___lam__0), 3, 2);
lean_closure_set(x_6, 0, x_3);
lean_closure_set(x_6, 1, x_5);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_1);
lean_ctor_set(x_7, 1, x_4);
lean_ctor_set(x_7, 2, x_3);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_5);
lean_ctor_set(x_8, 2, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_2, x_3);
x_7 = lean_apply_2(x_5, x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get(x_3, 1);
x_8 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
lean_inc_ref(x_8);
x_9 = lean_alloc_closure((void*)(lp_mathlib_HeytingAlgebra_ofCompl___redArg___lam__0), 4, 2);
lean_closure_set(x_9, 0, x_8);
lean_closure_set(x_9, 1, x_4);
lean_inc(x_6);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_2);
lean_ctor_set(x_10, 1, x_6);
lean_ctor_set(x_10, 2, x_9);
lean_inc(x_7);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_7);
lean_ctor_set(x_11, 2, x_4);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_inc_ref(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_HeytingAlgebra_ofCompl___redArg___lam__0), 4, 2);
lean_closure_set(x_7, 0, x_6);
lean_closure_set(x_7, 1, x_3);
lean_inc(x_4);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_1);
lean_ctor_set(x_8, 1, x_4);
lean_ctor_set(x_8, 2, x_7);
lean_inc(x_5);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_5);
lean_ctor_set(x_9, 2, x_3);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_HeytingAlgebra_ofCompl(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_HeytingAlgebra_ofCompl___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_HeytingAlgebra_ofCompl___redArg(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 1);
lean_inc(x_7);
lean_dec_ref(x_3);
lean_inc(x_6);
lean_inc(x_4);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CoheytingAlgebra_ofSDiff___redArg___lam__0), 3, 2);
lean_closure_set(x_8, 0, x_4);
lean_closure_set(x_8, 1, x_6);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_7);
lean_ctor_set(x_9, 2, x_4);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_6);
lean_ctor_set(x_10, 2, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofSDiff___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec_ref(x_2);
lean_inc(x_4);
lean_inc(x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CoheytingAlgebra_ofSDiff___redArg___lam__0), 3, 2);
lean_closure_set(x_6, 0, x_3);
lean_closure_set(x_6, 1, x_4);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_1);
lean_ctor_set(x_7, 1, x_5);
lean_ctor_set(x_7, 2, x_3);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_4);
lean_ctor_set(x_8, 2, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_2(x_5, x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_3, 0);
x_7 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_inc_ref(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_CoheytingAlgebra_ofHNot___redArg___lam__0), 4, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_4);
lean_inc(x_7);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_2);
lean_ctor_set(x_9, 1, x_7);
lean_ctor_set(x_9, 2, x_8);
lean_inc(x_6);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_6);
lean_ctor_set(x_10, 2, x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CoheytingAlgebra_ofHNot___redArg___lam__0), 4, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_3);
lean_inc(x_5);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_1);
lean_ctor_set(x_7, 1, x_5);
lean_ctor_set(x_7, 2, x_6);
lean_inc(x_4);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_4);
lean_ctor_set(x_8, 2, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CoheytingAlgebra_ofHNot(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_ofHNot___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CoheytingAlgebra_ofHNot___redArg(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_GeneralizedHeytingAlgebra_toDistribLattice___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, x_3);
lean_inc(x_5);
x_7 = lean_apply_1(x_5, x_2);
x_8 = lean_apply_2(x_1, x_6, x_7);
x_9 = lean_apply_1(x_5, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_mathlib_OrderDual_instLattice___redArg(x_3);
lean_ctor_set(x_1, 2, x_5);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_9);
x_11 = lp_mathlib_OrderDual_instLattice___redArg(x_7);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_8);
lean_ctor_set(x_12, 2, x_10);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_ctor_get(x_2, 2);
x_10 = lp_mathlib_Prod_instLattice___redArg(x_3, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_8);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_12, 0, x_5);
lean_closure_set(x_12, 1, x_9);
lean_ctor_set(x_2, 2, x_12);
lean_ctor_set(x_2, 1, x_11);
lean_ctor_set(x_2, 0, x_10);
return x_2;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_ctor_get(x_2, 0);
x_14 = lean_ctor_get(x_2, 1);
x_15 = lean_ctor_get(x_2, 2);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_2);
x_16 = lp_mathlib_Prod_instLattice___redArg(x_3, x_13);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_4);
lean_ctor_set(x_17, 1, x_14);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_18, 0, x_5);
lean_closure_set(x_18, 1, x_15);
x_19 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_19, 0, x_16);
lean_ctor_set(x_19, 1, x_17);
lean_ctor_set(x_19, 2, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedHeytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 2);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg___lam__2), 4, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lp_mathlib_Pi_instLattice___redArg(x_2);
x_6 = lp_mathlib_Pi_instOrderTop___redArg(x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Pi_instHImpForall___redArg___lam__0), 4, 1);
lean_closure_set(x_7, 0, x_4);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedHeytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_GeneralizedCoheytingAlgebra_toDistribLattice___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0;
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, x_3);
lean_inc(x_5);
x_7 = lean_apply_1(x_5, x_2);
x_8 = lean_apply_2(x_1, x_6, x_7);
x_9 = lean_apply_1(x_5, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 2);
x_5 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_mathlib_OrderDual_instLattice___redArg(x_3);
lean_ctor_set(x_1, 2, x_5);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_9);
x_11 = lp_mathlib_OrderDual_instLattice___redArg(x_7);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_8);
lean_ctor_set(x_12, 2, x_10);
return x_12;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedCoheytingAlgebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_ctor_get(x_2, 2);
x_10 = lp_mathlib_Prod_instLattice___redArg(x_3, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_8);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_12, 0, x_5);
lean_closure_set(x_12, 1, x_9);
lean_ctor_set(x_2, 2, x_12);
lean_ctor_set(x_2, 1, x_11);
lean_ctor_set(x_2, 0, x_10);
return x_2;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_ctor_get(x_2, 0);
x_14 = lean_ctor_get(x_2, 1);
x_15 = lean_ctor_get(x_2, 2);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_2);
x_16 = lp_mathlib_Prod_instLattice___redArg(x_3, x_13);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_4);
lean_ctor_set(x_17, 1, x_14);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_18, 0, x_5);
lean_closure_set(x_18, 1, x_15);
x_19 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_19, 0, x_16);
lean_ctor_set(x_19, 1, x_17);
lean_ctor_set(x_19, 2, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instGeneralizedCoheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instGeneralizedCoheytingAlgebra___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_apply_1(x_1, x_2);
x_6 = lean_ctor_get(x_5, 2);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_2(x_6, x_3, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__2), 4, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lp_mathlib_Pi_instLattice___redArg(x_2);
x_6 = lp_mathlib_Pi_instOrderBot___redArg(x_3);
x_7 = lp_mathlib_Pi_sdiff___redArg(x_4);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instGeneralizedCoheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 2);
x_8 = lean_ctor_get(x_2, 1);
lean_dec(x_8);
x_9 = lp_mathlib_OrderDual_instLattice___redArg(x_6);
x_10 = lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(x_1);
x_11 = !lean_is_exclusive(x_1);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_12 = lean_ctor_get(x_1, 2);
lean_dec(x_12);
x_13 = lean_ctor_get(x_1, 1);
lean_dec(x_13);
x_14 = lean_ctor_get(x_1, 0);
lean_dec(x_14);
x_15 = lean_ctor_get(x_10, 0);
lean_inc(x_15);
lean_dec_ref(x_10);
x_16 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_16, 0, x_7);
lean_ctor_set(x_2, 2, x_16);
lean_ctor_set(x_2, 1, x_15);
lean_ctor_set(x_2, 0, x_9);
x_17 = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0;
x_18 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_18, 0, x_17);
lean_inc_ref(x_18);
x_19 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, lean_box(0));
lean_closure_set(x_19, 2, lean_box(0));
lean_closure_set(x_19, 3, x_4);
lean_closure_set(x_19, 4, x_18);
x_20 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_20, 0, lean_box(0));
lean_closure_set(x_20, 1, lean_box(0));
lean_closure_set(x_20, 2, lean_box(0));
lean_closure_set(x_20, 3, x_18);
lean_closure_set(x_20, 4, x_19);
lean_ctor_set(x_1, 2, x_20);
return x_1;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; 
lean_dec(x_1);
x_21 = lean_ctor_get(x_10, 0);
lean_inc(x_21);
lean_dec_ref(x_10);
x_22 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_22, 0, x_7);
lean_ctor_set(x_2, 2, x_22);
lean_ctor_set(x_2, 1, x_21);
lean_ctor_set(x_2, 0, x_9);
x_23 = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0;
x_24 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_24, 0, x_23);
lean_inc_ref(x_24);
x_25 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_25, 0, lean_box(0));
lean_closure_set(x_25, 1, lean_box(0));
lean_closure_set(x_25, 2, lean_box(0));
lean_closure_set(x_25, 3, x_4);
lean_closure_set(x_25, 4, x_24);
x_26 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_26, 0, lean_box(0));
lean_closure_set(x_26, 1, lean_box(0));
lean_closure_set(x_26, 2, lean_box(0));
lean_closure_set(x_26, 3, x_24);
lean_closure_set(x_26, 4, x_25);
x_27 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_27, 0, x_2);
lean_ctor_set(x_27, 1, x_3);
lean_ctor_set(x_27, 2, x_26);
return x_27;
}
}
else
{
lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_28 = lean_ctor_get(x_2, 0);
x_29 = lean_ctor_get(x_2, 2);
lean_inc(x_29);
lean_inc(x_28);
lean_dec(x_2);
x_30 = lp_mathlib_OrderDual_instLattice___redArg(x_28);
x_31 = lp_mathlib_HeytingAlgebra_toBoundedOrder___redArg(x_1);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 lean_ctor_release(x_1, 1);
 lean_ctor_release(x_1, 2);
 x_32 = x_1;
} else {
 lean_dec_ref(x_1);
 x_32 = lean_box(0);
}
x_33 = lean_ctor_get(x_31, 0);
lean_inc(x_33);
lean_dec_ref(x_31);
x_34 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_34, 0, x_29);
x_35 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_35, 0, x_30);
lean_ctor_set(x_35, 1, x_33);
lean_ctor_set(x_35, 2, x_34);
x_36 = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0;
x_37 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_37, 0, x_36);
lean_inc_ref(x_37);
x_38 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_38, 0, lean_box(0));
lean_closure_set(x_38, 1, lean_box(0));
lean_closure_set(x_38, 2, lean_box(0));
lean_closure_set(x_38, 3, x_4);
lean_closure_set(x_38, 4, x_37);
x_39 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_39, 0, lean_box(0));
lean_closure_set(x_39, 1, lean_box(0));
lean_closure_set(x_39, 2, lean_box(0));
lean_closure_set(x_39, 3, x_37);
lean_closure_set(x_39, 4, x_38);
if (lean_is_scalar(x_32)) {
 x_40 = lean_alloc_ctor(0, 3, 0);
} else {
 x_40 = x_32;
}
lean_ctor_set(x_40, 0, x_35);
lean_ctor_set(x_40, 1, x_3);
lean_ctor_set(x_40, 2, x_39);
return x_40;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instCoheytingAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHeytingAlgebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 2);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = !lean_is_exclusive(x_2);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_ctor_get(x_2, 2);
x_10 = lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(x_3, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_8);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_12, 0, x_5);
lean_closure_set(x_12, 1, x_9);
lean_ctor_set(x_2, 2, x_12);
lean_ctor_set(x_2, 1, x_11);
lean_ctor_set(x_2, 0, x_10);
return x_2;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_13 = lean_ctor_get(x_2, 0);
x_14 = lean_ctor_get(x_2, 1);
x_15 = lean_ctor_get(x_2, 2);
lean_inc(x_15);
lean_inc(x_14);
lean_inc(x_13);
lean_dec(x_2);
x_16 = lp_mathlib_Prod_instGeneralizedHeytingAlgebra___redArg(x_3, x_13);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_4);
lean_ctor_set(x_17, 1, x_14);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_18, 0, x_5);
lean_closure_set(x_18, 1, x_15);
x_19 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_19, 0, x_16);
lean_ctor_set(x_19, 1, x_17);
lean_ctor_set(x_19, 2, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instHeytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instHeytingAlgebra___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_1(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Pi_instHeytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lp_mathlib_Pi_instGeneralizedHeytingAlgebra___redArg(x_2);
x_6 = lp_mathlib_Pi_instOrderBot___redArg(x_3);
x_7 = lp_mathlib_Pi_hasCompl___redArg(x_4);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instHeytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instHeytingAlgebra___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toDistribLattice___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toDistribLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CoheytingAlgebra_toDistribLattice___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toDistribLattice___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CoheytingAlgebra_toDistribLattice(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CoheytingAlgebra_toDistribLattice___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_CoheytingAlgebra_toDistribLattice___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_OrderDual_instHeytingAlgebra___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0;
x_2 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instCoheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instHeytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 2);
x_8 = lean_ctor_get(x_2, 1);
lean_dec(x_8);
x_9 = lp_mathlib_OrderDual_instLattice___redArg(x_6);
x_10 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_1);
x_11 = !lean_is_exclusive(x_1);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_12 = lean_ctor_get(x_1, 2);
lean_dec(x_12);
x_13 = lean_ctor_get(x_1, 1);
lean_dec(x_13);
x_14 = lean_ctor_get(x_1, 0);
lean_dec(x_14);
x_15 = lean_ctor_get(x_10, 1);
lean_inc(x_15);
lean_dec_ref(x_10);
x_16 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_16, 0, x_7);
lean_ctor_set(x_2, 2, x_16);
lean_ctor_set(x_2, 1, x_15);
lean_ctor_set(x_2, 0, x_9);
x_17 = lp_mathlib_OrderDual_instHeytingAlgebra___redArg___closed__0;
x_18 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, lean_box(0));
lean_closure_set(x_18, 2, lean_box(0));
lean_closure_set(x_18, 3, x_4);
lean_closure_set(x_18, 4, x_17);
x_19 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, lean_box(0));
lean_closure_set(x_19, 2, lean_box(0));
lean_closure_set(x_19, 3, x_17);
lean_closure_set(x_19, 4, x_18);
lean_ctor_set(x_1, 2, x_19);
return x_1;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
lean_dec(x_1);
x_20 = lean_ctor_get(x_10, 1);
lean_inc(x_20);
lean_dec_ref(x_10);
x_21 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_21, 0, x_7);
lean_ctor_set(x_2, 2, x_21);
lean_ctor_set(x_2, 1, x_20);
lean_ctor_set(x_2, 0, x_9);
x_22 = lp_mathlib_OrderDual_instHeytingAlgebra___redArg___closed__0;
x_23 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_23, 0, lean_box(0));
lean_closure_set(x_23, 1, lean_box(0));
lean_closure_set(x_23, 2, lean_box(0));
lean_closure_set(x_23, 3, x_4);
lean_closure_set(x_23, 4, x_22);
x_24 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_24, 0, lean_box(0));
lean_closure_set(x_24, 1, lean_box(0));
lean_closure_set(x_24, 2, lean_box(0));
lean_closure_set(x_24, 3, x_22);
lean_closure_set(x_24, 4, x_23);
x_25 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_25, 0, x_2);
lean_ctor_set(x_25, 1, x_3);
lean_ctor_set(x_25, 2, x_24);
return x_25;
}
}
else
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_26 = lean_ctor_get(x_2, 0);
x_27 = lean_ctor_get(x_2, 2);
lean_inc(x_27);
lean_inc(x_26);
lean_dec(x_2);
x_28 = lp_mathlib_OrderDual_instLattice___redArg(x_26);
x_29 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_1);
if (lean_is_exclusive(x_1)) {
 lean_ctor_release(x_1, 0);
 lean_ctor_release(x_1, 1);
 lean_ctor_release(x_1, 2);
 x_30 = x_1;
} else {
 lean_dec_ref(x_1);
 x_30 = lean_box(0);
}
x_31 = lean_ctor_get(x_29, 1);
lean_inc(x_31);
lean_dec_ref(x_29);
x_32 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instGeneralizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_32, 0, x_27);
x_33 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_33, 0, x_28);
lean_ctor_set(x_33, 1, x_31);
lean_ctor_set(x_33, 2, x_32);
x_34 = lp_mathlib_OrderDual_instHeytingAlgebra___redArg___closed__0;
x_35 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_35, 0, lean_box(0));
lean_closure_set(x_35, 1, lean_box(0));
lean_closure_set(x_35, 2, lean_box(0));
lean_closure_set(x_35, 3, x_4);
lean_closure_set(x_35, 4, x_34);
x_36 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, lean_box(0));
lean_closure_set(x_36, 2, lean_box(0));
lean_closure_set(x_36, 3, x_34);
lean_closure_set(x_36, 4, x_35);
if (lean_is_scalar(x_30)) {
 x_37 = lean_alloc_ctor(0, 3, 0);
} else {
 x_37 = x_30;
}
lean_ctor_set(x_37, 0, x_33);
lean_ctor_set(x_37, 1, x_3);
lean_ctor_set(x_37, 2, x_36);
return x_37;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instHeytingAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderDual_instHeytingAlgebra___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoheytingAlgebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 2);
lean_inc(x_6);
x_7 = lean_ctor_get(x_3, 0);
x_8 = lean_ctor_get(x_3, 2);
lean_inc(x_8);
x_9 = lean_ctor_get(x_2, 1);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 2);
lean_inc(x_10);
x_11 = !lean_is_exclusive(x_4);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; uint8_t x_17; 
x_12 = lean_ctor_get(x_4, 0);
x_13 = lean_ctor_get(x_4, 2);
x_14 = lean_ctor_get(x_4, 1);
lean_dec(x_14);
lean_inc_ref(x_7);
x_15 = lp_mathlib_Prod_instLattice___redArg(x_7, x_12);
x_16 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_1);
lean_dec_ref(x_1);
x_17 = !lean_is_exclusive(x_16);
if (x_17 == 0)
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; uint8_t x_21; 
x_18 = lean_ctor_get(x_16, 1);
x_19 = lean_ctor_get(x_16, 0);
lean_dec(x_19);
x_20 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_2);
x_21 = !lean_is_exclusive(x_2);
if (x_21 == 0)
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_22 = lean_ctor_get(x_2, 2);
lean_dec(x_22);
x_23 = lean_ctor_get(x_2, 1);
lean_dec(x_23);
x_24 = lean_ctor_get(x_2, 0);
lean_dec(x_24);
x_25 = !lean_is_exclusive(x_20);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_26 = lean_ctor_get(x_20, 0);
lean_dec(x_26);
lean_ctor_set(x_20, 0, x_18);
x_27 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_27, 0, x_8);
lean_closure_set(x_27, 1, x_13);
lean_ctor_set(x_4, 2, x_27);
lean_ctor_set(x_4, 1, x_20);
lean_ctor_set(x_4, 0, x_15);
lean_ctor_set(x_16, 1, x_9);
lean_ctor_set(x_16, 0, x_5);
x_28 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_28, 0, x_6);
lean_closure_set(x_28, 1, x_10);
lean_ctor_set(x_2, 2, x_28);
lean_ctor_set(x_2, 1, x_16);
return x_2;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
x_29 = lean_ctor_get(x_20, 1);
lean_inc(x_29);
lean_dec(x_20);
x_30 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_30, 0, x_18);
lean_ctor_set(x_30, 1, x_29);
x_31 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_31, 0, x_8);
lean_closure_set(x_31, 1, x_13);
lean_ctor_set(x_4, 2, x_31);
lean_ctor_set(x_4, 1, x_30);
lean_ctor_set(x_4, 0, x_15);
lean_ctor_set(x_16, 1, x_9);
lean_ctor_set(x_16, 0, x_5);
x_32 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_32, 0, x_6);
lean_closure_set(x_32, 1, x_10);
lean_ctor_set(x_2, 2, x_32);
lean_ctor_set(x_2, 1, x_16);
return x_2;
}
}
else
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; 
lean_dec(x_2);
x_33 = lean_ctor_get(x_20, 1);
lean_inc(x_33);
if (lean_is_exclusive(x_20)) {
 lean_ctor_release(x_20, 0);
 lean_ctor_release(x_20, 1);
 x_34 = x_20;
} else {
 lean_dec_ref(x_20);
 x_34 = lean_box(0);
}
if (lean_is_scalar(x_34)) {
 x_35 = lean_alloc_ctor(0, 2, 0);
} else {
 x_35 = x_34;
}
lean_ctor_set(x_35, 0, x_18);
lean_ctor_set(x_35, 1, x_33);
x_36 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_36, 0, x_8);
lean_closure_set(x_36, 1, x_13);
lean_ctor_set(x_4, 2, x_36);
lean_ctor_set(x_4, 1, x_35);
lean_ctor_set(x_4, 0, x_15);
lean_ctor_set(x_16, 1, x_9);
lean_ctor_set(x_16, 0, x_5);
x_37 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_37, 0, x_6);
lean_closure_set(x_37, 1, x_10);
x_38 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_38, 0, x_4);
lean_ctor_set(x_38, 1, x_16);
lean_ctor_set(x_38, 2, x_37);
return x_38;
}
}
else
{
lean_object* x_39; lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; 
x_39 = lean_ctor_get(x_16, 1);
lean_inc(x_39);
lean_dec(x_16);
x_40 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_2);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 x_41 = x_2;
} else {
 lean_dec_ref(x_2);
 x_41 = lean_box(0);
}
x_42 = lean_ctor_get(x_40, 1);
lean_inc(x_42);
if (lean_is_exclusive(x_40)) {
 lean_ctor_release(x_40, 0);
 lean_ctor_release(x_40, 1);
 x_43 = x_40;
} else {
 lean_dec_ref(x_40);
 x_43 = lean_box(0);
}
if (lean_is_scalar(x_43)) {
 x_44 = lean_alloc_ctor(0, 2, 0);
} else {
 x_44 = x_43;
}
lean_ctor_set(x_44, 0, x_39);
lean_ctor_set(x_44, 1, x_42);
x_45 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_45, 0, x_8);
lean_closure_set(x_45, 1, x_13);
lean_ctor_set(x_4, 2, x_45);
lean_ctor_set(x_4, 1, x_44);
lean_ctor_set(x_4, 0, x_15);
x_46 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_46, 0, x_5);
lean_ctor_set(x_46, 1, x_9);
x_47 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_47, 0, x_6);
lean_closure_set(x_47, 1, x_10);
if (lean_is_scalar(x_41)) {
 x_48 = lean_alloc_ctor(0, 3, 0);
} else {
 x_48 = x_41;
}
lean_ctor_set(x_48, 0, x_4);
lean_ctor_set(x_48, 1, x_46);
lean_ctor_set(x_48, 2, x_47);
return x_48;
}
}
else
{
lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; lean_object* x_60; lean_object* x_61; lean_object* x_62; lean_object* x_63; lean_object* x_64; 
x_49 = lean_ctor_get(x_4, 0);
x_50 = lean_ctor_get(x_4, 2);
lean_inc(x_50);
lean_inc(x_49);
lean_dec(x_4);
lean_inc_ref(x_7);
x_51 = lp_mathlib_Prod_instLattice___redArg(x_7, x_49);
x_52 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_1);
lean_dec_ref(x_1);
x_53 = lean_ctor_get(x_52, 1);
lean_inc(x_53);
if (lean_is_exclusive(x_52)) {
 lean_ctor_release(x_52, 0);
 lean_ctor_release(x_52, 1);
 x_54 = x_52;
} else {
 lean_dec_ref(x_52);
 x_54 = lean_box(0);
}
x_55 = lp_mathlib_CoheytingAlgebra_toBoundedOrder___redArg(x_2);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 lean_ctor_release(x_2, 1);
 lean_ctor_release(x_2, 2);
 x_56 = x_2;
} else {
 lean_dec_ref(x_2);
 x_56 = lean_box(0);
}
x_57 = lean_ctor_get(x_55, 1);
lean_inc(x_57);
if (lean_is_exclusive(x_55)) {
 lean_ctor_release(x_55, 0);
 lean_ctor_release(x_55, 1);
 x_58 = x_55;
} else {
 lean_dec_ref(x_55);
 x_58 = lean_box(0);
}
if (lean_is_scalar(x_58)) {
 x_59 = lean_alloc_ctor(0, 2, 0);
} else {
 x_59 = x_58;
}
lean_ctor_set(x_59, 0, x_53);
lean_ctor_set(x_59, 1, x_57);
x_60 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHImp___redArg___lam__0), 4, 2);
lean_closure_set(x_60, 0, x_8);
lean_closure_set(x_60, 1, x_50);
x_61 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_61, 0, x_51);
lean_ctor_set(x_61, 1, x_59);
lean_ctor_set(x_61, 2, x_60);
if (lean_is_scalar(x_54)) {
 x_62 = lean_alloc_ctor(0, 2, 0);
} else {
 x_62 = x_54;
}
lean_ctor_set(x_62, 0, x_5);
lean_ctor_set(x_62, 1, x_9);
x_63 = lean_alloc_closure((void*)(lp_mathlib_Prod_instHNot___redArg___lam__0), 3, 2);
lean_closure_set(x_63, 0, x_6);
lean_closure_set(x_63, 1, x_10);
if (lean_is_scalar(x_56)) {
 x_64 = lean_alloc_ctor(0, 3, 0);
} else {
 x_64 = x_56;
}
lean_ctor_set(x_64, 0, x_61);
lean_ctor_set(x_64, 1, x_62);
lean_ctor_set(x_64, 2, x_63);
return x_64;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCoheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instCoheytingAlgebra___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_1(x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg(x_2);
x_6 = lp_mathlib_Pi_instOrderTop___redArg(x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Pi_instHNotForall___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_4);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCoheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instCoheytingAlgebra___redArg(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Prop_instHeytingAlgebra___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Prop_instDistribLattice;
x_2 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
lean_ctor_set(x_2, 2, lean_box(0));
return x_2;
}
}
static lean_object* _init_lp_mathlib_Prop_instHeytingAlgebra___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Prop_instHeytingAlgebra___closed__0;
x_2 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
lean_ctor_set(x_2, 2, lean_box(0));
return x_2;
}
}
static lean_object* _init_lp_mathlib_Prop_instHeytingAlgebra() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Prop_instHeytingAlgebra___closed__1;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
lean_inc(x_3);
x_6 = lean_apply_2(x_5, x_3, x_4);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
return x_3;
}
else
{
lean_dec(x_3);
lean_inc(x_2);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
lean_inc(x_4);
x_6 = lean_apply_2(x_5, x_3, x_4);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
return x_4;
}
else
{
lean_dec(x_4);
lean_inc(x_2);
return x_2;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 5);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
lean_inc(x_2);
x_6 = lean_apply_2(x_5, x_4, x_2);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
return x_2;
}
else
{
lean_dec(x_2);
lean_inc(x_3);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = lean_ctor_get(x_1, 5);
lean_inc_ref(x_5);
lean_dec_ref(x_1);
lean_inc(x_2);
x_6 = lean_apply_2(x_5, x_4, x_2);
x_7 = lean_unbox(x_6);
if (x_7 == 0)
{
return x_2;
}
else
{
lean_dec(x_2);
lean_inc(x_3);
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1(x_1, x_2, x_3, x_4);
lean_dec(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3(x_1, x_2, x_3, x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_4 = lp_mathlib_LinearOrder_toLattice___redArg(x_2);
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_3, 1);
lean_inc(x_6);
lean_dec_ref(x_3);
lean_inc(x_6);
lean_inc_ref(x_2);
x_7 = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_6);
lean_inc(x_5);
lean_inc_ref(x_2);
x_8 = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1___boxed), 4, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_5);
lean_inc(x_5);
lean_inc(x_6);
lean_inc_ref(x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2___boxed), 4, 3);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_6);
lean_closure_set(x_9, 2, x_5);
lean_inc(x_6);
lean_inc(x_5);
x_10 = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3___boxed), 4, 3);
lean_closure_set(x_10, 0, x_2);
lean_closure_set(x_10, 1, x_5);
lean_closure_set(x_10, 2, x_6);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_5);
lean_ctor_set(x_11, 2, x_8);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_6);
lean_ctor_set(x_12, 2, x_9);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_7);
lean_ctor_set(x_13, 2, x_10);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_3 = lp_mathlib_LinearOrder_toLattice___redArg(x_1);
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec_ref(x_2);
lean_inc(x_5);
lean_inc_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__0___boxed), 4, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
lean_inc(x_4);
lean_inc_ref(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__1___boxed), 4, 2);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_4);
lean_inc(x_4);
lean_inc(x_5);
lean_inc_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__2___boxed), 4, 3);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_5);
lean_closure_set(x_8, 2, x_4);
lean_inc(x_5);
lean_inc(x_4);
x_9 = lean_alloc_closure((void*)(lp_mathlib_LinearOrder_toBiheytingAlgebra___redArg___lam__3___boxed), 4, 3);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_4);
lean_closure_set(x_9, 2, x_5);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_3);
lean_ctor_set(x_10, 1, x_4);
lean_ctor_set(x_10, 2, x_7);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_5);
lean_ctor_set(x_11, 2, x_8);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_6);
lean_ctor_set(x_12, 2, x_9);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBiheytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(x_1);
x_3 = lp_mathlib_OrderDual_instHeytingAlgebra___redArg(x_2);
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 2);
lean_dec(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_dec(x_7);
x_8 = lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(x_5);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_8, 2);
lean_inc(x_10);
lean_dec_ref(x_8);
x_11 = lean_ctor_get(x_9, 2);
lean_inc(x_11);
lean_dec_ref(x_9);
lean_ctor_set(x_1, 2, x_10);
lean_ctor_set(x_1, 1, x_11);
lean_ctor_set(x_1, 0, x_3);
return x_1;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_12 = lean_ctor_get(x_1, 0);
lean_inc(x_12);
lean_dec(x_1);
x_13 = lp_mathlib_OrderDual_instCoheytingAlgebra___redArg(x_12);
x_14 = lean_ctor_get(x_13, 0);
lean_inc_ref(x_14);
x_15 = lean_ctor_get(x_13, 2);
lean_inc(x_15);
lean_dec_ref(x_13);
x_16 = lean_ctor_get(x_14, 2);
lean_inc(x_16);
lean_dec_ref(x_14);
x_17 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_17, 0, x_3);
lean_ctor_set(x_17, 1, x_16);
lean_ctor_set(x_17, 2, x_15);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instBiheytingAlgebra(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderDual_instBiheytingAlgebra___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBiheytingAlgebra___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_5 = lp_mathlib_Prod_instHeytingAlgebra___redArg(x_3, x_4);
x_6 = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(x_1);
x_7 = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(x_2);
x_8 = lp_mathlib_Prod_instCoheytingAlgebra___redArg(x_6, x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
x_10 = lean_ctor_get(x_8, 2);
lean_inc(x_10);
lean_dec_ref(x_8);
x_11 = !lean_is_exclusive(x_9);
if (x_11 == 0)
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_12 = lean_ctor_get(x_9, 2);
x_13 = lean_ctor_get(x_9, 1);
lean_dec(x_13);
x_14 = lean_ctor_get(x_9, 0);
lean_dec(x_14);
lean_ctor_set(x_9, 2, x_10);
lean_ctor_set(x_9, 1, x_12);
lean_ctor_set(x_9, 0, x_5);
return x_9;
}
else
{
lean_object* x_15; lean_object* x_16; 
x_15 = lean_ctor_get(x_9, 2);
lean_inc(x_15);
lean_dec(x_9);
x_16 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_16, 0, x_5);
lean_ctor_set(x_16, 1, x_15);
lean_ctor_set(x_16, 2, x_10);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instBiheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instBiheytingAlgebra___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lp_mathlib_BiheytingAlgebra_toCoheytingAlgebra___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_instBiheytingAlgebra___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lp_mathlib_Pi_instHeytingAlgebra___redArg(x_2);
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__0), 2, 1);
lean_closure_set(x_5, 0, x_3);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Pi_instGeneralizedCoheytingAlgebra___redArg___lam__2), 4, 1);
lean_closure_set(x_6, 0, x_5);
x_7 = lp_mathlib_Pi_sdiff___redArg(x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCoheytingAlgebra___redArg___lam__2), 3, 1);
lean_closure_set(x_8, 0, x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Pi_instHNotForall___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_4);
lean_ctor_set(x_10, 1, x_7);
lean_ctor_set(x_10, 2, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instBiheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instBiheytingAlgebra___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_15, 0, x_4);
x_16 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_14);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_15);
x_19 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_5);
lean_ctor_set(x_19, 2, x_6);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_2);
x_7 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_6);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_3);
lean_ctor_set(x_10, 2, x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedHeytingAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_8);
lean_dec_ref(x_7);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_3);
x_15 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_15, 0, x_4);
x_16 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_14);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_17);
lean_ctor_set(x_18, 1, x_15);
x_19 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_19, 0, x_18);
lean_ctor_set(x_19, 1, x_5);
lean_ctor_set(x_19, 2, x_6);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_6, 0, x_2);
x_7 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_5);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_6);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_3);
lean_ctor_set(x_10, 2, x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_generalizedCoheytingAlgebra___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_Function_Injective_generalizedCoheytingAlgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec(x_8);
lean_dec_ref(x_7);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_18 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_19, 0, x_4);
x_20 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_18);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set(x_22, 1, x_19);
x_23 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_5);
lean_ctor_set(x_23, 2, x_8);
x_24 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_6);
lean_ctor_set(x_24, 2, x_7);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_2);
x_9 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_8);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_3);
lean_ctor_set(x_12, 2, x_6);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_4);
lean_ctor_set(x_13, 2, x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_heytingAlgebra___boxed(lean_object** _args) {
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
x_18 = lp_mathlib_Function_Injective_heytingAlgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_18 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_19, 0, x_4);
x_20 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_18);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set(x_22, 1, x_19);
x_23 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_23, 0, x_22);
lean_ctor_set(x_23, 1, x_6);
lean_ctor_set(x_23, 2, x_8);
x_24 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_24, 0, x_23);
lean_ctor_set(x_24, 1, x_5);
lean_ctor_set(x_24, 2, x_7);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_2);
x_9 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_8);
x_12 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_4);
lean_ctor_set(x_12, 2, x_6);
x_13 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_3);
lean_ctor_set(x_13, 2, x_5);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_coheytingAlgebra___boxed(lean_object** _args) {
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
x_18 = lp_mathlib_Function_Injective_coheytingAlgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17, lean_object* x_18, lean_object* x_19, lean_object* x_20, lean_object* x_21) {
_start:
{
lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_22 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_22, 0, x_3);
x_23 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_23, 0, x_4);
x_24 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_25 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_25, 0, x_24);
lean_ctor_set(x_25, 1, x_22);
x_26 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_26, 0, x_25);
lean_ctor_set(x_26, 1, x_23);
x_27 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_27, 0, x_26);
lean_ctor_set(x_27, 1, x_5);
lean_ctor_set(x_27, 2, x_9);
x_28 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_28, 0, x_27);
lean_ctor_set(x_28, 1, x_6);
lean_ctor_set(x_28, 2, x_7);
x_29 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_29, 0, x_28);
lean_ctor_set(x_29, 1, x_10);
lean_ctor_set(x_29, 2, x_8);
return x_29;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_9 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___redArg___lam__0), 3, 1);
lean_closure_set(x_10, 0, x_2);
x_11 = lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0;
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_11);
lean_ctor_set(x_12, 1, x_9);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_12);
lean_ctor_set(x_13, 1, x_10);
x_14 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_14, 0, x_13);
lean_ctor_set(x_14, 1, x_3);
lean_ctor_set(x_14, 2, x_7);
x_15 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_15, 0, x_14);
lean_ctor_set(x_15, 1, x_4);
lean_ctor_set(x_15, 2, x_5);
x_16 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_16, 0, x_15);
lean_ctor_set(x_16, 1, x_8);
lean_ctor_set(x_16, 2, x_6);
return x_16;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_biheytingAlgebra___boxed(lean_object** _args) {
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
lean_object* x_18 = _args[17];
lean_object* x_19 = _args[18];
lean_object* x_20 = _args[19];
lean_object* x_21 = _args[20];
_start:
{
lean_object* x_22; 
x_22 = lp_mathlib_Function_Injective_biheytingAlgebra(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20, x_21);
lean_dec(x_12);
lean_dec_ref(x_11);
return x_22;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instBiheytingAlgebra___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_box(0);
return x_3;
}
}
static lean_object* _init_lp_mathlib_PUnit_instBiheytingAlgebra() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_1 = lp_mathlib_PUnit_instLinearOrder;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_alloc_closure((void*)(lp_mathlib_PUnit_instBiheytingAlgebra___lam__0), 2, 0);
lean_inc_ref(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
lean_inc_ref(x_3);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_3);
x_6 = lean_box(0);
x_7 = lean_alloc_closure((void*)(lp_mathlib_PUnit_instBiheytingAlgebra___lam__1), 2, 1);
lean_closure_set(x_7, 0, x_6);
lean_inc_ref(x_3);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_3);
lean_inc_ref(x_7);
x_9 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_6);
lean_ctor_set(x_9, 2, x_7);
x_10 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_3);
lean_ctor_set(x_10, 2, x_7);
return x_10;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_PropInstances(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_Heyting_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_PropInstances(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_GaloisConnection_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0 = _init_lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_OrderDual_instGeneralizedCoheytingAlgebra___redArg___lam__0___closed__0);
lp_mathlib_OrderDual_instHeytingAlgebra___redArg___closed__0 = _init_lp_mathlib_OrderDual_instHeytingAlgebra___redArg___closed__0();
lean_mark_persistent(lp_mathlib_OrderDual_instHeytingAlgebra___redArg___closed__0);
lp_mathlib_Prop_instHeytingAlgebra___closed__0 = _init_lp_mathlib_Prop_instHeytingAlgebra___closed__0();
lean_mark_persistent(lp_mathlib_Prop_instHeytingAlgebra___closed__0);
lp_mathlib_Prop_instHeytingAlgebra___closed__1 = _init_lp_mathlib_Prop_instHeytingAlgebra___closed__1();
lean_mark_persistent(lp_mathlib_Prop_instHeytingAlgebra___closed__1);
lp_mathlib_Prop_instHeytingAlgebra = _init_lp_mathlib_Prop_instHeytingAlgebra();
lean_mark_persistent(lp_mathlib_Prop_instHeytingAlgebra);
lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0 = _init_lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0();
lean_mark_persistent(lp_mathlib_Function_Injective_generalizedHeytingAlgebra___closed__0);
lp_mathlib_PUnit_instBiheytingAlgebra = _init_lp_mathlib_PUnit_instBiheytingAlgebra();
lean_mark_persistent(lp_mathlib_PUnit_instBiheytingAlgebra);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
