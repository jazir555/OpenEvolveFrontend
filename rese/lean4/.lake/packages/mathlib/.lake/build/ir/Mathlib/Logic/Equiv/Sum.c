// Lean compiler output
// Module: Mathlib.Logic.Equiv.Sum
// Imports: public import Init public import Mathlib.Data.Option.Defs public import Mathlib.Data.Sigma.Basic public import Mathlib.Logic.Equiv.Prod public import Mathlib.Tactic.Coe
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__0(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPSum___redArg(lean_object*, lean_object*);
lean_object* l_Sum_swap(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__0(lean_object*, lean_object*);
lean_object* l_Sum_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_prodComm(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__3___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__1(lean_object*);
static lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSum___lam__0(lean_object*);
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0(lean_object*);
lean_object* l_Sum_elim___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__2(lean_object*, lean_object*);
lean_object* l_Sum_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumComm(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__1;
lean_object* l_id___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodSumDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__1(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_sigmaSumDistrib___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0(uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__4(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1(lean_object*);
static lean_object* lp_mathlib_Equiv_sumCompl___redArg___closed__0;
static lean_object* lp_mathlib_Equiv_sumComm___closed__1;
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__3(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumSum___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
static lean_object* lp_mathlib_Equiv_emptySum___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_emptySum___closed__0;
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__3(lean_object*);
lean_object* lp_mathlib_Sigma_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1(lean_object*);
lean_object* lp_mathlib_Option_elim_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumProdDistrib(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_psumSum___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__3___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongr(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__3(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_emptySum(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSum(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__2(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_sumComm___closed__0;
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPSum(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__2;
static lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1;
static lean_object* lp_mathlib_Equiv_sumEmpty___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1___boxed(lean_object*);
static lean_object* lp_mathlib_Equiv_sumSumSumComm___closed__0;
static lean_object* lp_mathlib_Equiv_sumCompl___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__2(lean_object*);
lean_object* l_Sum_elim(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__5(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_prodSumDistrib___closed__4;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__2(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum___lam__0(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec(x_1);
x_4 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_4, 0, x_3);
return x_4;
}
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_1);
if (x_5 == 0)
{
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec(x_1);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumEquivSum(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_psumEquivSum___lam__0), 1, 0);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_psumEquivSum___lam__1), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_psumEquivSum___lam__2), 1, 0);
x_6 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, lean_box(0));
lean_closure_set(x_6, 3, x_4);
lean_closure_set(x_6, 4, x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_3);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCongr___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_1);
lean_inc_ref(x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCongr___redArg___lam__1), 2, 1);
lean_closure_set(x_4, 0, x_2);
x_5 = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, lean_box(0));
lean_closure_set(x_5, 3, lean_box(0));
lean_closure_set(x_5, 4, x_3);
lean_closure_set(x_5, 5, x_4);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCongr___redArg___lam__2), 2, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lp_mathlib_Equiv_symm___redArg(x_2);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCongr___redArg___lam__2), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, lean_box(0));
lean_closure_set(x_10, 4, x_7);
lean_closure_set(x_10, 5, x_9);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_5);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_sumCongr___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
uint8_t x_4; 
lean_dec_ref(x_2);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_1(x_6, x_5);
lean_ctor_set(x_3, 0, x_7);
return x_3;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
lean_dec(x_3);
x_9 = lean_ctor_get(x_1, 0);
lean_inc(x_9);
lean_dec_ref(x_1);
x_10 = lean_apply_1(x_9, x_8);
x_11 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_11, 0, x_10);
return x_11;
}
}
else
{
uint8_t x_12; 
lean_dec_ref(x_1);
x_12 = !lean_is_exclusive(x_3);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_3, 0);
x_14 = lean_ctor_get(x_2, 0);
lean_inc(x_14);
lean_dec_ref(x_2);
x_15 = lean_apply_1(x_14, x_13);
lean_ctor_set(x_3, 0, x_15);
return x_3;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_16 = lean_ctor_get(x_3, 0);
lean_inc(x_16);
lean_dec(x_3);
x_17 = lean_ctor_get(x_2, 0);
lean_inc(x_17);
lean_dec_ref(x_2);
x_18 = lean_apply_1(x_17, x_16);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
uint8_t x_4; 
lean_dec_ref(x_2);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lp_mathlib_Equiv_symm___redArg(x_1);
x_7 = lean_ctor_get(x_6, 0);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_apply_1(x_7, x_5);
lean_ctor_set(x_3, 0, x_8);
return x_3;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_3, 0);
lean_inc(x_9);
lean_dec(x_3);
x_10 = lp_mathlib_Equiv_symm___redArg(x_1);
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
x_12 = lean_apply_1(x_11, x_9);
x_13 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
else
{
uint8_t x_14; 
lean_dec_ref(x_1);
x_14 = !lean_is_exclusive(x_3);
if (x_14 == 0)
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_15 = lean_ctor_get(x_3, 0);
x_16 = lp_mathlib_Equiv_symm___redArg(x_2);
x_17 = lean_ctor_get(x_16, 0);
lean_inc(x_17);
lean_dec_ref(x_16);
x_18 = lean_apply_1(x_17, x_15);
lean_ctor_set(x_3, 0, x_18);
return x_3;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_19 = lean_ctor_get(x_3, 0);
lean_inc(x_19);
lean_dec(x_3);
x_20 = lp_mathlib_Equiv_symm___redArg(x_2);
x_21 = lean_ctor_get(x_20, 0);
lean_inc(x_21);
lean_dec_ref(x_20);
x_22 = lean_apply_1(x_21, x_19);
x_23 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_23, 0, x_22);
return x_23;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_psumCongr___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_psumCongr___redArg___lam__1), 3, 2);
lean_closure_set(x_4, 0, x_1);
lean_closure_set(x_4, 1, x_2);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_psumCongr___redArg(x_5, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Equiv_psumSum___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_psumEquivSum(lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumSum___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Equiv_psumCongr___redArg(x_1, x_2);
x_4 = lp_mathlib_Equiv_psumSum___redArg___closed__0;
x_5 = lp_mathlib_Equiv_trans___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_psumSum(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_psumSum___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPSum___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Equiv_symm___redArg(x_1);
x_4 = lp_mathlib_Equiv_symm___redArg(x_2);
x_5 = lp_mathlib_Equiv_psumSum___redArg(x_3, x_4);
x_6 = lp_mathlib_Equiv_symm___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumPSum(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_sumPSum___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSum___lam__0(lean_object* x_1) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc(x_3);
lean_dec(x_1);
x_4 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_4, 0, x_3);
return x_4;
}
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_1);
if (x_5 == 0)
{
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec(x_1);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_subtypeSum(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_subtypeSum___lam__0), 1, 0);
lean_inc_ref(x_4);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_sumCongr___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_sumCongr___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_sumCongr___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = 0;
return x_2;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = 1;
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(0);
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0(uint8_t x_1) {
_start:
{
if (x_1 == 0)
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0;
return x_2;
}
else
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1;
return x_3;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lean_unbox(x_1);
x_3 = lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Equiv_boolEquivPUnitSumPUnit() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___boxed), 1, 0);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__1___boxed), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__2___boxed), 1, 0);
x_4 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, lean_box(0));
lean_closure_set(x_4, 3, x_2);
lean_closure_set(x_4, 4, x_3);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_1);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumComm___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Sum_swap), 3, 2);
lean_closure_set(x_1, 0, lean_box(0));
lean_closure_set(x_1, 1, lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumComm___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Equiv_sumComm___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumComm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_sumComm___closed__1;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__2(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
x_3 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__3(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
x_3 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__4(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
x_3 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc___lam__5(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
x_3 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumAssoc(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumAssoc___lam__0), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumAssoc___lam__1), 1, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumAssoc___lam__2), 1, 0);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumAssoc___lam__3), 1, 0);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumAssoc___lam__4), 1, 0);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumAssoc___lam__5), 1, 0);
x_10 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, x_4);
lean_closure_set(x_10, 4, x_6);
x_11 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_11, 0, lean_box(0));
lean_closure_set(x_11, 1, lean_box(0));
lean_closure_set(x_11, 2, lean_box(0));
lean_closure_set(x_11, 3, x_10);
lean_closure_set(x_11, 4, x_7);
x_12 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, x_9);
lean_closure_set(x_12, 4, x_5);
x_13 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_13, 0, lean_box(0));
lean_closure_set(x_13, 1, lean_box(0));
lean_closure_set(x_13, 2, lean_box(0));
lean_closure_set(x_13, 3, x_8);
lean_closure_set(x_13, 4, x_12);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_11);
lean_ctor_set(x_14, 1, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__1(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__2(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__3(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_8 = lean_ctor_get(x_1, 0);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_ctor_get(x_2, 0);
lean_inc(x_9);
lean_dec_ref(x_2);
x_10 = lean_apply_1(x_8, x_7);
lean_inc(x_4);
x_11 = l_Sum_map___redArg(x_3, x_4, x_10);
lean_inc(x_4);
x_12 = l_Sum_map___redArg(x_5, x_4, x_11);
x_13 = l_Sum_map___redArg(x_6, x_4, x_12);
x_14 = lean_apply_1(x_9, x_13);
return x_14;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumAssoc(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Equiv_sumSumSumComm___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumComm(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumSumSumComm___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Equiv_sumSumSumComm___closed__2;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_sumSumSumComm___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSumSumComm(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__0___boxed), 1, 0);
x_6 = lp_mathlib_Equiv_sumSumSumComm___closed__0;
x_7 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__2), 2, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lp_mathlib_Equiv_sumSumSumComm___closed__1;
x_9 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__1), 2, 1);
lean_closure_set(x_9, 0, x_8);
x_10 = lp_mathlib_Equiv_sumSumSumComm___closed__2;
x_11 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__3), 2, 1);
lean_closure_set(x_11, 0, x_10);
lean_inc_ref(x_5);
x_12 = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(x_12, 0, lean_box(0));
lean_closure_set(x_12, 1, lean_box(0));
lean_closure_set(x_12, 2, lean_box(0));
lean_closure_set(x_12, 3, lean_box(0));
lean_closure_set(x_12, 4, x_5);
lean_closure_set(x_12, 5, x_11);
lean_inc_ref(x_9);
lean_inc_ref(x_5);
lean_inc_ref(x_7);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__5), 7, 6);
lean_closure_set(x_13, 0, x_8);
lean_closure_set(x_13, 1, x_6);
lean_closure_set(x_13, 2, x_7);
lean_closure_set(x_13, 3, x_5);
lean_closure_set(x_13, 4, x_12);
lean_closure_set(x_13, 5, x_9);
x_14 = lp_mathlib_Equiv_sumSumSumComm___closed__3;
x_15 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__3), 2, 1);
lean_closure_set(x_15, 0, x_14);
lean_inc_ref(x_5);
x_16 = lean_alloc_closure((void*)(l_Sum_map), 7, 6);
lean_closure_set(x_16, 0, lean_box(0));
lean_closure_set(x_16, 1, lean_box(0));
lean_closure_set(x_16, 2, lean_box(0));
lean_closure_set(x_16, 3, lean_box(0));
lean_closure_set(x_16, 4, x_5);
lean_closure_set(x_16, 5, x_15);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__5), 7, 6);
lean_closure_set(x_17, 0, x_8);
lean_closure_set(x_17, 1, x_6);
lean_closure_set(x_17, 2, x_7);
lean_closure_set(x_17, 3, x_5);
lean_closure_set(x_17, 4, x_16);
lean_closure_set(x_17, 5, x_9);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_13);
lean_ctor_set(x_18, 1, x_17);
return x_18;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__0(lean_object* x_1) {
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumEmpty___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_sumEmpty___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmpty(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEmpty___lam__0___boxed), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEmpty___lam__1), 1, 0);
x_6 = lp_mathlib_Equiv_sumEmpty___closed__0;
x_7 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_6);
lean_closure_set(x_7, 4, x_4);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_5);
return x_8;
}
}
static lean_object* _init_lp_mathlib_Equiv_emptySum___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumEmpty(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_emptySum___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Equiv_emptySum___closed__0;
x_2 = lp_mathlib_Equiv_sumSumSumComm___closed__2;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_emptySum(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_emptySum___closed__1;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = l_Sum_elim___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__0(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; lean_object* x_4; 
x_2 = 0;
x_3 = lean_box(x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__1(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; lean_object* x_4; 
x_2 = 1;
x_3 = lean_box(x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__3(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_unbox(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
x_5 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_5, 0, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool___lam__3___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_sumEquivSigmaBool___lam__3(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEquivSigmaBool(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEquivSigmaBool___lam__0), 1, 0);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEquivSigmaBool___lam__1), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEquivSigmaBool___lam__2), 3, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEquivSigmaBool___lam__3___boxed), 1, 0);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 1);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
lean_inc(x_2);
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__0___boxed), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaFiberEquiv___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaFiberEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_sigmaFiberEquiv___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
lean_inc(x_3);
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
if (x_5 == 0)
{
lean_object* x_6; 
x_6 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_6, 0, x_3);
return x_6;
}
else
{
lean_object* x_7; 
lean_dec(x_3);
x_7 = lean_box(0);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1___boxed), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Option_elim_x27___boxed), 5, 4);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_1);
lean_closure_set(x_5, 3, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_5);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, lean_box(0));
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaEquivOptionOfInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
lean_inc(x_2);
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_unbox(x_3);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_5, 0, x_2);
return x_5;
}
else
{
lean_object* x_6; 
x_6 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_6, 0, x_2);
return x_6;
}
}
}
static lean_object* _init_lp_mathlib_Equiv_sumCompl___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaEquivOptionOfInhabited___redArg___lam__1___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumCompl___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Equiv_sumCompl___redArg___closed__0;
x_2 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, lean_box(0));
lean_closure_set(x_2, 3, x_1);
lean_closure_set(x_2, 4, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumCompl___redArg___lam__2), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lp_mathlib_Equiv_sumCompl___redArg___closed__1;
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumCompl(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_sumCompl___redArg(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_prodComm(lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumProdDistrib(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Equiv_prodSumDistrib___closed__1;
x_2 = lp_mathlib_Equiv_prodSumDistrib___closed__0;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Equiv_prodSumDistrib___closed__0;
x_2 = lp_mathlib_Equiv_sumCongr___redArg(x_1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodSumDistrib___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Equiv_prodSumDistrib___closed__3;
x_2 = lp_mathlib_Equiv_prodSumDistrib___closed__2;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodSumDistrib(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_prodSumDistrib___closed__4;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__3(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_3, 0, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__2(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaSumDistrib___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_2);
lean_inc_ref(x_4);
x_5 = l_Sum_map___redArg(x_4, x_4, x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Equiv_sigmaSumDistrib___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSumSumComm___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_sigmaSumDistrib___lam__1(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib___lam__3___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_sigmaSumDistrib___lam__3(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sigmaSumDistrib(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaSumDistrib___lam__2), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaSumDistrib___lam__1___boxed), 2, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sigmaSumDistrib___lam__3___boxed), 2, 0);
x_7 = lp_mathlib_Equiv_sigmaSumDistrib___closed__0;
x_8 = lean_alloc_closure((void*)(lp_mathlib_Sigma_map), 7, 6);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, lean_box(0));
lean_closure_set(x_8, 4, x_7);
lean_closure_set(x_8, 5, x_5);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Sigma_map), 7, 6);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, lean_box(0));
lean_closure_set(x_9, 4, x_7);
lean_closure_set(x_9, 5, x_6);
x_10 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, x_8);
lean_closure_set(x_10, 4, x_9);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_4);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__1(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_7 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_7, 0, x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__2(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_1, 0, x_4);
return x_1;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_1);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_5);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_6);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
if (lean_obj_tag(x_2) == 0)
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_ctor_get(x_1, 0);
lean_dec(x_4);
x_5 = !lean_is_exclusive(x_2);
if (x_5 == 0)
{
lean_object* x_6; 
x_6 = lean_ctor_get(x_2, 0);
lean_ctor_set(x_1, 0, x_6);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
else
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec(x_2);
lean_ctor_set(x_1, 0, x_7);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_1);
return x_8;
}
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_dec(x_1);
x_10 = lean_ctor_get(x_2, 0);
lean_inc(x_10);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 x_11 = x_2;
} else {
 lean_dec_ref(x_2);
 x_11 = lean_box(0);
}
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_9);
if (lean_is_scalar(x_11)) {
 x_13 = lean_alloc_ctor(0, 1, 0);
} else {
 x_13 = x_11;
}
lean_ctor_set(x_13, 0, x_12);
return x_13;
}
}
else
{
uint8_t x_14; 
x_14 = !lean_is_exclusive(x_1);
if (x_14 == 0)
{
lean_object* x_15; uint8_t x_16; 
x_15 = lean_ctor_get(x_1, 0);
lean_dec(x_15);
x_16 = !lean_is_exclusive(x_2);
if (x_16 == 0)
{
lean_object* x_17; 
x_17 = lean_ctor_get(x_2, 0);
lean_ctor_set(x_1, 0, x_17);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
else
{
lean_object* x_18; lean_object* x_19; 
x_18 = lean_ctor_get(x_2, 0);
lean_inc(x_18);
lean_dec(x_2);
lean_ctor_set(x_1, 0, x_18);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_1);
return x_19;
}
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_20 = lean_ctor_get(x_1, 1);
lean_inc(x_20);
lean_dec(x_1);
x_21 = lean_ctor_get(x_2, 0);
lean_inc(x_21);
if (lean_is_exclusive(x_2)) {
 lean_ctor_release(x_2, 0);
 x_22 = x_2;
} else {
 lean_dec_ref(x_2);
 x_22 = lean_box(0);
}
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_21);
lean_ctor_set(x_23, 1, x_20);
if (lean_is_scalar(x_22)) {
 x_24 = lean_alloc_ctor(1, 1, 0);
} else {
 x_24 = x_22;
}
lean_ctor_set(x_24, 0, x_23);
return x_24;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumSigmaDistrib(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSigmaDistrib___lam__0), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSigmaDistrib___lam__1), 1, 0);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumSigmaDistrib___lam__2), 1, 0);
x_7 = lean_alloc_closure((void*)(l_Sum_elim), 6, 5);
lean_closure_set(x_7, 0, lean_box(0));
lean_closure_set(x_7, 1, lean_box(0));
lean_closure_set(x_7, 2, lean_box(0));
lean_closure_set(x_7, 3, x_5);
lean_closure_set(x_7, 4, x_6);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_4);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Sigma_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Coe(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Sum(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Sigma_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Coe(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_psumSum___redArg___closed__0 = _init_lp_mathlib_Equiv_psumSum___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_psumSum___redArg___closed__0);
lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0 = _init_lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__0);
lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1 = _init_lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_boolEquivPUnitSumPUnit___lam__0___closed__1);
lp_mathlib_Equiv_boolEquivPUnitSumPUnit = _init_lp_mathlib_Equiv_boolEquivPUnitSumPUnit();
lean_mark_persistent(lp_mathlib_Equiv_boolEquivPUnitSumPUnit);
lp_mathlib_Equiv_sumComm___closed__0 = _init_lp_mathlib_Equiv_sumComm___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_sumComm___closed__0);
lp_mathlib_Equiv_sumComm___closed__1 = _init_lp_mathlib_Equiv_sumComm___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_sumComm___closed__1);
lp_mathlib_Equiv_sumSumSumComm___closed__0 = _init_lp_mathlib_Equiv_sumSumSumComm___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_sumSumSumComm___closed__0);
lp_mathlib_Equiv_sumSumSumComm___closed__1 = _init_lp_mathlib_Equiv_sumSumSumComm___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_sumSumSumComm___closed__1);
lp_mathlib_Equiv_sumSumSumComm___closed__2 = _init_lp_mathlib_Equiv_sumSumSumComm___closed__2();
lean_mark_persistent(lp_mathlib_Equiv_sumSumSumComm___closed__2);
lp_mathlib_Equiv_sumSumSumComm___closed__3 = _init_lp_mathlib_Equiv_sumSumSumComm___closed__3();
lean_mark_persistent(lp_mathlib_Equiv_sumSumSumComm___closed__3);
lp_mathlib_Equiv_sumEmpty___closed__0 = _init_lp_mathlib_Equiv_sumEmpty___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_sumEmpty___closed__0);
lp_mathlib_Equiv_emptySum___closed__0 = _init_lp_mathlib_Equiv_emptySum___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_emptySum___closed__0);
lp_mathlib_Equiv_emptySum___closed__1 = _init_lp_mathlib_Equiv_emptySum___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_emptySum___closed__1);
lp_mathlib_Equiv_sumCompl___redArg___closed__0 = _init_lp_mathlib_Equiv_sumCompl___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_sumCompl___redArg___closed__0);
lp_mathlib_Equiv_sumCompl___redArg___closed__1 = _init_lp_mathlib_Equiv_sumCompl___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_sumCompl___redArg___closed__1);
lp_mathlib_Equiv_prodSumDistrib___closed__0 = _init_lp_mathlib_Equiv_prodSumDistrib___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_prodSumDistrib___closed__0);
lp_mathlib_Equiv_prodSumDistrib___closed__1 = _init_lp_mathlib_Equiv_prodSumDistrib___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_prodSumDistrib___closed__1);
lp_mathlib_Equiv_prodSumDistrib___closed__2 = _init_lp_mathlib_Equiv_prodSumDistrib___closed__2();
lean_mark_persistent(lp_mathlib_Equiv_prodSumDistrib___closed__2);
lp_mathlib_Equiv_prodSumDistrib___closed__3 = _init_lp_mathlib_Equiv_prodSumDistrib___closed__3();
lean_mark_persistent(lp_mathlib_Equiv_prodSumDistrib___closed__3);
lp_mathlib_Equiv_prodSumDistrib___closed__4 = _init_lp_mathlib_Equiv_prodSumDistrib___closed__4();
lean_mark_persistent(lp_mathlib_Equiv_prodSumDistrib___closed__4);
lp_mathlib_Equiv_sigmaSumDistrib___closed__0 = _init_lp_mathlib_Equiv_sigmaSumDistrib___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_sigmaSumDistrib___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
