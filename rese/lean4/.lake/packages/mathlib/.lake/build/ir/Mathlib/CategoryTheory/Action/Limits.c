// Lean compiler output
// Module: Mathlib.CategoryTheory.Action.Limits
// Imports: public import Init public import Mathlib.CategoryTheory.Abelian.FunctorCategory public import Mathlib.CategoryTheory.Abelian.Transfer public import Mathlib.CategoryTheory.Linear.LinearFunctor public import Mathlib.CategoryTheory.Linear.FunctorCategory public import Mathlib.CategoryTheory.Preadditive.AdditiveFunctor public import Mathlib.CategoryTheory.Action.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulNatHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Action_functorCategoryEquivalence___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instHasZeroMorphisms(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instLinear___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instZeroHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instPreadditive___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulIntHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulNatHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instZeroHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instNegHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instAddHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSubHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_abelianAux___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instNegHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instPreadditive(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instAddHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instHasZeroMorphisms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSubHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_abelianAux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instNegHom___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instHasZeroMorphisms___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_abelianAux(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instLinear___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instHasZeroMorphisms___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_abelianAux___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instLinear___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instAddHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instZeroHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instNegHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Injective_addCommGroup___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSubHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulNatHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSubHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instAddHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulIntHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instPreadditive___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulIntHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instPreadditive___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulNatHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulIntHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Action_instZeroHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_ctor_get(x_3, 0);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_apply_2(x_1, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instZeroHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instZeroHom___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instZeroHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instZeroHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instHasZeroMorphisms___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Action_instZeroHom___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instHasZeroMorphisms___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Action_instHasZeroMorphisms___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instHasZeroMorphisms(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Action_instHasZeroMorphisms___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instHasZeroMorphisms___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Action_instHasZeroMorphisms(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instAddHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_apply_2(x_3, x_6, x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_apply_2(x_10, x_4, x_5);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instAddHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Action_instAddHom___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instAddHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instAddHom___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instAddHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instAddHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instNegHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lean_apply_2(x_3, x_5, x_6);
x_8 = lp_mathlib_SubNegZeroMonoid_toNegZeroClass___redArg(x_7);
lean_dec_ref(x_7);
x_9 = lean_ctor_get(x_8, 1);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_apply_1(x_9, x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instNegHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Action_instNegHom___redArg___lam__0), 4, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instNegHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instNegHom___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instNegHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instNegHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSubHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_apply_2(x_3, x_6, x_7);
x_9 = lean_ctor_get(x_8, 2);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_apply_2(x_9, x_4, x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSubHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Action_instSubHom___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSubHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instSubHom___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSubHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instSubHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulNatHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_apply_2(x_3, x_6, x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_ctor_get(x_9, 2);
lean_inc(x_10);
lean_dec_ref(x_9);
x_11 = lean_apply_2(x_10, x_4, x_5);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulNatHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Action_instSMulNatHom___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulNatHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instSMulNatHom___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulNatHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instSMulNatHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulIntHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_ctor_get(x_2, 0);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_apply_2(x_3, x_6, x_7);
x_9 = lean_ctor_get(x_8, 3);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_apply_2(x_9, x_4, x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulIntHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Action_instSMulIntHom___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulIntHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instSMulIntHom___redArg(x_5, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instSMulIntHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Action_instSMulIntHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instPreadditive___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_1);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Action_instAddHom___redArg___lam__0), 5, 3);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
lean_closure_set(x_5, 2, x_1);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_6 = lp_mathlib_Action_instZeroHom___redArg(x_2, x_3, x_4);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Action_instSMulNatHom___redArg___lam__0), 5, 3);
lean_closure_set(x_7, 0, x_3);
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Action_instNegHom___redArg___lam__0), 4, 3);
lean_closure_set(x_8, 0, x_3);
lean_closure_set(x_8, 1, x_4);
lean_closure_set(x_8, 2, x_1);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Action_instSubHom___redArg___lam__0), 5, 3);
lean_closure_set(x_9, 0, x_3);
lean_closure_set(x_9, 1, x_4);
lean_closure_set(x_9, 2, x_1);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Action_instSMulIntHom___redArg___lam__0), 5, 3);
lean_closure_set(x_10, 0, x_3);
lean_closure_set(x_10, 1, x_4);
lean_closure_set(x_10, 2, x_1);
x_11 = lp_mathlib_Function_Injective_addCommGroup___redArg(x_5, x_6, x_7, x_8, x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instPreadditive___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
lean_inc_ref(x_1);
x_2 = lp_mathlib_CategoryTheory_Preadditive_preadditiveHasZeroMorphisms___redArg(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Action_instPreadditive___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instPreadditive(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Action_instPreadditive___redArg(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instPreadditive___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Action_instPreadditive(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instLinear___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
lean_dec_ref(x_2);
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_dec_ref(x_3);
x_8 = lean_apply_4(x_1, x_6, x_7, x_4, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instLinear___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Action_instLinear___redArg___lam__0), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instLinear(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Action_instLinear___redArg(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_instLinear___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Action_instLinear(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_abelianAux(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Action_functorCategoryEquivalence___redArg(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_abelianAux___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Action_functorCategoryEquivalence___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_abelianAux___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Action_abelianAux(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Action_abelianAux___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Action_abelianAux___redArg(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_FunctorCategory(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Abelian_Transfer(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Linear_LinearFunctor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Linear_FunctorCategory(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Action_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Action_Limits(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Abelian_FunctorCategory(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Abelian_Transfer(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Linear_LinearFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Linear_FunctorCategory(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Preadditive_AdditiveFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Action_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
