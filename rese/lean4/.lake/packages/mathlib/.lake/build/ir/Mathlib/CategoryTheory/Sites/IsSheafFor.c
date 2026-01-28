// Lean compiler output
// Module: Mathlib.CategoryTheory.Sites.IsSheafFor
// Imports: public import Init public import Mathlib.CategoryTheory.Sites.Sieves public import Mathlib.CategoryTheory.Limits.Shapes.Pullback.Mono
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_map___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_compPresheafMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_pullback___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_pullback___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_pullback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_functorPullback___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_restrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_functorPullback___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_compPresheafMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_restrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_functorPullback(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CategoryTheory_eqToHom___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_restrict___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_compPresheafMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_internal_panic_unreachable();
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot___lam__0(x_1, x_2, x_3);
lean_dec(x_2);
lean_dec(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot___lam__0___boxed), 3, 0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CategoryTheory_Presieve_instInhabitedFamilyOfElementsBot(x_1, x_2, x_3, x_4);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_restrict(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lean_apply_3(x_8, x_9, x_10, lean_box(0));
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_restrict___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_3(x_1, x_2, x_3, lean_box(0));
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_restrict___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_restrict(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_map___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_5 = lean_apply_3(x_1, x_3, x_4, lean_box(0));
x_6 = lean_apply_2(x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_map___redArg(x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_map___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_map(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_3(x_3, x_1, x_2, lean_box(0));
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_dec_ref(x_1);
lean_inc(x_3);
x_9 = lp_mathlib_CategoryTheory_eqToHom___redArg(x_2, x_3);
x_10 = lean_apply_4(x_8, x_3, x_5, x_9, x_4);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__0), 3, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg___lam__1___boxed), 7, 3);
lean_closure_set(x_6, 0, x_2);
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_3);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_5);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___redArg(x_2, x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_singletonEquiv(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_functorPullback___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
lean_inc(x_4);
x_8 = lean_apply_1(x_6, x_4);
x_9 = lean_apply_3(x_7, x_4, x_2, x_5);
x_10 = lean_apply_3(x_3, x_8, x_9, lean_box(0));
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_functorPullback(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_functorPullback___redArg(x_6, x_7, x_9, x_10, x_11);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_functorPullback___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_functorPullback(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_pullback___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_ctor_get(x_1, 2);
lean_inc(x_8);
lean_dec_ref(x_1);
lean_inc(x_6);
x_9 = lean_apply_5(x_8, x_6, x_3, x_2, x_7, x_4);
x_10 = lean_apply_3(x_5, x_6, x_9, lean_box(0));
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_pullback(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_pullback___redArg(x_2, x_4, x_5, x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_pullback___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_pullback(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec_ref(x_3);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_compPresheafMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
lean_inc(x_3);
x_5 = lean_apply_3(x_2, x_3, x_4, lean_box(0));
x_6 = lean_apply_2(x_1, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_compPresheafMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_compPresheafMap___redArg(x_7, x_8, x_9, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_compPresheafMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CategoryTheory_Presieve_FamilyOfElements_compPresheafMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_2(x_1, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_3(x_1, x_2, x_3, lean_box(0));
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily___lam__0), 4, 0);
x_7 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily___lam__1), 3, 0);
x_8 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_8, 0, x_6);
lean_ctor_set(x_8, 1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_Presieve_natTransEquivCompatibleFamily(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
lean_inc(x_6);
x_8 = lean_apply_1(x_2, x_6);
x_9 = lean_apply_1(x_3, x_6);
x_10 = lean_apply_4(x_7, x_4, x_8, x_9, x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible___redArg___lam__0), 6, 5);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_3);
lean_closure_set(x_6, 2, x_4);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible___redArg(x_3, x_4, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_Presieve_Arrows_toCompatible(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_2);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_Sieves(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_Mono(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_CategoryTheory_Sites_IsSheafFor(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Sites_Sieves(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Shapes_Pullback_Mono(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
