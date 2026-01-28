// Lean compiler output
// Module: Mathlib.Algebra.Homology.ShortComplex.Linear
// Imports: public import Init public import Mathlib.Algebra.Homology.ShortComplex.Preadditive public import Mathlib.CategoryTheory.Linear.LinearFunctor
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
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_HomologyMapData_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instLinear___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_HomologyMapData_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instLinear___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_Homotopy_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instModuleHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_HomologyMapData_smul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_Homotopy_smul___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instModuleHom___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instSMulHom(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instLinear___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instModuleHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_Homotopy_smul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instLinear(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 2);
lean_inc(x_8);
lean_dec_ref(x_1);
x_9 = lean_ctor_get(x_2, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_2, 1);
lean_inc(x_10);
x_11 = lean_ctor_get(x_2, 2);
lean_inc(x_11);
lean_dec_ref(x_2);
x_12 = !lean_is_exclusive(x_5);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
x_13 = lean_ctor_get(x_5, 0);
x_14 = lean_ctor_get(x_5, 1);
x_15 = lean_ctor_get(x_5, 2);
lean_inc(x_3);
lean_inc(x_4);
x_16 = lean_apply_4(x_3, x_6, x_9, x_4, x_13);
lean_inc(x_3);
lean_inc(x_4);
x_17 = lean_apply_4(x_3, x_7, x_10, x_4, x_14);
x_18 = lean_apply_4(x_3, x_8, x_11, x_4, x_15);
lean_ctor_set(x_5, 2, x_18);
lean_ctor_set(x_5, 1, x_17);
lean_ctor_set(x_5, 0, x_16);
return x_5;
}
else
{
lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; 
x_19 = lean_ctor_get(x_5, 0);
x_20 = lean_ctor_get(x_5, 1);
x_21 = lean_ctor_get(x_5, 2);
lean_inc(x_21);
lean_inc(x_20);
lean_inc(x_19);
lean_dec(x_5);
lean_inc(x_3);
lean_inc(x_4);
x_22 = lean_apply_4(x_3, x_6, x_9, x_4, x_19);
lean_inc(x_3);
lean_inc(x_4);
x_23 = lean_apply_4(x_3, x_7, x_10, x_4, x_20);
x_24 = lean_apply_4(x_3, x_8, x_11, x_4, x_21);
x_25 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_25, 0, x_22);
lean_ctor_set(x_25, 1, x_23);
lean_ctor_set(x_25, 2, x_24);
return x_25;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instSMulHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg(x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_ShortComplex_instSMulHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instModuleHom(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg___lam__0), 5, 3);
lean_closure_set(x_9, 0, x_7);
lean_closure_set(x_9, 1, x_8);
lean_closure_set(x_9, 2, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instModuleHom___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg___lam__0), 5, 3);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
lean_closure_set(x_4, 2, x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instModuleHom___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_CategoryTheory_ShortComplex_instModuleHom(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instLinear___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_CategoryTheory_ShortComplex_instSMulHom___redArg___lam__0(x_2, x_3, x_1, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instLinear___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_CategoryTheory_ShortComplex_instLinear___redArg___lam__0), 5, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instLinear(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_ShortComplex_instLinear___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_instLinear___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_CategoryTheory_ShortComplex_instLinear(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_3, 1);
lean_inc(x_9);
lean_dec_ref(x_3);
x_10 = !lean_is_exclusive(x_4);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_4, 0);
x_12 = lean_ctor_get(x_4, 1);
lean_inc(x_1);
lean_inc(x_5);
x_13 = lean_apply_4(x_1, x_6, x_8, x_5, x_11);
x_14 = lean_apply_4(x_1, x_7, x_9, x_5, x_12);
lean_ctor_set(x_4, 1, x_14);
lean_ctor_set(x_4, 0, x_13);
return x_4;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_4, 0);
x_16 = lean_ctor_get(x_4, 1);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_4);
lean_inc(x_1);
lean_inc(x_5);
x_17 = lean_apply_4(x_1, x_6, x_8, x_5, x_15);
x_18 = lean_apply_4(x_1, x_7, x_9, x_5, x_16);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul___redArg(x_6, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
lean_dec_ref(x_2);
x_8 = lean_ctor_get(x_3, 0);
lean_inc(x_8);
x_9 = lean_ctor_get(x_3, 1);
lean_inc(x_9);
lean_dec_ref(x_3);
x_10 = !lean_is_exclusive(x_4);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_4, 0);
x_12 = lean_ctor_get(x_4, 1);
lean_inc(x_1);
lean_inc(x_5);
x_13 = lean_apply_4(x_1, x_6, x_8, x_5, x_11);
x_14 = lean_apply_4(x_1, x_7, x_9, x_5, x_12);
lean_ctor_set(x_4, 1, x_14);
lean_ctor_set(x_4, 0, x_13);
return x_4;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_4, 0);
x_16 = lean_ctor_get(x_4, 1);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_4);
lean_inc(x_1);
lean_inc(x_5);
x_17 = lean_apply_4(x_1, x_6, x_8, x_5, x_15);
x_18 = lean_apply_4(x_1, x_7, x_9, x_5, x_16);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul___redArg(x_6, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_HomologyMapData_smul___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_6 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_7);
lean_dec_ref(x_2);
x_8 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_8);
x_9 = lean_ctor_get(x_3, 1);
lean_inc_ref(x_9);
lean_dec_ref(x_3);
x_10 = !lean_is_exclusive(x_4);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_4, 0);
x_12 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
lean_inc(x_1);
x_13 = lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul___redArg(x_1, x_6, x_8, x_11, x_5);
x_14 = lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul___redArg(x_1, x_7, x_9, x_12, x_5);
lean_ctor_set(x_4, 1, x_14);
lean_ctor_set(x_4, 0, x_13);
return x_4;
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_15 = lean_ctor_get(x_4, 0);
x_16 = lean_ctor_get(x_4, 1);
lean_inc(x_16);
lean_inc(x_15);
lean_dec(x_4);
lean_inc(x_5);
lean_inc(x_1);
x_17 = lp_mathlib_CategoryTheory_ShortComplex_LeftHomologyMapData_smul___redArg(x_1, x_6, x_8, x_15, x_5);
x_18 = lp_mathlib_CategoryTheory_ShortComplex_RightHomologyMapData_smul___redArg(x_1, x_7, x_9, x_16, x_5);
x_19 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_19, 0, x_17);
lean_ctor_set(x_19, 1, x_18);
return x_19;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_HomologyMapData_smul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_ShortComplex_HomologyMapData_smul___redArg(x_6, x_10, x_11, x_12, x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_HomologyMapData_smul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13) {
_start:
{
lean_object* x_14; 
x_14 = lp_mathlib_CategoryTheory_ShortComplex_HomologyMapData_smul(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13);
lean_dec_ref(x_9);
lean_dec_ref(x_8);
lean_dec_ref(x_7);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_Homotopy_smul___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_6 = lean_ctor_get(x_2, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_2, 1);
lean_inc(x_7);
x_8 = lean_ctor_get(x_2, 2);
lean_inc(x_8);
lean_dec_ref(x_2);
x_9 = lean_ctor_get(x_3, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_3, 1);
lean_inc(x_10);
x_11 = lean_ctor_get(x_3, 2);
lean_inc(x_11);
lean_dec_ref(x_3);
x_12 = !lean_is_exclusive(x_4);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_13 = lean_ctor_get(x_4, 0);
x_14 = lean_ctor_get(x_4, 1);
x_15 = lean_ctor_get(x_4, 2);
x_16 = lean_ctor_get(x_4, 3);
lean_inc(x_1);
lean_inc(x_5);
lean_inc(x_9);
x_17 = lean_apply_4(x_1, x_6, x_9, x_5, x_13);
lean_inc(x_1);
lean_inc(x_5);
x_18 = lean_apply_4(x_1, x_7, x_9, x_5, x_14);
lean_inc(x_1);
lean_inc(x_5);
lean_inc(x_8);
x_19 = lean_apply_4(x_1, x_8, x_10, x_5, x_15);
x_20 = lean_apply_4(x_1, x_8, x_11, x_5, x_16);
lean_ctor_set(x_4, 3, x_20);
lean_ctor_set(x_4, 2, x_19);
lean_ctor_set(x_4, 1, x_18);
lean_ctor_set(x_4, 0, x_17);
return x_4;
}
else
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; 
x_21 = lean_ctor_get(x_4, 0);
x_22 = lean_ctor_get(x_4, 1);
x_23 = lean_ctor_get(x_4, 2);
x_24 = lean_ctor_get(x_4, 3);
lean_inc(x_24);
lean_inc(x_23);
lean_inc(x_22);
lean_inc(x_21);
lean_dec(x_4);
lean_inc(x_1);
lean_inc(x_5);
lean_inc(x_9);
x_25 = lean_apply_4(x_1, x_6, x_9, x_5, x_21);
lean_inc(x_1);
lean_inc(x_5);
x_26 = lean_apply_4(x_1, x_7, x_9, x_5, x_22);
lean_inc(x_1);
lean_inc(x_5);
lean_inc(x_8);
x_27 = lean_apply_4(x_1, x_8, x_10, x_5, x_23);
x_28 = lean_apply_4(x_1, x_8, x_11, x_5, x_24);
x_29 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_29, 0, x_25);
lean_ctor_set(x_29, 1, x_26);
lean_ctor_set(x_29, 2, x_27);
lean_ctor_set(x_29, 3, x_28);
return x_29;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_Homotopy_smul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_ShortComplex_Homotopy_smul___redArg(x_6, x_7, x_8, x_11, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_ShortComplex_Homotopy_smul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CategoryTheory_ShortComplex_Homotopy_smul(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec_ref(x_10);
lean_dec_ref(x_9);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_13;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_Preadditive(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Linear_LinearFunctor(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_Linear(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Homology_ShortComplex_Preadditive(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Linear_LinearFunctor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
