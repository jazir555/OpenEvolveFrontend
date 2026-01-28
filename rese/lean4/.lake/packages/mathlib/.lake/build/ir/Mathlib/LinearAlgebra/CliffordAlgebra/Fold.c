// Lean compiler output
// Module: Mathlib.LinearAlgebra.CliffordAlgebra.Fold
// Imports: public import Init public import Mathlib.LinearAlgebra.CliffordAlgebra.Conjugation
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
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_AlgHom_toLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Module_End_instSemiring___redArg(lean_object*);
lean_object* lp_mathlib_Prod_subNegMonoid___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_prod___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instSMul___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_instRingCliffordAlgebra___redArg(lean_object*);
lean_object* lp_mathlib_Module_End_instAlgebra___redArg(lean_object*);
static lean_object* lp_mathlib_CliffordAlgebra_foldr_x27___redArg___closed__0;
lean_object* lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(lean_object*);
lean_object* lp_mathlib_CliffordAlgebra_00_u03b9___redArg(lean_object*);
lean_object* lp_mathlib_CliffordAlgebra_reverse___redArg(lean_object*);
lean_object* lp_mathlib_CliffordAlgebra_lift___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_fst___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_Algebra_lmul___redArg(lean_object*);
lean_object* lp_mathlib_Ring_toAddCommGroup___redArg(lean_object*);
lean_object* lp_mathlib_Semiring_toNonAssocSemiring___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_flip___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_snd___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_instAlgebraCliffordAlgebra___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_compl_u2082___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_5 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_5);
lean_dec_ref(x_2);
x_6 = lp_mathlib_Module_End_instSemiring___redArg(x_5);
x_7 = lp_mathlib_Module_End_instAlgebra___redArg(x_3);
x_8 = lp_mathlib_CliffordAlgebra_lift___redArg(x_1, x_6, x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lean_apply_1(x_9, x_4);
x_11 = lp_mathlib_AlgHom_toLinearMap___redArg(x_10);
x_12 = lp_mathlib_LinearMap_flip___redArg(x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CliffordAlgebra_foldr___redArg(x_4, x_6, x_8, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CliffordAlgebra_foldr(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_5);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldl___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_1);
x_5 = lp_mathlib_CliffordAlgebra_foldr___redArg(x_1, x_2, x_3, x_4);
x_6 = lp_mathlib_CliffordAlgebra_reverse___redArg(x_1);
x_7 = lp_mathlib_LinearMap_compl_u2082___redArg(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CliffordAlgebra_foldl___redArg(x_4, x_6, x_8, x_10);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldl___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
lean_object* x_12; 
x_12 = lp_mathlib_CliffordAlgebra_foldl(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_5);
return x_12;
}
}
static lean_object* _init_lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_fst___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lp_mathlib_LinearMap_compl_u2082___redArg(x_1, x_2);
lean_inc(x_4);
x_7 = lean_apply_1(x_6, x_4);
x_8 = lean_apply_1(x_3, x_4);
x_9 = lp_mathlib_LinearMap_prod___redArg(x_7, x_8);
x_10 = lean_apply_1(x_9, x_5);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_Algebra_lmul___redArg(x_4);
x_6 = lp_mathlib_AlgHom_toLinearMap___redArg(x_5);
x_7 = lp_mathlib_CliffordAlgebra_00_u03b9___redArg(x_1);
x_8 = lp_mathlib_LinearMap_comp___redArg(x_6, x_7);
x_9 = lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___closed__0;
x_10 = lean_alloc_closure((void*)(lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___lam__0), 5, 3);
lean_closure_set(x_10, 0, x_8);
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_2);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg(x_4, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27Aux___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_CliffordAlgebra_foldr_x27Aux(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_9);
lean_dec(x_8);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec_ref(x_5);
return x_11;
}
}
static lean_object* _init_lp_mathlib_CliffordAlgebra_foldr_x27___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_snd___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_6 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_1);
x_7 = lp_mathlib_instRingCliffordAlgebra___redArg(x_1);
lean_inc_ref(x_1);
x_8 = lp_mathlib_instAlgebraCliffordAlgebra___redArg(x_1);
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; 
x_10 = lean_ctor_get(x_8, 0);
x_11 = lean_ctor_get(x_8, 1);
lean_dec(x_11);
x_12 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_6);
x_13 = lp_mathlib_Ring_toAddCommGroup___redArg(x_7);
x_14 = lp_mathlib_Prod_subNegMonoid___redArg(x_13, x_2);
x_15 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_12);
x_16 = lean_ctor_get(x_15, 2);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lp_mathlib_Prod_instSMul___redArg(x_10, x_3);
x_18 = lp_mathlib_CliffordAlgebra_foldr_x27___redArg___closed__0;
lean_inc_ref(x_1);
x_19 = lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg(x_1, x_4);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_16);
lean_ctor_set(x_8, 1, x_5);
lean_ctor_set(x_8, 0, x_20);
x_21 = lp_mathlib_CliffordAlgebra_foldr___redArg(x_1, x_14, x_17, x_19);
x_22 = lean_apply_1(x_21, x_8);
x_23 = lp_mathlib_LinearMap_comp___redArg(x_18, x_22);
return x_23;
}
else
{
lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_24 = lean_ctor_get(x_8, 0);
lean_inc(x_24);
lean_dec(x_8);
x_25 = lp_mathlib_Semiring_toNonAssocSemiring___redArg(x_6);
x_26 = lp_mathlib_Ring_toAddCommGroup___redArg(x_7);
x_27 = lp_mathlib_Prod_subNegMonoid___redArg(x_26, x_2);
x_28 = lp_mathlib_NonAssocSemiring_toAddCommMonoidWithOne___redArg(x_25);
x_29 = lean_ctor_get(x_28, 2);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = lp_mathlib_Prod_instSMul___redArg(x_24, x_3);
x_31 = lp_mathlib_CliffordAlgebra_foldr_x27___redArg___closed__0;
lean_inc_ref(x_1);
x_32 = lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg(x_1, x_4);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_29);
x_34 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_34, 0, x_33);
lean_ctor_set(x_34, 1, x_5);
x_35 = lp_mathlib_CliffordAlgebra_foldr___redArg(x_1, x_27, x_30, x_32);
x_36 = lean_apply_1(x_35, x_34);
x_37 = lp_mathlib_LinearMap_comp___redArg(x_31, x_36);
return x_37;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CliffordAlgebra_foldr_x27___redArg(x_4, x_6, x_8, x_10, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CliffordAlgebra_foldr_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12) {
_start:
{
lean_object* x_13; 
x_13 = lp_mathlib_CliffordAlgebra_foldr_x27(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12);
lean_dec(x_9);
lean_dec(x_7);
lean_dec_ref(x_5);
return x_13;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Conjugation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Fold(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_CliffordAlgebra_Conjugation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___closed__0 = _init_lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CliffordAlgebra_foldr_x27Aux___redArg___closed__0);
lp_mathlib_CliffordAlgebra_foldr_x27___redArg___closed__0 = _init_lp_mathlib_CliffordAlgebra_foldr_x27___redArg___closed__0();
lean_mark_persistent(lp_mathlib_CliffordAlgebra_foldr_x27___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
