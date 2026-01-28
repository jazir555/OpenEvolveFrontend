// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.ProdHom
// Imports: public import Init public import Mathlib.Algebra.Group.Prod public import Mathlib.Algebra.GroupWithZero.Commute public import Mathlib.Algebra.GroupWithZero.Units.Lemmas public import Mathlib.Algebra.GroupWithZero.WithZero
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
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_MulEquiv_instEquivLike(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_lift_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inr___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_snd___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_fst(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__2;
lean_object* lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_snd(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_fst___redArg(lean_object*);
lean_object* lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(lean_object*);
static lean_object* lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inl___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Units_coeHom___lam__0___boxed(lean_object*);
static lean_object* lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__0;
lean_object* lp_mathlib_WithZero_map_x27___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Units_instMulOneClass___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Units_instMul___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inl___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_instMulZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inr___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_fst___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__1;
lean_object* lp_mathlib_MonoidHom_comp___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_inl___redArg(lean_object*);
lean_object* lp_mathlib_Prod_instMulOneClass___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_MonoidHom_inr___redArg(lean_object*);
lean_object* lp_mathlib_MonoidWithZeroHom_comp___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__0;
lean_object* lp_mathlib_MulHom_fst___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_MulHom_snd___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_snd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inl___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
x_5 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_4);
x_6 = lean_ctor_get(x_4, 0);
x_7 = lp_mathlib_Units_instMulOneClass___redArg(x_6);
x_8 = lean_ctor_get(x_2, 0);
x_9 = lean_ctor_get(x_8, 0);
x_10 = lp_mathlib_Units_instMulOneClass___redArg(x_9);
lean_inc_ref(x_10);
x_11 = lp_mathlib_Prod_instMulOneClass___redArg(x_7, x_10);
x_12 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_5);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_Units_instMul___redArg(x_6);
x_15 = lp_mathlib_WithZero_instMulZeroClass___redArg(x_14);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lp_mathlib_MulEquiv_instEquivLike(lean_box(0), lean_box(0), x_13, x_16);
lean_dec(x_16);
lean_dec(x_13);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lp_mathlib_MonoidHom_inl___redArg(x_10);
x_20 = lp_mathlib_WithZero_map_x27___redArg(x_11, x_19);
x_21 = lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(x_1, x_3);
x_22 = lp_mathlib_Equiv_symm___redArg(x_21);
x_23 = lean_apply_1(x_18, x_22);
x_24 = lp_mathlib_MonoidWithZeroHom_comp___redArg(x_20, x_23);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_inl___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inl___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_inl(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inl___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MonoidWithZeroHom_inl___redArg(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inr___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_4);
x_5 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_4);
x_6 = lean_ctor_get(x_4, 0);
x_7 = lp_mathlib_Units_instMulOneClass___redArg(x_6);
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_8, 0);
x_10 = lp_mathlib_Units_instMulOneClass___redArg(x_9);
lean_inc_ref(x_10);
x_11 = lp_mathlib_Prod_instMulOneClass___redArg(x_10, x_7);
x_12 = lp_mathlib_MulZeroOneClass_toMulZeroClass___redArg(x_5);
x_13 = lean_ctor_get(x_12, 0);
lean_inc(x_13);
lean_dec_ref(x_12);
x_14 = lp_mathlib_Units_instMul___redArg(x_6);
x_15 = lp_mathlib_WithZero_instMulZeroClass___redArg(x_14);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
lean_dec_ref(x_15);
x_17 = lp_mathlib_MulEquiv_instEquivLike(lean_box(0), lean_box(0), x_13, x_16);
lean_dec(x_16);
lean_dec(x_13);
x_18 = lean_ctor_get(x_17, 0);
lean_inc(x_18);
lean_dec_ref(x_17);
x_19 = lp_mathlib_MonoidHom_inr___redArg(x_10);
x_20 = lp_mathlib_WithZero_map_x27___redArg(x_11, x_19);
x_21 = lp_mathlib_WithZero_withZeroUnitsEquiv___redArg(x_2, x_3);
x_22 = lp_mathlib_Equiv_symm___redArg(x_21);
x_23 = lean_apply_1(x_18, x_22);
x_24 = lp_mathlib_MonoidWithZeroHom_comp___redArg(x_20, x_23);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_inr___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidWithZeroHom_inr(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_inr___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MonoidWithZeroHom_inr___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
static lean_object* _init_lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_MulHom_fst___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Units_coeHom___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__1;
x_2 = lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__0;
x_3 = lp_mathlib_MonoidHom_comp___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_fst___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_2);
x_4 = lp_mathlib_WithZero_lift_x27___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__2;
x_7 = lean_apply_1(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_fst(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_MonoidWithZeroHom_fst___redArg(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_fst___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_MonoidWithZeroHom_fst(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_MulHom_snd___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__0;
x_2 = lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__0;
x_3 = lp_mathlib_MonoidHom_comp___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_snd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = lp_mathlib_MonoidWithZero_toMulZeroOneClass___redArg(x_2);
x_4 = lp_mathlib_WithZero_lift_x27___redArg(x_3);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__1;
x_7 = lean_apply_1(x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_snd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_MonoidWithZeroHom_snd___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidWithZeroHom_snd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_MonoidWithZeroHom_snd(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Commute(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_ProdHom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Commute(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_WithZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__1 = _init_lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__1();
lean_mark_persistent(lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__1);
lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__0 = _init_lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__0();
lean_mark_persistent(lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__0);
lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__2 = _init_lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__2();
lean_mark_persistent(lp_mathlib_MonoidWithZeroHom_fst___redArg___closed__2);
lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__0 = _init_lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__0();
lean_mark_persistent(lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__0);
lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__1 = _init_lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__1();
lean_mark_persistent(lp_mathlib_MonoidWithZeroHom_snd___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
