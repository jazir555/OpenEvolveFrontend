// Lean compiler output
// Module: Mathlib.Algebra.Group.Nat.Hom
// Imports: public import Init public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Algebra.Group.TypeTags.Hom public import Mathlib.Tactic.Spread
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
lean_object* lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Additive_addMonoid___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powersMulHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg___lam__1(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_instAddCancelCommMonoid;
LEAN_EXPORT lean_object* lp_mathlib_powersHom___redArg(lean_object*);
lean_object* lp_mathlib_Additive_ofMul(lean_object*);
lean_object* lp_mathlib_AddMonoid_toAddZeroClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powersMulHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesAddHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesAddHom(lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_powersHom(lean_object*, lean_object*);
static lean_object* lp_mathlib_powersHom___redArg___closed__0;
static lean_object* lp_mathlib_powersHom___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lean_unsigned_to_nat(1u);
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_3, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_multiplesHom___redArg___lam__0), 1, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_multiplesHom___redArg___lam__1), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesHom(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_multiplesHom___redArg(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_powersHom___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_instAddCancelCommMonoid;
x_2 = lp_mathlib_AddMonoid_toAddZeroClass___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_powersHom___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Additive_ofMul(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powersHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_2 = lp_mathlib_powersHom___redArg___closed__0;
x_3 = lp_mathlib_powersHom___redArg___closed__1;
lean_inc_ref(x_1);
x_4 = lp_mathlib_Additive_addMonoid___redArg(x_1);
x_5 = lp_mathlib_multiplesHom___redArg(x_4);
x_6 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
lean_dec_ref(x_1);
x_7 = lp_mathlib_AddMonoidHom_toMultiplicativeLeft(lean_box(0), lean_box(0), x_2, x_6);
lean_dec_ref(x_6);
x_8 = lp_mathlib_Equiv_trans___redArg(x_5, x_7);
x_9 = lp_mathlib_Equiv_trans___redArg(x_3, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powersHom(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_powersHom___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesAddHom(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_multiplesHom___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_multiplesAddHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_multiplesHom___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powersMulHom(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_powersHom___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_powersMulHom___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_powersHom___redArg(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Spread(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Hom(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_TypeTags_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Spread(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_powersHom___redArg___closed__0 = _init_lp_mathlib_powersHom___redArg___closed__0();
lean_mark_persistent(lp_mathlib_powersHom___redArg___closed__0);
lp_mathlib_powersHom___redArg___closed__1 = _init_lp_mathlib_powersHom___redArg___closed__1();
lean_mark_persistent(lp_mathlib_powersHom___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
