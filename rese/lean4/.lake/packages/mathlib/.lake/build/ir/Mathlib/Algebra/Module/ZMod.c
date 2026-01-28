// Lean compiler output
// Module: Mathlib.Algebra.Module.ZMod
// Imports: public import Init public import Mathlib.Algebra.Module.LinearMap.Defs public import Mathlib.Algebra.Module.Submodule.Defs public import Mathlib.GroupTheory.Sylow
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
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_ZMod_commRing(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMap___redArg___boxed(lean_object*);
lean_object* lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMapEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMap___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toZModSubmodule___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toZModSubmodule___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule___redArg___boxed(lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
static lean_object* lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toZModSubmodule(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Submodule_toAddSubgroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_instFunLike___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule___redArg___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMap___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule___redArg(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; lean_object* x_8; 
x_6 = lean_unsigned_to_nat(0u);
x_7 = lean_nat_dec_eq(x_1, x_6);
x_8 = lean_alloc_closure((void*)(lp_mathlib_AddCommMonoid_zmodModule___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; lean_object* x_5; 
x_3 = lean_unsigned_to_nat(0u);
x_4 = lean_nat_dec_eq(x_1, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_AddCommMonoid_zmodModule___redArg___lam__0), 3, 1);
lean_closure_set(x_5, 0, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddCommMonoid_zmodModule(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommMonoid_zmodModule___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommMonoid_zmodModule___redArg(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; uint8_t x_8; 
x_5 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_3, 3);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lean_unsigned_to_nat(0u);
x_8 = lean_nat_dec_eq(x_1, x_7);
if (x_8 == 1)
{
lean_dec_ref(x_5);
return x_6;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; lean_object* x_13; 
lean_dec(x_6);
x_9 = lean_unsigned_to_nat(1u);
x_10 = lean_nat_sub(x_1, x_9);
x_11 = lean_nat_add(x_10, x_9);
lean_dec(x_10);
x_12 = lean_nat_dec_eq(x_11, x_7);
lean_dec(x_11);
x_13 = lean_alloc_closure((void*)(lp_mathlib_AddCommGroup_zmodModule___redArg___lam__0), 3, 1);
lean_closure_set(x_13, 0, x_5);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_2, 3);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_1, x_5);
if (x_6 == 1)
{
lean_dec_ref(x_3);
return x_4;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; lean_object* x_11; 
lean_dec(x_4);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_1, x_7);
x_9 = lean_nat_add(x_8, x_7);
lean_dec(x_8);
x_10 = lean_nat_dec_eq(x_9, x_5);
lean_dec(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_AddCommGroup_zmodModule___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_3);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddCommGroup_zmodModule(x_1, x_2, x_3, x_4);
lean_dec(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGroup_zmodModule___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCommGroup_zmodModule___redArg(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_2(x_5, x_2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
lean_inc_ref(x_3);
x_6 = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(x_3);
x_7 = lean_ctor_get(x_6, 3);
lean_inc(x_7);
lean_dec_ref(x_6);
x_8 = lean_unsigned_to_nat(0u);
x_9 = lean_nat_dec_eq(x_1, x_8);
if (x_9 == 1)
{
lean_dec_ref(x_3);
return x_7;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; lean_object* x_14; 
lean_dec(x_7);
x_10 = lean_unsigned_to_nat(1u);
x_11 = lean_nat_sub(x_1, x_10);
x_12 = lean_nat_add(x_11, x_10);
lean_dec(x_11);
x_13 = lean_nat_dec_eq(x_12, x_8);
lean_dec(x_12);
x_14 = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_zmodModule___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_3);
return x_14;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; 
lean_inc_ref(x_2);
x_3 = lp_mathlib_QuotientAddGroup_Quotient_addGroup___redArg(x_2);
x_4 = lean_ctor_get(x_3, 3);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_unsigned_to_nat(0u);
x_6 = lean_nat_dec_eq(x_1, x_5);
if (x_6 == 1)
{
lean_dec_ref(x_2);
return x_4;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; lean_object* x_11; 
lean_dec(x_4);
x_7 = lean_unsigned_to_nat(1u);
x_8 = lean_nat_sub(x_1, x_7);
x_9 = lean_nat_add(x_8, x_7);
lean_dec(x_8);
x_10 = lean_nat_dec_eq(x_9, x_5);
lean_dec(x_9);
x_11 = lean_alloc_closure((void*)(lp_mathlib_QuotientAddGroup_zmodModule___redArg___lam__0), 3, 1);
lean_closure_set(x_11, 0, x_2);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_QuotientAddGroup_zmodModule(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_QuotientAddGroup_zmodModule___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_QuotientAddGroup_zmodModule___redArg(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_inc(x_8);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMap___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMap___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_AddMonoidHom_toZModLinearMap(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_1);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMap___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddMonoidHom_toZModLinearMap___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_instFunLike___lam__0), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMapEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___lam__0), 2, 0);
x_9 = lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___closed__0;
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_8);
lean_ctor_set(x_10, 1, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_AddMonoidHom_toZModLinearMapEquiv(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
lean_dec_ref(x_4);
lean_dec(x_1);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toZModSubmodule___redArg___lam__0(lean_object* x_1) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toZModSubmodule___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_AddSubgroup_toZModSubmodule___redArg___lam__0), 1, 0);
x_5 = lp_mathlib_ZMod_commRing(x_1);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Submodule_toAddSubgroup___boxed), 6, 5);
lean_closure_set(x_6, 0, lean_box(0));
lean_closure_set(x_6, 1, lean_box(0));
lean_closure_set(x_6, 2, x_5);
lean_closure_set(x_6, 3, x_2);
lean_closure_set(x_6, 4, x_3);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_4);
lean_ctor_set(x_7, 1, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddSubgroup_toZModSubmodule(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddSubgroup_toZModSubmodule___redArg(x_1, x_3, x_4);
return x_5;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Sylow(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Module_ZMod(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_LinearMap_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Sylow(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___closed__0 = _init_lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___closed__0();
lean_mark_persistent(lp_mathlib_AddMonoidHom_toZModLinearMapEquiv___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
