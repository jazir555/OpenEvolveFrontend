// Lean compiler output
// Module: Mathlib.GroupTheory.Congruence.Basic
// Imports: public import Init public import Mathlib.Algebra.Group.Submonoid.Operations import all Mathlib.Algebra.Group.Submonoid.Operations public import Mathlib.Data.Setoid.Basic public import Mathlib.GroupTheory.Congruence.Hom
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
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Con_lift___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_submonoid(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Con_congr___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid(lean_object*, lean_object*);
lean_object* lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ofAddSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_prod___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ofAddSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_pi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_pi___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_ofSubmonoid___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubmonoid(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Con_congr___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_ofSubmonoid(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___redArg(lean_object*);
lean_object* lp_mathlib_Con_toQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_pi(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_prod(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_submonoid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_AddCon_toQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Quot_congr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubmonoid___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Con_prod(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_prod___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Con_prod(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_prod(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_box(0);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_prod___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AddCon_prod(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_4);
lean_dec(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_pi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_pi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Con_pi(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_pi(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_pi___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddCon_pi(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec(x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Con_congr___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Con_congr___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Con_congr___closed__0;
x_2 = lp_mathlib_Quot_congr___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_congr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Con_congr___closed__1;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_congr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Con_congr(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Con_congr___closed__1;
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_congr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddCon_congr(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_submonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_submonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Con_submonoid(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubmonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addSubmonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddCon_addSubmonoid(x_1, x_2, x_3);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_ofSubmonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_ofSubmonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Con_ofSubmonoid(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ofAddSubmonoid(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_ofAddSubmonoid___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddCon_ofAddSubmonoid(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Con_toSubmonoid___lam__0), 1, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_toSubmonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Con_toSubmonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_AddCon_toAddSubmonoid___lam__0), 1, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_toAddSubmonoid___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddCon_toAddSubmonoid(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Con_lift___redArg___lam__0(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 0);
lean_dec(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_2);
x_8 = lean_box(0);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Con_toQuotient___boxed), 4, 3);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_5);
lean_closure_set(x_9, 2, x_8);
x_10 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, x_9);
lean_closure_set(x_10, 4, x_3);
lean_ctor_set(x_1, 1, x_10);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_dec(x_1);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_2);
x_13 = lean_box(0);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Con_toQuotient___boxed), 4, 3);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_11);
lean_closure_set(x_14, 2, x_13);
x_15 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_15, 0, lean_box(0));
lean_closure_set(x_15, 1, lean_box(0));
lean_closure_set(x_15, 2, lean_box(0));
lean_closure_set(x_15, 3, x_14);
lean_closure_set(x_15, 4, x_3);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_16, 1, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg(x_3, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientKerEquivOfRightInverse___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Con_quotientKerEquivOfRightInverse(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_1);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_1, 0);
lean_dec(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(x_7, 0, x_2);
x_8 = lean_box(0);
x_9 = lean_alloc_closure((void*)(lp_mathlib_AddCon_toQuotient___boxed), 4, 3);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, x_5);
lean_closure_set(x_9, 2, x_8);
x_10 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_10, 0, lean_box(0));
lean_closure_set(x_10, 1, lean_box(0));
lean_closure_set(x_10, 2, lean_box(0));
lean_closure_set(x_10, 3, x_9);
lean_closure_set(x_10, 4, x_3);
lean_ctor_set(x_1, 1, x_10);
lean_ctor_set(x_1, 0, x_7);
return x_1;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_11 = lean_ctor_get(x_1, 1);
lean_inc(x_11);
lean_dec(x_1);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Con_quotientKerEquivOfRightInverse___redArg___lam__0), 2, 1);
lean_closure_set(x_12, 0, x_2);
x_13 = lean_box(0);
x_14 = lean_alloc_closure((void*)(lp_mathlib_AddCon_toQuotient___boxed), 4, 3);
lean_closure_set(x_14, 0, lean_box(0));
lean_closure_set(x_14, 1, x_11);
lean_closure_set(x_14, 2, x_13);
x_15 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_15, 0, lean_box(0));
lean_closure_set(x_15, 1, lean_box(0));
lean_closure_set(x_15, 2, lean_box(0));
lean_closure_set(x_15, 3, x_14);
lean_closure_set(x_15, 4, x_3);
x_16 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_16, 0, x_12);
lean_ctor_set(x_16, 1, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_AddCon_quotientKerEquivOfRightInverse___redArg(x_3, x_5, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientKerEquivOfRightInverse___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_AddCon_quotientKerEquivOfRightInverse(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_quotientQuotientEquivQuotient___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Con_quotientQuotientEquivQuotient(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Setoid_quotientQuotientEquivQuotient___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_quotientQuotientEquivQuotient___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddCon_quotientQuotientEquivQuotient(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_2);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Con_instSMul___redArg(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_instSMul___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Con_instSMul(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AddCon_instVAdd___redArg(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_instVAdd___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_AddCon_instVAdd(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulAction___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Con_mulAction(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCon_addAction___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_AddCon_addAction(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Con_instSMul___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Con_mulDistribMulAction___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Con_mulDistribMulAction(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec_ref(x_4);
lean_dec_ref(x_3);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Setoid_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Congruence_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Submonoid_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Setoid_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Congruence_Hom(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Con_congr___closed__0 = _init_lp_mathlib_Con_congr___closed__0();
lean_mark_persistent(lp_mathlib_Con_congr___closed__0);
lp_mathlib_Con_congr___closed__1 = _init_lp_mathlib_Con_congr___closed__1();
lean_mark_persistent(lp_mathlib_Con_congr___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
