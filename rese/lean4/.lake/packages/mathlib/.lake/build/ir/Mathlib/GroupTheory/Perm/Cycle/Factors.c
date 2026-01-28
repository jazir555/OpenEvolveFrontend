// Lean compiler output
// Module: Mathlib.GroupTheory.Perm.Cycle.Factors
// Imports: public import Init public import Mathlib.Data.List.Iterate public import Mathlib.Data.Set.Pairwise.List public import Mathlib.GroupTheory.Perm.Cycle.Basic public import Mathlib.GroupTheory.NoncommPiCoprod public import Mathlib.Tactic.Group
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
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_instDecidableRelSameCycle(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_lengthTR___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleOf___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_sort___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_List_iterateTR_loop___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsFinset(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_truncCycleFactors(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactors(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_instBEqOfDecidableEq___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsFinset___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_Perm_subtypePerm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_cycleFactors___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactors___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_List_elem___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactors___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_Perm_ofSubtype___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_truncCycleFactors___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleOf(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleOf___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_apply_1(x_2, x_3);
x_5 = lp_mathlib_Equiv_Perm_subtypePerm___redArg(x_1);
x_6 = lp_mathlib_Equiv_Perm_ofSubtype___redArg(x_4);
x_7 = lean_apply_1(x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleOf(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_cycleOf___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; 
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_3);
x_7 = l_instBEqOfDecidableEq___redArg(x_1);
x_8 = l_List_lengthTR___redArg(x_2);
x_9 = lean_box(0);
x_10 = lp_mathlib_List_iterateTR_loop___redArg(x_6, x_4, x_8, x_9);
x_11 = l_List_elem___redArg(x_7, x_5, x_10);
return x_11;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_instDecidableRelSameCycle(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; 
x_7 = lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg(x_2, x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lp_mathlib_Equiv_Perm_instDecidableRelSameCycle(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_3);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; 
x_6 = lp_mathlib_Equiv_Perm_instDecidableRelSameCycle___redArg(x_1, x_2, x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
uint8_t x_6; lean_object* x_7; 
x_6 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg___lam__0(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_2);
x_7 = lean_box(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_6; 
lean_dec_ref(x_5);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
x_6 = lean_box(0);
return x_6;
}
else
{
uint8_t x_7; 
x_7 = !lean_is_exclusive(x_4);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_8 = lean_ctor_get(x_4, 0);
x_9 = lean_ctor_get(x_4, 1);
x_10 = lean_ctor_get(x_5, 0);
lean_inc(x_10);
lean_inc(x_8);
x_11 = lean_apply_1(x_10, x_8);
lean_inc_ref(x_1);
lean_inc(x_8);
x_12 = lean_apply_2(x_1, x_11, x_8);
x_13 = lean_unbox(x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; 
lean_inc_ref(x_3);
lean_inc(x_2);
lean_inc_ref(x_1);
x_14 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_14, 0, x_1);
lean_closure_set(x_14, 1, x_2);
lean_closure_set(x_14, 2, x_3);
lean_inc_ref(x_3);
x_15 = lp_mathlib_Equiv_Perm_cycleOf___redArg(x_3, x_14, x_8);
lean_inc_ref(x_15);
x_16 = lp_mathlib_Equiv_symm___redArg(x_15);
x_17 = lp_mathlib_Equiv_trans___redArg(x_5, x_16);
x_18 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_1, x_2, x_3, x_9, x_17);
lean_ctor_set(x_4, 1, x_18);
lean_ctor_set(x_4, 0, x_15);
return x_4;
}
else
{
lean_free_object(x_4);
lean_dec(x_8);
x_4 = x_9;
goto _start;
}
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; uint8_t x_25; 
x_20 = lean_ctor_get(x_4, 0);
x_21 = lean_ctor_get(x_4, 1);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_4);
x_22 = lean_ctor_get(x_5, 0);
lean_inc(x_22);
lean_inc(x_20);
x_23 = lean_apply_1(x_22, x_20);
lean_inc_ref(x_1);
lean_inc(x_20);
x_24 = lean_apply_2(x_1, x_23, x_20);
x_25 = lean_unbox(x_24);
if (x_25 == 0)
{
lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; 
lean_inc_ref(x_3);
lean_inc(x_2);
lean_inc_ref(x_1);
x_26 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg___lam__0___boxed), 5, 3);
lean_closure_set(x_26, 0, x_1);
lean_closure_set(x_26, 1, x_2);
lean_closure_set(x_26, 2, x_3);
lean_inc_ref(x_3);
x_27 = lp_mathlib_Equiv_Perm_cycleOf___redArg(x_3, x_26, x_20);
lean_inc_ref(x_27);
x_28 = lp_mathlib_Equiv_symm___redArg(x_27);
x_29 = lp_mathlib_Equiv_trans___redArg(x_5, x_28);
x_30 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_1, x_2, x_3, x_21, x_29);
x_31 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_31, 0, x_27);
lean_ctor_set(x_31, 1, x_30);
return x_31;
}
else
{
lean_dec(x_20);
x_4 = x_21;
goto _start;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux_go(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_2, x_3, x_4, x_5, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
lean_inc_ref(x_5);
x_7 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_2, x_3, x_5, x_4, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsAux___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
lean_inc_ref(x_4);
x_5 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_1, x_2, x_4, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Perm_cycleFactors___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactors___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_Equiv_Perm_cycleFactors___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactors___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_2, 4);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_2, 5);
lean_inc_ref(x_5);
lean_dec_ref(x_2);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Perm_cycleFactors___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_6, 0, x_5);
lean_inc(x_1);
x_7 = lp_mathlib_Multiset_sort___redArg(x_1, x_4);
lean_inc_ref(x_3);
x_8 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_6, x_1, x_3, x_7, x_3);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactors(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Equiv_Perm_cycleFactors___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_truncCycleFactors(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
lean_inc_ref(x_4);
lean_inc(x_3);
x_5 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_2, x_3, x_4, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_truncCycleFactors___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
lean_inc_ref(x_3);
lean_inc(x_2);
x_4 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_1, x_2, x_3, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsFinset(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
lean_inc_ref(x_4);
lean_inc(x_3);
x_5 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_2, x_3, x_4, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Perm_cycleFactorsFinset___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
lean_inc_ref(x_3);
lean_inc(x_2);
x_4 = lp_mathlib_Equiv_Perm_cycleFactorsAux_go___redArg(x_1, x_2, x_3, x_2, x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_List_Iterate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Pairwise_List(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_Cycle_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_NoncommPiCoprod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Tactic_Group(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_GroupTheory_Perm_Cycle_Factors(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_List_Iterate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Pairwise_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Perm_Cycle_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_NoncommPiCoprod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Tactic_Group(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
