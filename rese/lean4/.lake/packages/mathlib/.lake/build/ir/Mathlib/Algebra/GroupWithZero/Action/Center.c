// Lean compiler output
// Module: Mathlib.Algebra.GroupWithZero.Action.Center
// Imports: public import Init public import Mathlib.Algebra.GroupWithZero.Units.Basic public import Mathlib.GroupTheory.Subgroup.Center
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
lean_object* lp_mathlib_SubgroupClass_toGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter(lean_object*, lean_object*);
lean_object* lp_mathlib_unitsCenterToCenterUnits(lean_object*, lean_object*);
lean_object* lp_mathlib_Units_instDivInvMonoid___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__0(lean_object*);
lean_object* lp_mathlib_MonoidHom_toHomUnits___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_unitsCenterToCenterUnits(lean_box(0), x_1);
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__1(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_MonoidHom_toHomUnits___redArg(x_1, x_2);
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__2(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_dec_ref(x_1);
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_dec(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__0___boxed), 1, 0);
lean_inc_ref(x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_7, 0, x_4);
x_8 = lp_mathlib_Units_instDivInvMonoid___redArg(x_4);
x_9 = lp_mathlib_SubgroupClass_toGroup___redArg(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__2___boxed), 3, 2);
lean_closure_set(x_10, 0, x_9);
lean_closure_set(x_10, 1, x_6);
lean_ctor_set(x_2, 1, x_7);
lean_ctor_set(x_2, 0, x_10);
return x_2;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_11 = lean_ctor_get(x_2, 0);
lean_inc(x_11);
lean_dec(x_2);
x_12 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__0___boxed), 1, 0);
lean_inc_ref(x_11);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__1___boxed), 2, 1);
lean_closure_set(x_13, 0, x_11);
x_14 = lp_mathlib_Units_instDivInvMonoid___redArg(x_11);
x_15 = lp_mathlib_SubgroupClass_toGroup___redArg(x_14);
x_16 = lean_alloc_closure((void*)(lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg___lam__2___boxed), 3, 2);
lean_closure_set(x_16, 0, x_15);
lean_closure_set(x_16, 1, x_12);
x_17 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_17, 0, x_16);
lean_ctor_set(x_17, 1, x_13);
return x_17;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Subgroup_centerUnitsEquivUnitsCenter___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_GroupTheory_Subgroup_Center(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Action_Center(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_GroupTheory_Subgroup_Center(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
