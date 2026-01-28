// Lean compiler output
// Module: Batteries.Control.Nondet.Basic
// Imports: public import Init public import Batteries.Tactic.Lint.Misc public import Batteries.Data.MLList.Basic import Lean.Util.MonadBacktrack
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
lean_object* lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_nil(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_firstM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_batteries_Nondet_toList_x27___redArg___closed__0;
LEAN_EXPORT lean_object* lp_batteries_Nondet_singleton___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__12(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_firstM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_MLList_head___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofList___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__6(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_AlternativeMonad_toMonad___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Function_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__0(lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__9(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_MLList_append___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_MLList_singletonM___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instMonadLift(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_map(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instMonadLift___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_instMonadBacktrackUnitId__batteries___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instInhabited(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOption___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_nil___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__4(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_MLList_ofListM___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__3(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_head(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_map___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_MLList_force___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_instMonadBacktrackUnitId__batteries;
LEAN_EXPORT lean_object* lp_batteries_Nondet_map___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__7(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Function_const___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__11(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instInhabited___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_MLList_map___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__8(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_singleton(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Nondet_nil(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_nil___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Nondet_nil(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instInhabited(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_box(0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instInhabited___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Nondet_instInhabited(x_1, x_2, x_3, x_4);
lean_dec_ref(x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_2(x_1, lean_box(0), x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_box(0);
x_6 = lean_apply_1(x_1, x_5);
x_7 = lean_apply_4(x_2, lean_box(0), lean_box(0), x_6, x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_ctor_get(x_3, 1);
lean_inc(x_5);
lean_dec_ref(x_3);
x_6 = lean_alloc_closure((void*)(lp_batteries_Nondet_squash___redArg___lam__0), 2, 1);
lean_closure_set(x_6, 0, x_5);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_squash___redArg___lam__1), 4, 3);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_6);
x_8 = lean_alloc_ctor(3, 1, 0);
lean_ctor_set(x_8, 0, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_squash___redArg(x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_squash___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_squash(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Nondet_bind___redArg___lam__0(x_1, x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; 
lean_dec(x_4);
lean_dec_ref(x_3);
x_6 = lean_apply_2(x_1, lean_box(0), x_2);
return x_6;
}
else
{
lean_object* x_7; uint8_t x_8; 
lean_dec(x_2);
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec_ref(x_5);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_9 = lean_ctor_get(x_7, 1);
x_10 = lp_batteries_MLList_append___redArg(x_3, x_9, x_4);
lean_ctor_set_tag(x_7, 1);
lean_ctor_set(x_7, 1, x_10);
x_11 = lean_apply_2(x_1, lean_box(0), x_7);
return x_11;
}
else
{
lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_12 = lean_ctor_get(x_7, 0);
x_13 = lean_ctor_get(x_7, 1);
lean_inc(x_13);
lean_inc(x_12);
lean_dec(x_7);
x_14 = lp_batteries_MLList_append___redArg(x_3, x_13, x_4);
x_15 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_14);
x_16 = lean_apply_2(x_1, lean_box(0), x_15);
return x_16;
}
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_apply_1(x_1, x_2);
x_8 = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl(lean_box(0), lean_box(0), x_3, x_7);
x_9 = lean_apply_4(x_4, lean_box(0), lean_box(0), x_8, x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
if (lean_obj_tag(x_6) == 0)
{
lean_object* x_7; lean_object* x_8; 
lean_dec(x_5);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_7 = lean_box(0);
x_8 = lean_apply_2(x_1, lean_box(0), x_7);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_9 = lean_ctor_get(x_6, 0);
lean_inc(x_9);
lean_dec_ref(x_6);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc(x_11);
lean_dec(x_9);
x_12 = lean_ctor_get(x_10, 0);
lean_inc(x_12);
x_13 = lean_ctor_get(x_10, 1);
lean_inc(x_13);
lean_dec(x_10);
x_14 = lean_ctor_get(x_2, 1);
lean_inc(x_14);
lean_inc(x_4);
lean_inc_ref(x_3);
x_15 = lp_batteries_Nondet_bind___redArg(x_3, x_2, x_11, x_4);
lean_inc(x_15);
x_16 = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_16, 0, x_15);
lean_inc_ref(x_3);
x_17 = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__1), 5, 4);
lean_closure_set(x_17, 0, x_1);
lean_closure_set(x_17, 1, x_15);
lean_closure_set(x_17, 2, x_3);
lean_closure_set(x_17, 3, x_16);
lean_inc(x_5);
x_18 = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__2), 6, 5);
lean_closure_set(x_18, 0, x_4);
lean_closure_set(x_18, 1, x_12);
lean_closure_set(x_18, 2, x_3);
lean_closure_set(x_18, 3, x_5);
lean_closure_set(x_18, 4, x_17);
x_19 = lean_apply_1(x_14, x_13);
x_20 = lean_apply_4(x_5, lean_box(0), lean_box(0), x_19, x_18);
return x_20;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lp_batteries___private_Batteries_Data_MLList_Basic_0__MLList_unconsImpl(lean_box(0), lean_box(0), x_1, x_2);
x_7 = lean_apply_4(x_3, lean_box(0), lean_box(0), x_6, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
x_7 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_inc_ref(x_1);
lean_inc(x_7);
x_8 = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__3), 6, 5);
lean_closure_set(x_8, 0, x_7);
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_1);
lean_closure_set(x_8, 3, x_4);
lean_closure_set(x_8, 4, x_6);
lean_inc(x_6);
lean_inc_ref(x_1);
x_9 = lean_alloc_closure((void*)(lp_batteries_Nondet_bind___redArg___lam__4), 5, 4);
lean_closure_set(x_9, 0, x_1);
lean_closure_set(x_9, 1, x_3);
lean_closure_set(x_9, 2, x_6);
lean_closure_set(x_9, 3, x_8);
x_10 = lp_batteries_Nondet_squash___redArg(x_1, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_bind(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Nondet_bind___redArg(x_3, x_4, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_1);
lean_ctor_set(x_4, 1, x_3);
x_5 = lean_apply_2(x_2, lean_box(0), x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_ctor_get(x_1, 0);
lean_inc(x_5);
lean_dec_ref(x_1);
x_6 = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM___redArg___lam__0), 3, 2);
lean_closure_set(x_6, 0, x_4);
lean_closure_set(x_6, 1, x_2);
x_7 = lean_apply_4(x_3, lean_box(0), lean_box(0), x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM___redArg___lam__1), 4, 3);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_6);
lean_closure_set(x_7, 2, x_5);
lean_inc(x_5);
x_8 = lean_apply_4(x_5, lean_box(0), lean_box(0), x_3, x_7);
x_9 = lp_batteries_MLList_singletonM___redArg(x_1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singletonM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_singletonM___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singleton___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
x_6 = lean_apply_2(x_5, lean_box(0), x_3);
x_7 = lp_batteries_Nondet_singletonM___redArg(x_1, x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_singleton(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_singleton___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_box(0);
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__0___boxed), 2, 1);
lean_closure_set(x_7, 0, x_6);
x_8 = lp_batteries_Nondet_bind___redArg(x_1, x_2, x_5, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_box(0);
x_4 = lean_apply_1(x_1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__2), 2, 1);
lean_closure_set(x_5, 0, x_4);
x_6 = lp_batteries_MLList_append___redArg(x_1, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_2(x_1, lean_box(0), x_4);
x_6 = lp_batteries_Nondet_singletonM___redArg(x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, lean_box(0));
lean_closure_set(x_8, 3, x_1);
lean_closure_set(x_8, 4, x_6);
x_9 = lp_batteries_Nondet_bind___redArg(x_2, x_3, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__7(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_8 = lean_alloc_closure((void*)(l_Function_const___boxed), 4, 3);
lean_closure_set(x_8, 0, lean_box(0));
lean_closure_set(x_8, 1, lean_box(0));
lean_closure_set(x_8, 2, x_6);
x_9 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, x_1);
lean_closure_set(x_9, 4, x_8);
x_10 = lp_batteries_Nondet_bind___redArg(x_2, x_3, x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__6(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_apply_2(x_1, lean_box(0), x_5);
x_7 = lp_batteries_Nondet_singletonM___redArg(x_2, x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__8(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_6 = lean_box(0);
x_7 = lean_apply_1(x_1, x_6);
x_8 = lean_apply_1(x_2, lean_box(0));
x_9 = lean_alloc_closure((void*)(l_Function_comp), 6, 5);
lean_closure_set(x_9, 0, lean_box(0));
lean_closure_set(x_9, 1, lean_box(0));
lean_closure_set(x_9, 2, lean_box(0));
lean_closure_set(x_9, 3, x_8);
lean_closure_set(x_9, 4, x_5);
x_10 = lp_batteries_Nondet_bind___redArg(x_3, x_4, x_7, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__9(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_8 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__8), 5, 4);
lean_closure_set(x_8, 0, x_7);
lean_closure_set(x_8, 1, x_1);
lean_closure_set(x_8, 2, x_2);
lean_closure_set(x_8, 3, x_3);
x_9 = lp_batteries_Nondet_bind___redArg(x_2, x_3, x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, lean_box(0), x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__11(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__10___boxed), 3, 2);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
x_7 = lean_box(0);
x_8 = lean_apply_1(x_2, x_7);
x_9 = lp_batteries_Nondet_bind___redArg(x_3, x_4, x_8, x_6);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg___lam__12(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; 
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_8 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__11), 5, 4);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_7);
lean_closure_set(x_8, 2, x_2);
lean_closure_set(x_8, 3, x_3);
x_9 = lp_batteries_Nondet_bind___redArg(x_2, x_3, x_6, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_3);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_5 = lean_ctor_get(x_3, 1);
x_6 = lean_ctor_get(x_3, 4);
lean_dec(x_6);
x_7 = lean_ctor_get(x_3, 3);
lean_dec(x_7);
x_8 = lean_ctor_get(x_3, 2);
lean_dec(x_8);
x_9 = lean_ctor_get(x_3, 0);
lean_dec(x_9);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_10 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__1), 6, 2);
lean_closure_set(x_10, 0, x_1);
lean_closure_set(x_10, 1, x_2);
lean_inc_ref(x_1);
x_11 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__3), 4, 1);
lean_closure_set(x_11, 0, x_1);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_5);
x_12 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__4), 4, 3);
lean_closure_set(x_12, 0, x_5);
lean_closure_set(x_12, 1, x_1);
lean_closure_set(x_12, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_12);
x_13 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__5), 7, 3);
lean_closure_set(x_13, 0, x_12);
lean_closure_set(x_13, 1, x_1);
lean_closure_set(x_13, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_14 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__7), 7, 3);
lean_closure_set(x_14, 0, x_12);
lean_closure_set(x_14, 1, x_1);
lean_closure_set(x_14, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_15 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__6), 5, 3);
lean_closure_set(x_15, 0, x_5);
lean_closure_set(x_15, 1, x_1);
lean_closure_set(x_15, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_15);
x_16 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__9), 7, 3);
lean_closure_set(x_16, 0, x_15);
lean_closure_set(x_16, 1, x_1);
lean_closure_set(x_16, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_15);
x_17 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__12), 7, 3);
lean_closure_set(x_17, 0, x_15);
lean_closure_set(x_17, 1, x_1);
lean_closure_set(x_17, 2, x_2);
x_18 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_18, 0, x_13);
lean_ctor_set(x_18, 1, x_14);
lean_ctor_set(x_3, 4, x_10);
lean_ctor_set(x_3, 3, x_17);
lean_ctor_set(x_3, 2, x_16);
lean_ctor_set(x_3, 1, x_15);
lean_ctor_set(x_3, 0, x_18);
lean_inc_ref(x_2);
x_19 = lean_alloc_closure((void*)(lp_batteries_Nondet_nil___boxed), 4, 3);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, lean_box(0));
lean_closure_set(x_19, 2, x_2);
x_20 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_20, 0, x_3);
lean_ctor_set(x_20, 1, x_19);
lean_ctor_set(x_20, 2, x_11);
x_21 = lean_alloc_closure((void*)(lp_batteries_Nondet_bind), 8, 4);
lean_closure_set(x_21, 0, lean_box(0));
lean_closure_set(x_21, 1, lean_box(0));
lean_closure_set(x_21, 2, x_1);
lean_closure_set(x_21, 3, x_2);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_20);
lean_ctor_set(x_22, 1, x_21);
return x_22;
}
else
{
lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; 
x_23 = lean_ctor_get(x_3, 1);
lean_inc(x_23);
lean_dec(x_3);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_24 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__1), 6, 2);
lean_closure_set(x_24, 0, x_1);
lean_closure_set(x_24, 1, x_2);
lean_inc_ref(x_1);
x_25 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__3), 4, 1);
lean_closure_set(x_25, 0, x_1);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_23);
x_26 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__4), 4, 3);
lean_closure_set(x_26, 0, x_23);
lean_closure_set(x_26, 1, x_1);
lean_closure_set(x_26, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_26);
x_27 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__5), 7, 3);
lean_closure_set(x_27, 0, x_26);
lean_closure_set(x_27, 1, x_1);
lean_closure_set(x_27, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_28 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__7), 7, 3);
lean_closure_set(x_28, 0, x_26);
lean_closure_set(x_28, 1, x_1);
lean_closure_set(x_28, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_29 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__6), 5, 3);
lean_closure_set(x_29, 0, x_23);
lean_closure_set(x_29, 1, x_1);
lean_closure_set(x_29, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_29);
x_30 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__9), 7, 3);
lean_closure_set(x_30, 0, x_29);
lean_closure_set(x_30, 1, x_1);
lean_closure_set(x_30, 2, x_2);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc_ref(x_29);
x_31 = lean_alloc_closure((void*)(lp_batteries_Nondet_instAlternativeMonad___redArg___lam__12), 7, 3);
lean_closure_set(x_31, 0, x_29);
lean_closure_set(x_31, 1, x_1);
lean_closure_set(x_31, 2, x_2);
x_32 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_32, 0, x_27);
lean_ctor_set(x_32, 1, x_28);
x_33 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_33, 0, x_32);
lean_ctor_set(x_33, 1, x_29);
lean_ctor_set(x_33, 2, x_30);
lean_ctor_set(x_33, 3, x_31);
lean_ctor_set(x_33, 4, x_24);
lean_inc_ref(x_2);
x_34 = lean_alloc_closure((void*)(lp_batteries_Nondet_nil___boxed), 4, 3);
lean_closure_set(x_34, 0, lean_box(0));
lean_closure_set(x_34, 1, lean_box(0));
lean_closure_set(x_34, 2, x_2);
x_35 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_35, 0, x_33);
lean_ctor_set(x_35, 1, x_34);
lean_ctor_set(x_35, 2, x_25);
x_36 = lean_alloc_closure((void*)(lp_batteries_Nondet_bind), 8, 4);
lean_closure_set(x_36, 0, lean_box(0));
lean_closure_set(x_36, 1, lean_box(0));
lean_closure_set(x_36, 2, x_1);
lean_closure_set(x_36, 3, x_2);
x_37 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_37, 0, x_35);
lean_ctor_set(x_37, 1, x_36);
return x_37;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instAlternativeMonad(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_batteries_Nondet_instAlternativeMonad___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instMonadLift(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM), 6, 4);
lean_closure_set(x_5, 0, lean_box(0));
lean_closure_set(x_5, 1, lean_box(0));
lean_closure_set(x_5, 2, x_3);
lean_closure_set(x_5, 3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_instMonadLift___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM), 6, 4);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, lean_box(0));
lean_closure_set(x_3, 2, x_1);
lean_closure_set(x_3, 3, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_batteries_Nondet_singletonM___redArg___lam__0), 3, 2);
lean_closure_set(x_5, 0, x_4);
lean_closure_set(x_5, 1, x_1);
x_6 = lean_apply_4(x_2, lean_box(0), lean_box(0), x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_4(x_1, lean_box(0), lean_box(0), x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_1);
x_6 = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__0), 4, 3);
lean_closure_set(x_6, 0, x_1);
lean_closure_set(x_6, 1, x_5);
lean_closure_set(x_6, 2, x_2);
x_7 = lean_apply_1(x_3, x_4);
x_8 = lean_apply_4(x_1, lean_box(0), lean_box(0), x_7, x_6);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__2), 5, 4);
lean_closure_set(x_8, 0, x_1);
lean_closure_set(x_8, 1, x_2);
lean_closure_set(x_8, 2, x_3);
lean_closure_set(x_8, 3, x_7);
x_9 = lean_box(0);
x_10 = l_List_mapTR_loop___redArg(x_8, x_4, x_9);
x_11 = lp_batteries_MLList_ofListM___redArg(x_5, x_10);
x_12 = lean_apply_2(x_6, lean_box(0), x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg___lam__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
x_8 = lean_ctor_get(x_1, 1);
lean_inc(x_8);
lean_dec_ref(x_1);
lean_inc(x_7);
lean_inc(x_3);
lean_inc(x_2);
x_9 = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__1), 4, 3);
lean_closure_set(x_9, 0, x_2);
lean_closure_set(x_9, 1, x_3);
lean_closure_set(x_9, 2, x_7);
lean_inc(x_3);
x_10 = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__3), 7, 6);
lean_closure_set(x_10, 0, x_3);
lean_closure_set(x_10, 1, x_9);
lean_closure_set(x_10, 2, x_8);
lean_closure_set(x_10, 3, x_4);
lean_closure_set(x_10, 4, x_5);
lean_closure_set(x_10, 5, x_2);
x_11 = lean_apply_4(x_3, lean_box(0), lean_box(0), x_7, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_1);
lean_inc(x_5);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_ofListM___redArg___lam__4), 6, 5);
lean_closure_set(x_7, 0, x_2);
lean_closure_set(x_7, 1, x_6);
lean_closure_set(x_7, 2, x_5);
lean_closure_set(x_7, 3, x_3);
lean_closure_set(x_7, 4, x_1);
x_8 = lp_batteries_Nondet_squash___redArg(x_1, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofListM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_ofListM___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofList___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
x_6 = lean_apply_1(x_5, lean_box(0));
x_7 = lean_box(0);
x_8 = l_List_mapTR_loop___redArg(x_6, x_3, x_7);
x_9 = lp_batteries_Nondet_ofListM___redArg(x_1, x_2, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofList(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_ofList___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_1, x_4);
x_6 = lp_batteries_Nondet_singletonM___redArg(x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_5 = lean_alloc_closure((void*)(lp_batteries_Nondet_mapM___redArg___lam__0), 4, 3);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_2);
x_6 = lp_batteries_Nondet_bind___redArg(x_1, x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_mapM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Nondet_mapM___redArg(x_3, x_4, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_map___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_2(x_2, lean_box(0), x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_map___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_map___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_3);
lean_closure_set(x_7, 1, x_6);
x_8 = lp_batteries_Nondet_mapM___redArg(x_1, x_2, x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Nondet_map___redArg(x_3, x_4, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; lean_object* x_6; 
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_5 = lean_box(0);
x_6 = lean_apply_2(x_1, lean_box(0), x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_4, 0);
lean_inc(x_7);
lean_dec_ref(x_4);
x_8 = lp_batteries_Nondet_singleton___redArg(x_2, x_3, x_7);
x_9 = lean_apply_2(x_1, lean_box(0), x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_apply_4(x_1, lean_box(0), lean_box(0), x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_1);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_ofOptionM___redArg___lam__0), 4, 3);
lean_closure_set(x_7, 0, x_6);
lean_closure_set(x_7, 1, x_1);
lean_closure_set(x_7, 2, x_2);
lean_inc(x_5);
x_8 = lean_alloc_closure((void*)(lp_batteries_Nondet_ofOptionM___redArg___lam__1), 4, 3);
lean_closure_set(x_8, 0, x_5);
lean_closure_set(x_8, 1, x_3);
lean_closure_set(x_8, 2, x_7);
x_9 = lp_batteries_Nondet_squash___redArg(x_1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOptionM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_ofOptionM___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOption___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
x_6 = lean_apply_2(x_5, lean_box(0), x_3);
x_7 = lp_batteries_Nondet_ofOptionM___redArg(x_1, x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_ofOption(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_ofOption___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_1(x_1, x_4);
x_6 = lp_batteries_Nondet_ofOptionM___redArg(x_2, x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_5 = lean_alloc_closure((void*)(lp_batteries_Nondet_filterMapM___redArg___lam__0), 4, 3);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_1);
lean_closure_set(x_5, 2, x_2);
x_6 = lp_batteries_Nondet_bind___redArg(x_1, x_2, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMapM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Nondet_filterMapM___redArg(x_3, x_4, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_2(x_2, lean_box(0), x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_filterMap___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_3);
lean_closure_set(x_7, 1, x_6);
x_8 = lp_batteries_Nondet_filterMapM___redArg(x_1, x_2, x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Nondet_filterMap___redArg(x_3, x_4, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__0(lean_object* x_1, lean_object* x_2, uint8_t x_3) {
_start:
{
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
lean_dec(x_2);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_box(0);
x_6 = lean_apply_2(x_4, lean_box(0), x_5);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_8, 0, x_2);
x_9 = lean_apply_2(x_7, lean_box(0), x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lean_unbox(x_3);
x_5 = lp_batteries_Nondet_filterM___redArg___lam__0(x_1, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_batteries_Nondet_filterM___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
x_6 = lean_apply_1(x_2, x_4);
x_7 = lean_apply_4(x_3, lean_box(0), lean_box(0), x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 1);
lean_inc(x_6);
lean_inc_ref(x_5);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_filterM___redArg___lam__1), 4, 3);
lean_closure_set(x_7, 0, x_5);
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_6);
x_8 = lp_batteries_Nondet_filterMapM___redArg(x_1, x_2, x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filterM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries_Nondet_filterM___redArg(x_3, x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_2(x_2, lean_box(0), x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_filter___redArg___lam__0), 3, 2);
lean_closure_set(x_7, 0, x_3);
lean_closure_set(x_7, 1, x_6);
x_8 = lp_batteries_Nondet_filterM___redArg(x_1, x_2, x_7, x_4);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_filter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries_Nondet_filter___redArg(x_3, x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc(x_1);
x_6 = lean_apply_1(x_1, x_2);
lean_inc_ref(x_4);
lean_inc_ref(x_3);
x_7 = lean_alloc_closure((void*)(lp_batteries_Nondet_iterate___redArg), 4, 3);
lean_closure_set(x_7, 0, x_3);
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_1);
x_8 = lp_batteries_Nondet_bind___redArg(x_3, x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_2);
lean_inc_ref(x_1);
lean_inc(x_4);
x_5 = lean_alloc_closure((void*)(lp_batteries_Nondet_iterate___redArg___lam__0), 5, 4);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
lean_closure_set(x_5, 2, x_1);
lean_closure_set(x_5, 3, x_2);
lean_inc_ref(x_1);
x_6 = lp_batteries_Nondet_singleton___redArg(x_1, x_2, x_4);
x_7 = lp_batteries_MLList_append___redArg(x_1, x_6, x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_iterate(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_batteries_Nondet_iterate___redArg(x_3, x_4, x_6, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_batteries_Nondet_toMLList_x27___redArg___lam__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_batteries_Nondet_toMLList_x27___redArg___lam__0___boxed), 1, 0);
x_4 = lp_batteries_MLList_map___redArg(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_toMLList_x27___redArg(x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toMLList_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_toMLList_x27(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_MLList_force___redArg(x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_MLList_force___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_toList(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
static lean_object* _init_lp_batteries_Nondet_toList_x27___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_batteries_Nondet_toMLList_x27___redArg___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_batteries_Nondet_toList_x27___redArg___closed__0;
lean_inc_ref(x_1);
x_4 = lp_batteries_MLList_map___redArg(x_1, x_3, x_2);
x_5 = lp_batteries_MLList_force___redArg(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_toList_x27___redArg(x_3, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_toList_x27___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_toList_x27(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, lean_box(0), x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_alloc_closure((void*)(lp_batteries_Nondet_head___redArg___lam__0), 3, 2);
lean_closure_set(x_8, 0, x_2);
lean_closure_set(x_8, 1, x_5);
x_9 = lean_apply_1(x_7, x_6);
x_10 = lean_apply_4(x_3, lean_box(0), lean_box(0), x_9, x_8);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_head___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
lean_inc_ref(x_1);
x_4 = lp_batteries_AlternativeMonad_toMonad___redArg(x_1);
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_5, 0);
x_7 = lean_ctor_get(x_4, 1);
lean_inc(x_7);
lean_dec_ref(x_4);
x_8 = lean_ctor_get(x_6, 1);
lean_inc(x_8);
x_9 = lp_batteries_MLList_head___redArg(x_1, x_3);
lean_inc(x_7);
x_10 = lean_alloc_closure((void*)(lp_batteries_Nondet_head___redArg___lam__1), 4, 3);
lean_closure_set(x_10, 0, x_2);
lean_closure_set(x_10, 1, x_8);
lean_closure_set(x_10, 2, x_7);
x_11 = lean_apply_4(x_7, lean_box(0), lean_box(0), x_9, x_10);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_head(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_Nondet_head___redArg(x_3, x_4, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_firstM___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
lean_inc_ref(x_1);
x_5 = lp_batteries_AlternativeMonad_toMonad___redArg(x_1);
lean_inc_ref(x_2);
x_6 = lp_batteries_Nondet_filterMapM___redArg(x_5, x_2, x_4, x_3);
x_7 = lp_batteries_Nondet_head___redArg(x_1, x_2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_batteries_Nondet_firstM(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_batteries_Nondet_firstM___redArg(x_3, x_4, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_instMonadBacktrackUnitId__batteries___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
return x_1;
}
}
static lean_object* _init_lp_batteries_instMonadBacktrackUnitId__batteries() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_alloc_closure((void*)(lp_batteries_instMonadBacktrackUnitId__batteries___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Lint_Misc(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_MLList_Basic(uint8_t builtin);
lean_object* initialize_Lean_Util_MonadBacktrack(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Control_Nondet_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Lint_Misc(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_MLList_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Util_MonadBacktrack(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Nondet_toList_x27___redArg___closed__0 = _init_lp_batteries_Nondet_toList_x27___redArg___closed__0();
lean_mark_persistent(lp_batteries_Nondet_toList_x27___redArg___closed__0);
lp_batteries_instMonadBacktrackUnitId__batteries = _init_lp_batteries_instMonadBacktrackUnitId__batteries();
lean_mark_persistent(lp_batteries_instMonadBacktrackUnitId__batteries);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
