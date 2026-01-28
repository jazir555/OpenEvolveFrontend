// Lean compiler output
// Module: Mathlib.Order.CompleteLattice.Basic
// Imports: public import Init public import Mathlib.Data.Set.NAry public import Mathlib.Data.ULift public import Mathlib.Order.CompleteLattice.Defs public import Mathlib.Order.Hom.Set
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
lean_object* lp_mathlib_Pi_instBoundedOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___boxed(lean_object**);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prop_instCompleteLattice;
extern lean_object* lp_mathlib_Prop_instBoundedOrder;
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__1(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_infSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_infSet(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Prop_instCompleteLattice___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteLattice___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instBoundedOrder___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_Pi_instLattice___redArg(lean_object*);
extern lean_object* lp_mathlib_Prop_instDistribLattice;
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__2(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_infSet___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_infSet(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Prod_instLattice___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteLattice(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
static lean_object* lp_mathlib_Function_Injective_completeLattice___closed__0;
static lean_object* _init_lp_mathlib_Prop_instCompleteLattice___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Prop_instBoundedOrder;
x_2 = lp_mathlib_Prop_instDistribLattice;
x_3 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, lean_box(0));
lean_ctor_set(x_3, 2, lean_box(0));
lean_ctor_set(x_3, 3, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Prop_instCompleteLattice() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Prop_instCompleteLattice___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_3, lean_box(0));
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_supSet(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_supSet___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_infSet___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_infSet(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_infSet___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 3);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_apply_1(x_1, x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(x_4);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_6, lean_box(0));
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg___lam__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(x_4);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_apply_1(x_6, lean_box(0));
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteLattice___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
lean_inc_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteLattice___redArg___lam__2), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Pi_instCompleteLattice___redArg___lam__3), 3, 1);
lean_closure_set(x_5, 0, x_1);
x_6 = lp_mathlib_Pi_instBoundedOrder___redArg(x_2);
x_7 = lp_mathlib_Pi_instLattice___redArg(x_3);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_4);
x_9 = lean_alloc_closure((void*)(lp_mathlib_Pi_supSet___redArg___lam__0), 3, 1);
lean_closure_set(x_9, 0, x_5);
x_10 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_10, 0, x_7);
lean_ctor_set(x_10, 1, x_8);
lean_ctor_set(x_10, 2, x_9);
lean_ctor_set(x_10, 3, x_6);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_instCompleteLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_instCompleteLattice___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_apply_1(x_1, lean_box(0));
x_5 = lean_apply_1(x_2, lean_box(0));
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_supSet(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_supSet___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_infSet___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_infSet(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_infSet___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteLattice___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
lean_inc_ref(x_1);
x_3 = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(x_1);
lean_inc_ref(x_2);
x_4 = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(x_2);
x_5 = lean_ctor_get(x_1, 0);
x_6 = lean_ctor_get(x_1, 3);
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_8);
lean_inc_ref(x_6);
x_9 = lp_mathlib_Prod_instBoundedOrder___redArg(x_6, x_8);
lean_inc_ref(x_7);
lean_inc_ref(x_5);
x_10 = lp_mathlib_Prod_instLattice___redArg(x_5, x_7);
x_11 = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(x_1);
x_12 = lean_ctor_get(x_11, 1);
lean_inc(x_12);
lean_dec_ref(x_11);
x_13 = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(x_2);
x_14 = lean_ctor_get(x_13, 1);
lean_inc(x_14);
lean_dec_ref(x_13);
x_15 = lean_ctor_get(x_3, 1);
lean_inc(x_15);
lean_dec_ref(x_3);
x_16 = lean_ctor_get(x_4, 1);
lean_inc(x_16);
lean_dec_ref(x_4);
x_17 = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(x_17, 0, x_12);
lean_closure_set(x_17, 1, x_14);
x_18 = lean_alloc_closure((void*)(lp_mathlib_Prod_supSet___redArg___lam__0), 3, 2);
lean_closure_set(x_18, 0, x_15);
lean_closure_set(x_18, 1, x_16);
x_19 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_19, 0, x_10);
lean_ctor_set(x_19, 1, x_17);
lean_ctor_set(x_19, 2, x_18);
lean_ctor_set(x_19, 3, x_9);
return x_19;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Prod_instCompleteLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Prod_instCompleteLattice___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_Function_Injective_completeLattice___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11, lean_object* x_12, lean_object* x_13, lean_object* x_14, lean_object* x_15, lean_object* x_16, lean_object* x_17) {
_start:
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_18 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_completeLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_18, 0, x_3);
x_19 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_completeLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_19, 0, x_4);
x_20 = lp_mathlib_Function_Injective_completeLattice___closed__0;
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_18);
x_22 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_22, 0, x_21);
lean_ctor_set(x_22, 1, x_19);
x_23 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_23, 0, x_7);
lean_ctor_set(x_23, 1, x_8);
x_24 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_24, 0, x_22);
lean_ctor_set(x_24, 1, x_5);
lean_ctor_set(x_24, 2, x_6);
lean_ctor_set(x_24, 3, x_23);
return x_24;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_completeLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_7, 0, x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Function_Injective_completeLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_8, 0, x_2);
x_9 = lp_mathlib_Function_Injective_completeLattice___closed__0;
x_10 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_10, 0, x_9);
lean_ctor_set(x_10, 1, x_7);
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_8);
x_12 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_12, 0, x_5);
lean_ctor_set(x_12, 1, x_6);
x_13 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_13, 0, x_11);
lean_ctor_set(x_13, 1, x_3);
lean_ctor_set(x_13, 2, x_4);
lean_ctor_set(x_13, 3, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Function_Injective_completeLattice___boxed(lean_object** _args) {
lean_object* x_1 = _args[0];
lean_object* x_2 = _args[1];
lean_object* x_3 = _args[2];
lean_object* x_4 = _args[3];
lean_object* x_5 = _args[4];
lean_object* x_6 = _args[5];
lean_object* x_7 = _args[6];
lean_object* x_8 = _args[7];
lean_object* x_9 = _args[8];
lean_object* x_10 = _args[9];
lean_object* x_11 = _args[10];
lean_object* x_12 = _args[11];
lean_object* x_13 = _args[12];
lean_object* x_14 = _args[13];
lean_object* x_15 = _args[14];
lean_object* x_16 = _args[15];
lean_object* x_17 = _args[16];
_start:
{
lean_object* x_18; 
x_18 = lp_mathlib_Function_Injective_completeLattice(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17);
lean_dec(x_10);
lean_dec_ref(x_9);
return x_18;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_NAry(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ULift(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Hom_Set(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_NAry(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ULift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Hom_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Prop_instCompleteLattice___closed__0 = _init_lp_mathlib_Prop_instCompleteLattice___closed__0();
lean_mark_persistent(lp_mathlib_Prop_instCompleteLattice___closed__0);
lp_mathlib_Prop_instCompleteLattice = _init_lp_mathlib_Prop_instCompleteLattice();
lean_mark_persistent(lp_mathlib_Prop_instCompleteLattice);
lp_mathlib_Function_Injective_completeLattice___closed__0 = _init_lp_mathlib_Function_Injective_completeLattice___closed__0();
lean_mark_persistent(lp_mathlib_Function_Injective_completeLattice___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
