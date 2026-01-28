// Lean compiler output
// Module: Mathlib.Order.ConditionallyCompleteLattice.Basic
// Imports: public import Init public import Mathlib.Data.Set.Lattice public import Mathlib.Order.ConditionallyCompleteLattice.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_SemilatticeSup_toMax___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SemilatticeInf_toMin___redArg(lean_object*);
lean_object* lp_mathlib_Pi_infSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_OrderDual_instLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder(lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__2(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Pi_supSet___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLattice(lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Lattice_toSemilatticeInf___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object*);
lean_object* lp_mathlib_OrderDual_instLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_3, 0);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_6);
x_7 = lean_ctor_get(x_1, 3);
lean_inc_ref(x_7);
x_8 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_8);
lean_dec_ref(x_1);
x_9 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_9);
x_10 = lp_mathlib_Lattice_toSemilatticeInf___redArg(x_3);
x_11 = lp_mathlib_SemilatticeInf_toMin___redArg(x_10);
x_12 = lp_mathlib_SemilatticeSup_toMax___redArg(x_4);
x_13 = lean_alloc_ctor(0, 7, 0);
lean_ctor_set(x_13, 0, x_9);
lean_ctor_set(x_13, 1, x_11);
lean_ctor_set(x_13, 2, x_12);
lean_ctor_set(x_13, 3, x_5);
lean_ctor_set(x_13, 4, x_6);
lean_ctor_set(x_13, 5, x_7);
lean_ctor_set(x_13, 6, x_8);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_3 = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(x_1);
x_4 = lean_ctor_get(x_3, 1);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(x_1);
x_6 = lean_ctor_get(x_5, 1);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_7, 0, x_2);
lean_ctor_set(x_7, 1, x_4);
lean_ctor_set(x_7, 2, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLattice_toConditionallyCompleteLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 5);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 6);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 7);
lean_inc_ref(x_5);
x_6 = lean_ctor_get(x_1, 8);
lean_inc_ref(x_6);
lean_dec_ref(x_1);
lean_inc_ref(x_2);
x_7 = lp_mathlib_CompleteLattice_toConditionallyCompleteLattice___redArg(x_2);
x_8 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_8, 0, x_7);
lean_ctor_set(x_8, 1, x_3);
lean_ctor_set(x_8, 2, x_4);
lean_ctor_set(x_8, 3, x_5);
lean_ctor_set(x_8, 4, x_6);
x_9 = lean_ctor_get(x_2, 3);
lean_inc_ref(x_9);
lean_dec_ref(x_2);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; 
x_11 = lean_ctor_get(x_9, 0);
lean_dec(x_11);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
else
{
lean_object* x_12; lean_object* x_13; 
x_12 = lean_ctor_get(x_9, 1);
lean_inc(x_12);
lean_dec(x_9);
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_8);
lean_ctor_set(x_13, 1, x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_CompleteLinearOrder_toConditionallyCompleteLinearOrderBot___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLattice___redArg(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = !lean_is_exclusive(x_1);
if (x_2 == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lp_mathlib_OrderDual_instLattice___redArg(x_3);
lean_ctor_set(x_1, 2, x_4);
lean_ctor_set(x_1, 1, x_5);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_7 = lean_ctor_get(x_1, 0);
x_8 = lean_ctor_get(x_1, 1);
x_9 = lean_ctor_get(x_1, 2);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_dec(x_1);
x_10 = lp_mathlib_OrderDual_instLattice___redArg(x_7);
x_11 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_11, 0, x_10);
lean_ctor_set(x_11, 1, x_9);
lean_ctor_set(x_11, 2, x_8);
return x_11;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderDual_instConditionallyCompleteLattice___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_2, x_3);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT uint8_t lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_apply_2(x_1, x_3, x_2);
x_5 = lean_unbox(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_3);
x_4 = lean_ctor_get(x_1, 3);
lean_inc_ref(x_4);
x_5 = lean_ctor_get(x_1, 4);
lean_inc_ref(x_5);
lean_inc_ref(x_2);
x_6 = lp_mathlib_OrderDual_instConditionallyCompleteLattice___redArg(x_2);
x_7 = lp_mathlib_ConditionallyCompleteLinearOrder_toLinearOrder___redArg(x_1);
x_8 = lp_mathlib_OrderDual_instLinearOrder___redArg(x_7);
x_9 = lean_ctor_get(x_8, 3);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_10, 0, x_3);
x_11 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__1___boxed), 3, 1);
lean_closure_set(x_11, 0, x_4);
x_12 = lean_alloc_closure((void*)(lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg___lam__2___boxed), 3, 1);
lean_closure_set(x_12, 0, x_5);
x_13 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_13, 0, x_6);
lean_ctor_set(x_13, 1, x_9);
lean_ctor_set(x_13, 2, x_10);
lean_ctor_set(x_13, 3, x_11);
lean_ctor_set(x_13, 4, x_12);
return x_13;
}
}
LEAN_EXPORT lean_object* lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_OrderDual_instConditionallyCompleteLinearOrder___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
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
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lean_ctor_get(x_4, 1);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_1(x_5, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_apply_1(x_1, x_2);
x_5 = lean_ctor_get(x_4, 2);
lean_inc(x_5);
lean_dec_ref(x_4);
x_6 = lean_apply_1(x_5, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__1), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Pi_conditionallyCompleteLattice___redArg___lam__2), 3, 1);
lean_closure_set(x_4, 0, x_1);
x_5 = lp_mathlib_Pi_instLattice___redArg(x_2);
x_6 = lp_mathlib_Pi_supSet___redArg(x_3);
x_7 = lp_mathlib_Pi_infSet___redArg(x_4);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Pi_conditionallyCompleteLattice(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Pi_conditionallyCompleteLattice___redArg(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_ConditionallyCompleteLattice_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
