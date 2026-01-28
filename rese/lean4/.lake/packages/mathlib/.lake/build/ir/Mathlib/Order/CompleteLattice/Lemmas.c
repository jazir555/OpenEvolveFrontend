// Lean compiler output
// Module: Mathlib.Order.CompleteLattice.Lemmas
// Imports: public import Init public import Mathlib.Data.Bool.Set public import Mathlib.Data.Nat.Set public import Mathlib.Order.CompleteLattice.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet___redArg(lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(lean_object*);
lean_object* lp_mathlib_PUnit_instLinearOrder___lam__2___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_decidableLTOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_PUnit_instLinearOrder___lam__4___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___lam__0(lean_object*);
lean_object* lp_mathlib_decidableEqOfDecidableLE___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___lam__0(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_PUnit_instBiheytingAlgebra;
static lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder;
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___lam__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet(lean_object*, lean_object*);
lean_object* lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(lean_object*);
static lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__2;
LEAN_EXPORT uint8_t lp_mathlib_PUnit_instCompleteLinearOrder___lam__1(uint8_t, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_infSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_infSet___redArg(lean_object*);
static lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___closed__1;
extern lean_object* lp_mathlib_PUnit_instBooleanAlgebra;
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, lean_box(0));
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_supSet(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_supSet___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_infSet___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_infSet(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_infSet___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_2(x_4, x_2, x_3);
return x_5;
}
}
static lean_object* _init_lp_mathlib_ULift_instCompleteLattice___redArg___closed__0() {
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
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 3);
lean_inc_ref(x_3);
x_4 = !lean_is_exclusive(x_2);
if (x_4 == 0)
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; uint8_t x_10; 
x_5 = lean_ctor_get(x_2, 0);
x_6 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_1);
x_7 = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(x_1);
x_8 = lean_ctor_get(x_7, 1);
lean_inc(x_8);
lean_dec_ref(x_7);
x_9 = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(x_1);
x_10 = !lean_is_exclusive(x_9);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_11 = lean_ctor_get(x_9, 1);
x_12 = lean_ctor_get(x_9, 0);
lean_dec(x_12);
x_13 = !lean_is_exclusive(x_3);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_14 = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_14, 0, x_6);
x_15 = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__1), 3, 1);
lean_closure_set(x_15, 0, x_5);
x_16 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_16, 0, x_8);
x_17 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_17, 0, x_11);
x_18 = lp_mathlib_ULift_instCompleteLattice___redArg___closed__0;
lean_ctor_set(x_9, 1, x_15);
lean_ctor_set(x_9, 0, x_18);
lean_ctor_set(x_2, 1, x_14);
lean_ctor_set(x_2, 0, x_9);
x_19 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_19, 0, x_2);
lean_ctor_set(x_19, 1, x_16);
lean_ctor_set(x_19, 2, x_17);
lean_ctor_set(x_19, 3, x_3);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_20 = lean_ctor_get(x_3, 0);
x_21 = lean_ctor_get(x_3, 1);
lean_inc(x_21);
lean_inc(x_20);
lean_dec(x_3);
x_22 = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_22, 0, x_6);
x_23 = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__1), 3, 1);
lean_closure_set(x_23, 0, x_5);
x_24 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_24, 0, x_8);
x_25 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_25, 0, x_11);
x_26 = lp_mathlib_ULift_instCompleteLattice___redArg___closed__0;
lean_ctor_set(x_9, 1, x_23);
lean_ctor_set(x_9, 0, x_26);
lean_ctor_set(x_2, 1, x_22);
lean_ctor_set(x_2, 0, x_9);
x_27 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_27, 0, x_20);
lean_ctor_set(x_27, 1, x_21);
x_28 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_28, 0, x_2);
lean_ctor_set(x_28, 1, x_24);
lean_ctor_set(x_28, 2, x_25);
lean_ctor_set(x_28, 3, x_27);
return x_28;
}
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; lean_object* x_37; lean_object* x_38; lean_object* x_39; lean_object* x_40; 
x_29 = lean_ctor_get(x_9, 1);
lean_inc(x_29);
lean_dec(x_9);
x_30 = lean_ctor_get(x_3, 0);
lean_inc(x_30);
x_31 = lean_ctor_get(x_3, 1);
lean_inc(x_31);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 x_32 = x_3;
} else {
 lean_dec_ref(x_3);
 x_32 = lean_box(0);
}
x_33 = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_33, 0, x_6);
x_34 = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__1), 3, 1);
lean_closure_set(x_34, 0, x_5);
x_35 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_35, 0, x_8);
x_36 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_36, 0, x_29);
x_37 = lp_mathlib_ULift_instCompleteLattice___redArg___closed__0;
x_38 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_38, 0, x_37);
lean_ctor_set(x_38, 1, x_34);
lean_ctor_set(x_2, 1, x_33);
lean_ctor_set(x_2, 0, x_38);
if (lean_is_scalar(x_32)) {
 x_39 = lean_alloc_ctor(0, 2, 0);
} else {
 x_39 = x_32;
}
lean_ctor_set(x_39, 0, x_30);
lean_ctor_set(x_39, 1, x_31);
x_40 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_40, 0, x_2);
lean_ctor_set(x_40, 1, x_35);
lean_ctor_set(x_40, 2, x_36);
lean_ctor_set(x_40, 3, x_39);
return x_40;
}
}
else
{
lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; lean_object* x_52; lean_object* x_53; lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; lean_object* x_59; 
x_41 = lean_ctor_get(x_2, 0);
x_42 = lean_ctor_get(x_2, 1);
lean_inc(x_42);
lean_inc(x_41);
lean_dec(x_2);
lean_inc_ref(x_1);
x_43 = lp_mathlib_CompleteLattice_toCompleteSemilatticeSup___redArg(x_1);
x_44 = lean_ctor_get(x_43, 1);
lean_inc(x_44);
lean_dec_ref(x_43);
x_45 = lp_mathlib_CompleteLattice_toCompleteSemilatticeInf___redArg(x_1);
x_46 = lean_ctor_get(x_45, 1);
lean_inc(x_46);
if (lean_is_exclusive(x_45)) {
 lean_ctor_release(x_45, 0);
 lean_ctor_release(x_45, 1);
 x_47 = x_45;
} else {
 lean_dec_ref(x_45);
 x_47 = lean_box(0);
}
x_48 = lean_ctor_get(x_3, 0);
lean_inc(x_48);
x_49 = lean_ctor_get(x_3, 1);
lean_inc(x_49);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 x_50 = x_3;
} else {
 lean_dec_ref(x_3);
 x_50 = lean_box(0);
}
x_51 = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__0), 3, 1);
lean_closure_set(x_51, 0, x_42);
x_52 = lean_alloc_closure((void*)(lp_mathlib_ULift_instCompleteLattice___redArg___lam__1), 3, 1);
lean_closure_set(x_52, 0, x_41);
x_53 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_53, 0, x_44);
x_54 = lean_alloc_closure((void*)(lp_mathlib_ULift_supSet___redArg___lam__0), 2, 1);
lean_closure_set(x_54, 0, x_46);
x_55 = lp_mathlib_ULift_instCompleteLattice___redArg___closed__0;
if (lean_is_scalar(x_47)) {
 x_56 = lean_alloc_ctor(0, 2, 0);
} else {
 x_56 = x_47;
}
lean_ctor_set(x_56, 0, x_55);
lean_ctor_set(x_56, 1, x_52);
x_57 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_57, 0, x_56);
lean_ctor_set(x_57, 1, x_51);
if (lean_is_scalar(x_50)) {
 x_58 = lean_alloc_ctor(0, 2, 0);
} else {
 x_58 = x_50;
}
lean_ctor_set(x_58, 0, x_48);
lean_ctor_set(x_58, 1, x_49);
x_59 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_59, 0, x_57);
lean_ctor_set(x_59, 1, x_53);
lean_ctor_set(x_59, 2, x_54);
lean_ctor_set(x_59, 3, x_58);
return x_59;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_ULift_instCompleteLattice(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ULift_instCompleteLattice___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_PUnit_instCompleteLinearOrder___lam__1(uint8_t x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_box(0);
return x_2;
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteLinearOrder___closed__0() {
_start:
{
uint8_t x_1; lean_object* x_2; lean_object* x_3; 
x_1 = 1;
x_2 = lean_box(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_PUnit_instLinearOrder___lam__2___boxed), 3, 1);
lean_closure_set(x_3, 0, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteLinearOrder___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_PUnit_instCompleteLinearOrder___closed__0;
x_2 = lean_alloc_closure((void*)(lp_mathlib_PUnit_instLinearOrder___lam__4___boxed), 3, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteLinearOrder___closed__2() {
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
LEAN_EXPORT lean_object* lp_mathlib_PUnit_instCompleteLinearOrder___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; uint8_t x_5; lean_object* x_6; 
x_4 = lean_unbox(x_1);
x_5 = lp_mathlib_PUnit_instCompleteLinearOrder___lam__1(x_4, x_2, x_3);
x_6 = lean_box(x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_PUnit_instCompleteLinearOrder() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; 
x_1 = lp_mathlib_PUnit_instBooleanAlgebra;
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_1, 2);
lean_inc(x_4);
x_5 = lean_ctor_get(x_1, 3);
lean_inc(x_5);
x_6 = lean_ctor_get(x_1, 4);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 5);
lean_inc(x_7);
x_8 = lp_mathlib_PUnit_instBiheytingAlgebra;
x_9 = lean_ctor_get(x_8, 2);
lean_inc(x_9);
x_10 = lean_alloc_closure((void*)(lp_mathlib_PUnit_instCompleteLinearOrder___lam__0), 1, 0);
x_11 = 1;
x_12 = lp_mathlib_PUnit_instCompleteLinearOrder___closed__1;
x_13 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_13, 0, x_6);
lean_ctor_set(x_13, 1, x_7);
lean_inc_ref(x_10);
x_14 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_14, 0, x_2);
lean_ctor_set(x_14, 1, x_10);
lean_ctor_set(x_14, 2, x_10);
lean_ctor_set(x_14, 3, x_13);
x_15 = lean_box(x_11);
x_16 = lean_alloc_closure((void*)(lp_mathlib_PUnit_instCompleteLinearOrder___lam__1___boxed), 3, 1);
lean_closure_set(x_16, 0, x_15);
x_17 = lp_mathlib_PUnit_instCompleteLinearOrder___closed__2;
lean_inc_ref(x_16);
x_18 = lean_alloc_closure((void*)(lp_mathlib_decidableEqOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_18, 0, lean_box(0));
lean_closure_set(x_18, 1, x_17);
lean_closure_set(x_18, 2, x_16);
lean_inc_ref(x_16);
x_19 = lean_alloc_closure((void*)(lp_mathlib_decidableLTOfDecidableLE___boxed), 5, 3);
lean_closure_set(x_19, 0, lean_box(0));
lean_closure_set(x_19, 1, x_17);
lean_closure_set(x_19, 2, x_16);
x_20 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_20, 0, x_14);
lean_ctor_set(x_20, 1, x_5);
lean_ctor_set(x_20, 2, x_3);
lean_ctor_set(x_20, 3, x_4);
lean_ctor_set(x_20, 4, x_9);
lean_ctor_set(x_20, 5, x_12);
lean_ctor_set(x_20, 6, x_16);
lean_ctor_set(x_20, 7, x_18);
lean_ctor_set(x_20, 8, x_19);
return x_20;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Bool_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Order_CompleteLattice_Lemmas(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Bool_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteLattice_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ULift_instCompleteLattice___redArg___closed__0 = _init_lp_mathlib_ULift_instCompleteLattice___redArg___closed__0();
lean_mark_persistent(lp_mathlib_ULift_instCompleteLattice___redArg___closed__0);
lp_mathlib_PUnit_instCompleteLinearOrder___closed__0 = _init_lp_mathlib_PUnit_instCompleteLinearOrder___closed__0();
lean_mark_persistent(lp_mathlib_PUnit_instCompleteLinearOrder___closed__0);
lp_mathlib_PUnit_instCompleteLinearOrder___closed__1 = _init_lp_mathlib_PUnit_instCompleteLinearOrder___closed__1();
lean_mark_persistent(lp_mathlib_PUnit_instCompleteLinearOrder___closed__1);
lp_mathlib_PUnit_instCompleteLinearOrder___closed__2 = _init_lp_mathlib_PUnit_instCompleteLinearOrder___closed__2();
lean_mark_persistent(lp_mathlib_PUnit_instCompleteLinearOrder___closed__2);
lp_mathlib_PUnit_instCompleteLinearOrder = _init_lp_mathlib_PUnit_instCompleteLinearOrder();
lean_mark_persistent(lp_mathlib_PUnit_instCompleteLinearOrder);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
