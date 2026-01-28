// Lean compiler output
// Module: Aesop.Search.Queue
// Imports: public import Init public import Aesop.Options public import Aesop.Tracing public import Aesop.Tree public import Aesop.Search.Queue.Class public import Batteries.Data.BinomialHeap.Basic
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
double lp_aesop_Aesop_Goal_priority(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_queue(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
static double lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0;
lean_object* l_Array_reverse___redArg(lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(lean_object*, lean_object*);
uint8_t lean_float_decLt(double, double);
lean_object* lean_array_fget_borrowed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__2(lean_object*);
static lean_object* lp_aesop_Aesop_Options_queue___closed__1;
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue;
size_t lean_usize_of_nat(lean_object*);
static lean_object* lp_aesop_Aesop_Options_queue___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__0(lean_object*, lean_object*);
uint8_t lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_LIFOQueue_init___closed__0;
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_array_pop(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(lean_object*, lean_object*);
lean_object* l_Array_back_x3f___redArg(lean_object*);
static lean_object* lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0;
static lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_init;
static lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___closed__0;
extern lean_object* lp_aesop_Aesop_treeImpl;
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_addGoals(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___boxed(lean_object*, lean_object*);
double l_Float_ofScientific(lean_object*, uint8_t, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_addGoals___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_init;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_addGoals(lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_popGoal(lean_object*);
static lean_object* lp_aesop_Aesop_FIFOQueue_init___closed__0;
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_init;
size_t lean_usize_add(size_t, size_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_queue___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_addGoals___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__1___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue;
lean_object* lean_array_uget(lean_object*, size_t);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_popGoal(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__1___boxed(lean_object*, lean_object*);
uint8_t lp_aesop_Aesop_Percent_instOrd___lam__0(double, double);
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_popGoal(lean_object*);
LEAN_EXPORT uint8_t lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(lean_object*, lean_object*);
lean_object* lean_array_get_size(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_addGoals(lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__1(lean_object*);
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___boxed(lean_object*, lean_object*);
static lean_object* lp_aesop_Aesop_Options_queue___closed__2;
double lean_float_sub(double, double);
static double _init_lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0() {
_start:
{
lean_object* x_1; uint8_t x_2; lean_object* x_3; double x_4; 
x_1 = lean_unsigned_to_nat(5u);
x_2 = 1;
x_3 = lean_unsigned_to_nat(1u);
x_4 = l_Float_ofScientific(x_3, x_2, x_1);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(lean_object* x_1, lean_object* x_2) {
_start:
{
double x_3; lean_object* x_4; lean_object* x_5; double x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; uint8_t x_14; 
x_3 = lean_ctor_get_float(x_2, sizeof(void*)*3);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_2, 2);
x_6 = lean_ctor_get_float(x_1, sizeof(void*)*3);
x_7 = lean_ctor_get(x_1, 1);
x_8 = lean_ctor_get(x_1, 2);
x_14 = lp_aesop_Aesop_Percent_instOrd___lam__0(x_3, x_6);
if (x_14 == 0)
{
uint8_t x_15; 
x_15 = 1;
return x_15;
}
else
{
uint8_t x_16; 
x_16 = lean_float_decLt(x_3, x_6);
if (x_16 == 0)
{
double x_17; double x_18; uint8_t x_19; 
x_17 = lean_float_sub(x_3, x_6);
x_18 = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0;
x_19 = lean_float_decLt(x_17, x_18);
x_9 = x_19;
goto block_13;
}
else
{
double x_20; lean_object* x_21; lean_object* x_22; double x_23; uint8_t x_24; 
x_20 = lean_float_sub(x_6, x_3);
x_21 = lean_unsigned_to_nat(1u);
x_22 = lean_unsigned_to_nat(5u);
x_23 = l_Float_ofScientific(x_21, x_16, x_22);
x_24 = lean_float_decLt(x_20, x_23);
x_9 = x_24;
goto block_13;
}
}
block_13:
{
if (x_9 == 0)
{
return x_9;
}
else
{
uint8_t x_10; 
x_10 = lean_nat_dec_le(x_7, x_4);
if (x_10 == 0)
{
uint8_t x_11; 
x_11 = lean_nat_dec_eq(x_7, x_4);
if (x_11 == 0)
{
return x_11;
}
else
{
uint8_t x_12; 
x_12 = lean_nat_dec_le(x_8, x_5);
return x_12;
}
}
else
{
return x_10;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_treeImpl;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef(lean_object* x_1) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; double x_9; lean_object* x_10; 
x_3 = lean_st_ref_get(x_1);
x_4 = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___closed__0;
x_5 = lean_ctor_get(x_4, 1);
lean_inc_ref(x_5);
lean_inc(x_3);
x_6 = lean_apply_1(x_5, x_3);
x_7 = lean_ctor_get(x_6, 10);
lean_inc(x_7);
x_8 = lean_ctor_get(x_6, 11);
lean_inc(x_8);
lean_dec_ref(x_6);
x_9 = lp_aesop_Aesop_Goal_priority(x_3);
x_10 = lean_alloc_ctor(0, 3, 8);
lean_ctor_set(x_10, 0, x_1);
lean_ctor_set(x_10, 1, x_8);
lean_ctor_set(x_10, 2, x_7);
lean_ctor_set_float(x_10, sizeof(void*)*3, x_9);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef(x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_BestFirstQueue_init() {
_start:
{
lean_object* x_1; 
x_1 = lean_box(0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return x_2;
}
else
{
if (lean_obj_tag(x_2) == 0)
{
return x_1;
}
else
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; uint8_t x_27; 
x_3 = lean_ctor_get(x_1, 0);
x_4 = lean_ctor_get(x_1, 1);
x_5 = lean_ctor_get(x_1, 2);
x_6 = lean_ctor_get(x_1, 3);
x_7 = lean_ctor_get(x_2, 0);
x_8 = lean_ctor_get(x_2, 1);
x_9 = lean_ctor_get(x_2, 2);
x_10 = lean_ctor_get(x_2, 3);
x_27 = lean_nat_dec_lt(x_3, x_7);
if (x_27 == 0)
{
uint8_t x_28; 
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
x_28 = !lean_is_exclusive(x_2);
if (x_28 == 0)
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_29 = lean_ctor_get(x_2, 3);
lean_dec(x_29);
x_30 = lean_ctor_get(x_2, 2);
lean_dec(x_30);
x_31 = lean_ctor_get(x_2, 1);
lean_dec(x_31);
x_32 = lean_ctor_get(x_2, 0);
lean_dec(x_32);
x_33 = lean_nat_dec_lt(x_7, x_3);
if (x_33 == 0)
{
uint8_t x_34; 
lean_free_object(x_2);
lean_dec(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_dec_ref(x_1);
x_34 = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(x_4, x_8);
if (x_34 == 0)
{
lean_object* x_35; 
x_35 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_35, 0, x_4);
lean_ctor_set(x_35, 1, x_5);
lean_ctor_set(x_35, 2, x_9);
x_11 = x_8;
x_12 = x_35;
goto block_26;
}
else
{
lean_object* x_36; 
x_36 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_36, 0, x_8);
lean_ctor_set(x_36, 1, x_9);
lean_ctor_set(x_36, 2, x_5);
x_11 = x_4;
x_12 = x_36;
goto block_26;
}
}
else
{
lean_object* x_37; 
x_37 = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(x_1, x_10);
lean_ctor_set(x_2, 3, x_37);
return x_2;
}
}
else
{
uint8_t x_38; 
lean_dec(x_2);
x_38 = lean_nat_dec_lt(x_7, x_3);
if (x_38 == 0)
{
uint8_t x_39; 
lean_dec(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
lean_dec_ref(x_1);
x_39 = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le(x_4, x_8);
if (x_39 == 0)
{
lean_object* x_40; 
x_40 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_40, 0, x_4);
lean_ctor_set(x_40, 1, x_5);
lean_ctor_set(x_40, 2, x_9);
x_11 = x_8;
x_12 = x_40;
goto block_26;
}
else
{
lean_object* x_41; 
x_41 = lean_alloc_ctor(1, 3, 0);
lean_ctor_set(x_41, 0, x_8);
lean_ctor_set(x_41, 1, x_9);
lean_ctor_set(x_41, 2, x_5);
x_11 = x_4;
x_12 = x_41;
goto block_26;
}
}
else
{
lean_object* x_42; lean_object* x_43; 
x_42 = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(x_1, x_10);
x_43 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_43, 0, x_7);
lean_ctor_set(x_43, 1, x_8);
lean_ctor_set(x_43, 2, x_9);
lean_ctor_set(x_43, 3, x_42);
return x_43;
}
}
}
else
{
uint8_t x_44; 
lean_inc(x_6);
lean_inc(x_5);
lean_inc(x_4);
lean_inc(x_3);
x_44 = !lean_is_exclusive(x_1);
if (x_44 == 0)
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; lean_object* x_48; lean_object* x_49; 
x_45 = lean_ctor_get(x_1, 3);
lean_dec(x_45);
x_46 = lean_ctor_get(x_1, 2);
lean_dec(x_46);
x_47 = lean_ctor_get(x_1, 1);
lean_dec(x_47);
x_48 = lean_ctor_get(x_1, 0);
lean_dec(x_48);
x_49 = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(x_6, x_2);
lean_ctor_set(x_1, 3, x_49);
return x_1;
}
else
{
lean_object* x_50; lean_object* x_51; 
lean_dec(x_1);
x_50 = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(x_6, x_2);
x_51 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_51, 0, x_3);
lean_ctor_set(x_51, 1, x_4);
lean_ctor_set(x_51, 2, x_5);
lean_ctor_set(x_51, 3, x_50);
return x_51;
}
}
block_26:
{
lean_object* x_13; lean_object* x_14; uint8_t x_15; 
x_13 = lean_unsigned_to_nat(1u);
x_14 = lean_nat_add(x_3, x_13);
lean_dec(x_3);
x_15 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_6, x_14);
if (x_15 == 0)
{
uint8_t x_16; 
x_16 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_10, x_14);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; 
x_17 = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(x_6, x_10);
x_18 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_18, 0, x_14);
lean_ctor_set(x_18, 1, x_11);
lean_ctor_set(x_18, 2, x_12);
lean_ctor_set(x_18, 3, x_17);
return x_18;
}
else
{
lean_object* x_19; 
x_19 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_19, 0, x_14);
lean_ctor_set(x_19, 1, x_11);
lean_ctor_set(x_19, 2, x_12);
lean_ctor_set(x_19, 3, x_10);
x_1 = x_6;
x_2 = x_19;
goto _start;
}
}
else
{
uint8_t x_21; 
x_21 = lp_batteries_Batteries_BinomialHeap_Imp_instDecidableRankGT___redArg(x_10, x_14);
if (x_21 == 0)
{
lean_object* x_22; 
x_22 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_22, 0, x_14);
lean_ctor_set(x_22, 1, x_11);
lean_ctor_set(x_22, 2, x_12);
lean_ctor_set(x_22, 3, x_6);
x_1 = x_22;
x_2 = x_10;
goto _start;
}
else
{
lean_object* x_24; lean_object* x_25; 
x_24 = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(x_6, x_10);
x_25 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_25, 0, x_14);
lean_ctor_set(x_25, 1, x_11);
lean_ctor_set(x_25, 2, x_12);
lean_ctor_set(x_25, 3, x_24);
return x_25;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_6; 
x_6 = lean_usize_dec_eq(x_2, x_3);
if (x_6 == 0)
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; size_t x_14; size_t x_15; 
x_7 = lean_array_uget(x_1, x_2);
x_8 = lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef(x_7);
x_9 = lean_unsigned_to_nat(0u);
x_10 = lean_box(0);
x_11 = lean_box(0);
x_12 = lean_alloc_ctor(1, 4, 0);
lean_ctor_set(x_12, 0, x_9);
lean_ctor_set(x_12, 1, x_8);
lean_ctor_set(x_12, 2, x_10);
lean_ctor_set(x_12, 3, x_11);
x_13 = lp_aesop_Batteries_BinomialHeap_Imp_Heap_merge___at___00Aesop_BestFirstQueue_addGoals_spec__0(x_12, x_4);
x_14 = 1;
x_15 = lean_usize_add(x_2, x_14);
x_2 = x_15;
x_4 = x_13;
goto _start;
}
else
{
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_addGoals(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; lean_object* x_5; uint8_t x_6; 
x_4 = lean_unsigned_to_nat(0u);
x_5 = lean_array_get_size(x_2);
x_6 = lean_nat_dec_lt(x_4, x_5);
if (x_6 == 0)
{
return x_1;
}
else
{
uint8_t x_7; 
x_7 = lean_nat_dec_le(x_5, x_5);
if (x_7 == 0)
{
return x_1;
}
else
{
size_t x_8; size_t x_9; lean_object* x_10; 
x_8 = 0;
x_9 = lean_usize_of_nat(x_5);
x_10 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(x_2, x_8, x_9, x_1);
return x_10;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_addGoals___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_BestFirstQueue_addGoals(x_1, x_2);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
size_t x_6; size_t x_7; lean_object* x_8; 
x_6 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_7 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_8 = lp_aesop___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Aesop_BestFirstQueue_addGoals_spec__1(x_1, x_6, x_7, x_4);
lean_dec_ref(x_1);
return x_8;
}
}
static lean_object* _init_lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_BestFirstQueue_popGoal(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0;
lean_inc(x_1);
x_3 = lp_batteries_Batteries_BinomialHeap_Imp_Heap_deleteMin___redArg(x_2, x_1);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_box(0);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_1);
return x_5;
}
else
{
uint8_t x_6; 
lean_dec(x_1);
x_6 = !lean_is_exclusive(x_3);
if (x_6 == 0)
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_ctor_get(x_3, 0);
x_8 = !lean_is_exclusive(x_7);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; 
x_9 = lean_ctor_get(x_7, 0);
x_10 = lean_ctor_get(x_9, 0);
lean_inc(x_10);
lean_dec(x_9);
lean_ctor_set(x_3, 0, x_10);
lean_ctor_set(x_7, 0, x_3);
return x_7;
}
else
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; 
x_11 = lean_ctor_get(x_7, 0);
x_12 = lean_ctor_get(x_7, 1);
lean_inc(x_12);
lean_inc(x_11);
lean_dec(x_7);
x_13 = lean_ctor_get(x_11, 0);
lean_inc(x_13);
lean_dec(x_11);
lean_ctor_set(x_3, 0, x_13);
x_14 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_14, 0, x_3);
lean_ctor_set(x_14, 1, x_12);
return x_14;
}
}
else
{
lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
x_15 = lean_ctor_get(x_3, 0);
lean_inc(x_15);
lean_dec(x_3);
x_16 = lean_ctor_get(x_15, 0);
lean_inc(x_16);
x_17 = lean_ctor_get(x_15, 1);
lean_inc(x_17);
if (lean_is_exclusive(x_15)) {
 lean_ctor_release(x_15, 0);
 lean_ctor_release(x_15, 1);
 x_18 = x_15;
} else {
 lean_dec_ref(x_15);
 x_18 = lean_box(0);
}
x_19 = lean_ctor_get(x_16, 0);
lean_inc(x_19);
lean_dec(x_16);
x_20 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_20, 0, x_19);
if (lean_is_scalar(x_18)) {
 x_21 = lean_alloc_ctor(0, 2, 0);
} else {
 x_21 = x_18;
}
lean_ctor_set(x_21, 0, x_20);
lean_ctor_set(x_21, 1, x_17);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__1(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_BestFirstQueue_popGoal(x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instQueueBestFirstQueue___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_BestFirstQueue_addGoals___boxed), 3, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_instQueueBestFirstQueue___lam__0(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_instQueueBestFirstQueue___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_instQueueBestFirstQueue___lam__1(x_1);
lean_dec(x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_instQueueBestFirstQueue() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_instQueueBestFirstQueue___lam__0___boxed), 2, 0);
x_2 = lean_box(0);
x_3 = lean_alloc_closure((void*)(lp_aesop_Aesop_instQueueBestFirstQueue___lam__1___boxed), 2, 1);
lean_closure_set(x_3, 0, x_2);
x_4 = lp_aesop_Aesop_instQueueBestFirstQueue___closed__0;
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, x_4);
lean_ctor_set(x_5, 2, x_1);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_LIFOQueue_init___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_LIFOQueue_init() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_LIFOQueue_init___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_addGoals(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = l_Array_reverse___redArg(x_2);
x_4 = l_Array_append___redArg(x_1, x_3);
lean_dec_ref(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_popGoal(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_Array_back_x3f___redArg(x_1);
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_array_pop(x_1);
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_4);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__2(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_LIFOQueue_addGoals(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_LIFOQueue_popGoal(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_LIFOQueue_instQueue___lam__0(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_LIFOQueue_instQueue___lam__1(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_LIFOQueue_instQueue___lam__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_LIFOQueue_instQueue___lam__2(x_1);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_LIFOQueue_instQueue() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_LIFOQueue_instQueue___lam__0___boxed), 3, 0);
x_2 = lean_alloc_closure((void*)(lp_aesop_Aesop_LIFOQueue_instQueue___lam__1___boxed), 2, 0);
x_3 = lp_aesop_Aesop_LIFOQueue_init;
x_4 = lean_alloc_closure((void*)(lp_aesop_Aesop_LIFOQueue_instQueue___lam__2___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_1);
lean_ctor_set(x_5, 2, x_2);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_FIFOQueue_init___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lp_aesop_Aesop_LIFOQueue_init___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_FIFOQueue_init() {
_start:
{
lean_object* x_1; 
x_1 = lp_aesop_Aesop_FIFOQueue_init___closed__0;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_addGoals(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = l_Array_append___redArg(x_4, x_2);
lean_ctor_set(x_1, 0, x_5);
return x_1;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_6 = lean_ctor_get(x_1, 0);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_1);
x_8 = l_Array_append___redArg(x_6, x_2);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_8);
lean_ctor_set(x_9, 1, x_7);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_addGoals___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_FIFOQueue_addGoals(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_popGoal(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; uint8_t x_5; 
x_2 = lean_ctor_get(x_1, 0);
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_array_get_size(x_2);
x_5 = lean_nat_dec_lt(x_3, x_4);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_box(0);
x_7 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_7, 0, x_6);
lean_ctor_set(x_7, 1, x_1);
return x_7;
}
else
{
uint8_t x_8; 
lean_inc(x_3);
lean_inc_ref(x_2);
x_8 = !lean_is_exclusive(x_1);
if (x_8 == 0)
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_9 = lean_ctor_get(x_1, 1);
lean_dec(x_9);
x_10 = lean_ctor_get(x_1, 0);
lean_dec(x_10);
x_11 = lean_array_fget_borrowed(x_2, x_3);
lean_inc(x_11);
x_12 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_12, 0, x_11);
x_13 = lean_unsigned_to_nat(1u);
x_14 = lean_nat_add(x_3, x_13);
lean_dec(x_3);
lean_ctor_set(x_1, 1, x_14);
x_15 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_15, 0, x_12);
lean_ctor_set(x_15, 1, x_1);
return x_15;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; 
lean_dec(x_1);
x_16 = lean_array_fget_borrowed(x_2, x_3);
lean_inc(x_16);
x_17 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_17, 0, x_16);
x_18 = lean_unsigned_to_nat(1u);
x_19 = lean_nat_add(x_3, x_18);
lean_dec(x_3);
x_20 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_20, 0, x_2);
lean_ctor_set(x_20, 1, x_19);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_17);
lean_ctor_set(x_21, 1, x_20);
return x_21;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__2(lean_object* x_1) {
_start:
{
lean_inc_ref(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_FIFOQueue_addGoals(x_1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__1(lean_object* x_1) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_FIFOQueue_popGoal(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_FIFOQueue_instQueue___lam__0(x_1, x_2);
lean_dec_ref(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__1___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_FIFOQueue_instQueue___lam__1(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_FIFOQueue_instQueue___lam__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_FIFOQueue_instQueue___lam__2(x_1);
lean_dec_ref(x_1);
return x_3;
}
}
static lean_object* _init_lp_aesop_Aesop_FIFOQueue_instQueue() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lean_alloc_closure((void*)(lp_aesop_Aesop_FIFOQueue_instQueue___lam__0___boxed), 3, 0);
x_2 = lean_alloc_closure((void*)(lp_aesop_Aesop_FIFOQueue_instQueue___lam__1___boxed), 2, 0);
x_3 = lp_aesop_Aesop_FIFOQueue_init;
x_4 = lean_alloc_closure((void*)(lp_aesop_Aesop_FIFOQueue_instQueue___lam__2___boxed), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_5, 0, x_4);
lean_ctor_set(x_5, 1, x_1);
lean_ctor_set(x_5, 2, x_2);
return x_5;
}
}
static lean_object* _init_lp_aesop_Aesop_Options_queue___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_instQueueBestFirstQueue;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, lean_box(0));
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_Options_queue___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_LIFOQueue_instQueue;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, lean_box(0));
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_aesop_Aesop_Options_queue___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_aesop_Aesop_FIFOQueue_instQueue;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, lean_box(0));
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_queue(lean_object* x_1) {
_start:
{
uint8_t x_2; 
x_2 = lean_ctor_get_uint8(x_1, sizeof(void*)*6);
switch (x_2) {
case 0:
{
lean_object* x_3; 
x_3 = lp_aesop_Aesop_Options_queue___closed__0;
return x_3;
}
case 1:
{
lean_object* x_4; 
x_4 = lp_aesop_Aesop_Options_queue___closed__1;
return x_4;
}
default: 
{
lean_object* x_5; 
x_5 = lp_aesop_Aesop_Options_queue___closed__2;
return x_5;
}
}
}
}
LEAN_EXPORT lean_object* lp_aesop_Aesop_Options_queue___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_aesop_Aesop_Options_queue(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Options(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tracing(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Tree(uint8_t builtin);
lean_object* initialize_aesop_Aesop_Search_Queue_Class(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_BinomialHeap_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_aesop_Aesop_Search_Queue(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Options(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tracing(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Tree(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_aesop_Aesop_Search_Queue_Class(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_BinomialHeap_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0 = _init_lp_aesop_Aesop_BestFirstQueue_ActiveGoal_le___closed__0();
lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___closed__0 = _init_lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___closed__0();
lean_mark_persistent(lp_aesop_Aesop_BestFirstQueue_ActiveGoal_ofGoalRef___closed__0);
lp_aesop_Aesop_BestFirstQueue_init = _init_lp_aesop_Aesop_BestFirstQueue_init();
lean_mark_persistent(lp_aesop_Aesop_BestFirstQueue_init);
lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0 = _init_lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0();
lean_mark_persistent(lp_aesop_Aesop_BestFirstQueue_popGoal___closed__0);
lp_aesop_Aesop_instQueueBestFirstQueue___closed__0 = _init_lp_aesop_Aesop_instQueueBestFirstQueue___closed__0();
lean_mark_persistent(lp_aesop_Aesop_instQueueBestFirstQueue___closed__0);
lp_aesop_Aesop_instQueueBestFirstQueue = _init_lp_aesop_Aesop_instQueueBestFirstQueue();
lean_mark_persistent(lp_aesop_Aesop_instQueueBestFirstQueue);
lp_aesop_Aesop_LIFOQueue_init___closed__0 = _init_lp_aesop_Aesop_LIFOQueue_init___closed__0();
lean_mark_persistent(lp_aesop_Aesop_LIFOQueue_init___closed__0);
lp_aesop_Aesop_LIFOQueue_init = _init_lp_aesop_Aesop_LIFOQueue_init();
lean_mark_persistent(lp_aesop_Aesop_LIFOQueue_init);
lp_aesop_Aesop_LIFOQueue_instQueue = _init_lp_aesop_Aesop_LIFOQueue_instQueue();
lean_mark_persistent(lp_aesop_Aesop_LIFOQueue_instQueue);
lp_aesop_Aesop_FIFOQueue_init___closed__0 = _init_lp_aesop_Aesop_FIFOQueue_init___closed__0();
lean_mark_persistent(lp_aesop_Aesop_FIFOQueue_init___closed__0);
lp_aesop_Aesop_FIFOQueue_init = _init_lp_aesop_Aesop_FIFOQueue_init();
lean_mark_persistent(lp_aesop_Aesop_FIFOQueue_init);
lp_aesop_Aesop_FIFOQueue_instQueue = _init_lp_aesop_Aesop_FIFOQueue_instQueue();
lean_mark_persistent(lp_aesop_Aesop_FIFOQueue_instQueue);
lp_aesop_Aesop_Options_queue___closed__0 = _init_lp_aesop_Aesop_Options_queue___closed__0();
lean_mark_persistent(lp_aesop_Aesop_Options_queue___closed__0);
lp_aesop_Aesop_Options_queue___closed__1 = _init_lp_aesop_Aesop_Options_queue___closed__1();
lean_mark_persistent(lp_aesop_Aesop_Options_queue___closed__1);
lp_aesop_Aesop_Options_queue___closed__2 = _init_lp_aesop_Aesop_Options_queue___closed__2();
lean_mark_persistent(lp_aesop_Aesop_Options_queue___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
