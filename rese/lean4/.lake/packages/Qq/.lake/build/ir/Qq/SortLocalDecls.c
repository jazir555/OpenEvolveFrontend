// Lean compiler output
// Module: Qq.SortLocalDecls
// Imports: public import Init public import Lean.Meta.Basic
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_sortLocalDecls(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
static lean_object* lp_Qq_Qq_sortLocalDecls___closed__1;
uint8_t lean_usize_dec_eq(size_t, size_t);
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitExpr(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(lean_object*, lean_object*);
uint8_t l_Lean_Expr_hasMVar(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitExpr___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_sortLocalDecls___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_of_nat(lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_take(lean_object*);
uint8_t l_Lean_Expr_isMVar(lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(lean_object*, size_t, size_t, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
lean_object* lean_st_mk_ref(lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_fvarId(lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(lean_object*, size_t, size_t, lean_object*);
lean_object* l_Lean_NameSet_insert(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* l_Lean_NameSet_empty;
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_lt(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_NameSet_contains(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitLocalDecl___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_LocalDecl_type(lean_object*);
static lean_object* lp_Qq_Qq_sortLocalDecls___closed__2;
lean_object* l_Lean_instantiateMVarsCore(lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
lean_object* lean_st_ref_set(lean_object*, lean_object*);
static lean_object* lp_Qq_Qq_sortLocalDecls___closed__0;
lean_object* l_Lean_LocalDecl_value_x3f(lean_object*, uint8_t);
lean_object* lean_array_get_size(lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_4; 
x_4 = l_Lean_Expr_hasMVar(x_1);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_5, 0, x_1);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; uint8_t x_12; 
x_6 = lean_st_ref_get(x_2);
x_7 = lean_ctor_get(x_6, 0);
lean_inc_ref(x_7);
lean_dec(x_6);
x_8 = l_Lean_instantiateMVarsCore(x_7, x_1);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
x_10 = lean_ctor_get(x_8, 1);
lean_inc(x_10);
lean_dec_ref(x_8);
x_11 = lean_st_ref_take(x_2);
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_13 = lean_ctor_get(x_11, 0);
lean_dec(x_13);
lean_ctor_set(x_11, 0, x_10);
x_14 = lean_st_ref_set(x_2, x_11);
x_15 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_15, 0, x_9);
return x_15;
}
else
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_16 = lean_ctor_get(x_11, 1);
x_17 = lean_ctor_get(x_11, 2);
x_18 = lean_ctor_get(x_11, 3);
x_19 = lean_ctor_get(x_11, 4);
lean_inc(x_19);
lean_inc(x_18);
lean_inc(x_17);
lean_inc(x_16);
lean_dec(x_11);
x_20 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_20, 0, x_10);
lean_ctor_set(x_20, 1, x_16);
lean_ctor_set(x_20, 2, x_17);
lean_ctor_set(x_20, 3, x_18);
lean_ctor_set(x_20, 4, x_19);
x_21 = lean_st_ref_set(x_2, x_20);
x_22 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_22, 0, x_9);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; 
x_9 = lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(x_1, x_5);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitExpr(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
switch (lean_obj_tag(x_1)) {
case 11:
{
lean_object* x_21; 
x_21 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_21);
lean_dec_ref(x_1);
x_1 = x_21;
goto _start;
}
case 7:
{
lean_object* x_23; lean_object* x_24; 
x_23 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_23);
x_24 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_24);
lean_dec_ref(x_1);
x_9 = x_23;
x_10 = x_24;
x_11 = x_2;
x_12 = x_3;
x_13 = x_4;
x_14 = x_5;
x_15 = x_6;
x_16 = x_7;
x_17 = lean_box(0);
goto block_20;
}
case 6:
{
lean_object* x_25; lean_object* x_26; 
x_25 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_25);
x_26 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_26);
lean_dec_ref(x_1);
x_9 = x_25;
x_10 = x_26;
x_11 = x_2;
x_12 = x_3;
x_13 = x_4;
x_14 = x_5;
x_15 = x_6;
x_16 = x_7;
x_17 = lean_box(0);
goto block_20;
}
case 8:
{
lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_27 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_27);
x_28 = lean_ctor_get(x_1, 2);
lean_inc_ref(x_28);
x_29 = lean_ctor_get(x_1, 3);
lean_inc_ref(x_29);
lean_dec_ref(x_1);
x_30 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_27, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_30) == 0)
{
lean_object* x_31; 
lean_dec_ref(x_30);
x_31 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_28, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_31) == 0)
{
lean_dec_ref(x_31);
x_1 = x_29;
goto _start;
}
else
{
lean_dec_ref(x_29);
return x_31;
}
}
else
{
lean_dec_ref(x_29);
lean_dec_ref(x_28);
return x_30;
}
}
case 5:
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_33 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_33);
x_34 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_34);
lean_dec_ref(x_1);
x_35 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_33, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_35) == 0)
{
lean_dec_ref(x_35);
x_1 = x_34;
goto _start;
}
else
{
lean_dec_ref(x_34);
return x_35;
}
}
case 10:
{
lean_object* x_37; 
x_37 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_37);
lean_dec_ref(x_1);
x_1 = x_37;
goto _start;
}
case 2:
{
lean_object* x_39; 
x_39 = lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(x_1, x_5);
if (lean_obj_tag(x_39) == 0)
{
uint8_t x_40; 
x_40 = !lean_is_exclusive(x_39);
if (x_40 == 0)
{
lean_object* x_41; uint8_t x_42; 
x_41 = lean_ctor_get(x_39, 0);
x_42 = l_Lean_Expr_isMVar(x_41);
if (x_42 == 0)
{
lean_free_object(x_39);
x_1 = x_41;
goto _start;
}
else
{
lean_object* x_44; 
lean_dec(x_41);
x_44 = lean_box(0);
lean_ctor_set(x_39, 0, x_44);
return x_39;
}
}
else
{
lean_object* x_45; uint8_t x_46; 
x_45 = lean_ctor_get(x_39, 0);
lean_inc(x_45);
lean_dec(x_39);
x_46 = l_Lean_Expr_isMVar(x_45);
if (x_46 == 0)
{
x_1 = x_45;
goto _start;
}
else
{
lean_object* x_48; lean_object* x_49; 
lean_dec(x_45);
x_48 = lean_box(0);
x_49 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_49, 0, x_48);
return x_49;
}
}
}
else
{
uint8_t x_50; 
x_50 = !lean_is_exclusive(x_39);
if (x_50 == 0)
{
return x_39;
}
else
{
lean_object* x_51; lean_object* x_52; 
x_51 = lean_ctor_get(x_39, 0);
lean_inc(x_51);
lean_dec(x_39);
x_52 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_52, 0, x_51);
return x_52;
}
}
}
case 1:
{
lean_object* x_53; lean_object* x_54; 
x_53 = lean_ctor_get(x_1, 0);
lean_inc(x_53);
lean_dec_ref(x_1);
x_54 = l_Std_DTreeMap_Internal_Impl_Const_get_x3f___at___00Lean_NameMap_find_x3f_spec__0___redArg(x_2, x_53);
lean_dec(x_53);
if (lean_obj_tag(x_54) == 1)
{
lean_object* x_55; lean_object* x_56; 
x_55 = lean_ctor_get(x_54, 0);
lean_inc(x_55);
lean_dec_ref(x_54);
x_56 = lp_Qq_Qq_SortLocalDecls_visitLocalDecl(x_55, x_2, x_3, x_4, x_5, x_6, x_7);
return x_56;
}
else
{
lean_object* x_57; lean_object* x_58; 
lean_dec(x_54);
x_57 = lean_box(0);
x_58 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_58, 0, x_57);
return x_58;
}
}
default: 
{
lean_object* x_59; lean_object* x_60; 
lean_dec_ref(x_1);
x_59 = lean_box(0);
x_60 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_60, 0, x_59);
return x_60;
}
}
block_20:
{
lean_object* x_18; 
x_18 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_9, x_11, x_12, x_13, x_14, x_15, x_16);
if (lean_obj_tag(x_18) == 0)
{
lean_dec_ref(x_18);
x_1 = x_10;
x_2 = x_11;
x_3 = x_12;
x_4 = x_13;
x_5 = x_14;
x_6 = x_15;
x_7 = x_16;
goto _start;
}
else
{
lean_dec_ref(x_10);
return x_18;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitLocalDecl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_9; lean_object* x_10; lean_object* x_26; lean_object* x_27; lean_object* x_28; uint8_t x_29; 
x_26 = lean_st_ref_get(x_3);
x_27 = lean_ctor_get(x_26, 0);
lean_inc(x_27);
lean_dec(x_26);
x_28 = l_Lean_LocalDecl_fvarId(x_1);
x_29 = l_Lean_NameSet_contains(x_27, x_28);
lean_dec(x_27);
if (x_29 == 0)
{
lean_object* x_30; uint8_t x_31; 
x_30 = lean_st_ref_take(x_3);
x_31 = !lean_is_exclusive(x_30);
if (x_31 == 0)
{
lean_object* x_32; lean_object* x_33; lean_object* x_34; lean_object* x_35; lean_object* x_36; 
x_32 = lean_ctor_get(x_30, 0);
x_33 = l_Lean_NameSet_insert(x_32, x_28);
lean_ctor_set(x_30, 0, x_33);
x_34 = lean_st_ref_set(x_3, x_30);
x_35 = l_Lean_LocalDecl_type(x_1);
x_36 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_35, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_36) == 0)
{
lean_object* x_37; 
lean_dec_ref(x_36);
x_37 = l_Lean_LocalDecl_value_x3f(x_1, x_29);
if (lean_obj_tag(x_37) == 1)
{
lean_object* x_38; lean_object* x_39; 
x_38 = lean_ctor_get(x_37, 0);
lean_inc(x_38);
lean_dec_ref(x_37);
x_39 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_38, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_39) == 0)
{
lean_dec_ref(x_39);
x_9 = x_3;
x_10 = lean_box(0);
goto block_25;
}
else
{
lean_dec_ref(x_1);
return x_39;
}
}
else
{
lean_dec(x_37);
x_9 = x_3;
x_10 = lean_box(0);
goto block_25;
}
}
else
{
lean_dec_ref(x_1);
return x_36;
}
}
else
{
lean_object* x_40; lean_object* x_41; lean_object* x_42; lean_object* x_43; lean_object* x_44; lean_object* x_45; lean_object* x_46; 
x_40 = lean_ctor_get(x_30, 0);
x_41 = lean_ctor_get(x_30, 1);
lean_inc(x_41);
lean_inc(x_40);
lean_dec(x_30);
x_42 = l_Lean_NameSet_insert(x_40, x_28);
x_43 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_43, 0, x_42);
lean_ctor_set(x_43, 1, x_41);
x_44 = lean_st_ref_set(x_3, x_43);
x_45 = l_Lean_LocalDecl_type(x_1);
x_46 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_45, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_46) == 0)
{
lean_object* x_47; 
lean_dec_ref(x_46);
x_47 = l_Lean_LocalDecl_value_x3f(x_1, x_29);
if (lean_obj_tag(x_47) == 1)
{
lean_object* x_48; lean_object* x_49; 
x_48 = lean_ctor_get(x_47, 0);
lean_inc(x_48);
lean_dec_ref(x_47);
x_49 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_48, x_2, x_3, x_4, x_5, x_6, x_7);
if (lean_obj_tag(x_49) == 0)
{
lean_dec_ref(x_49);
x_9 = x_3;
x_10 = lean_box(0);
goto block_25;
}
else
{
lean_dec_ref(x_1);
return x_49;
}
}
else
{
lean_dec(x_47);
x_9 = x_3;
x_10 = lean_box(0);
goto block_25;
}
}
else
{
lean_dec_ref(x_1);
return x_46;
}
}
}
else
{
lean_object* x_50; lean_object* x_51; 
lean_dec(x_28);
lean_dec_ref(x_1);
x_50 = lean_box(0);
x_51 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_51, 0, x_50);
return x_51;
}
block_25:
{
lean_object* x_11; uint8_t x_12; 
x_11 = lean_st_ref_take(x_9);
x_12 = !lean_is_exclusive(x_11);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_13 = lean_ctor_get(x_11, 1);
x_14 = lean_array_push(x_13, x_1);
lean_ctor_set(x_11, 1, x_14);
x_15 = lean_st_ref_set(x_9, x_11);
x_16 = lean_box(0);
x_17 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_17, 0, x_16);
return x_17;
}
else
{
lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; 
x_18 = lean_ctor_get(x_11, 0);
x_19 = lean_ctor_get(x_11, 1);
lean_inc(x_19);
lean_inc(x_18);
lean_dec(x_11);
x_20 = lean_array_push(x_19, x_1);
x_21 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_21, 0, x_18);
lean_ctor_set(x_21, 1, x_20);
x_22 = lean_st_ref_set(x_9, x_21);
x_23 = lean_box(0);
x_24 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_24, 0, x_23);
return x_24;
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_Qq_Lean_instantiateMVars___at___00Qq_SortLocalDecls_visitExpr_spec__0___redArg(x_1, x_2);
lean_dec(x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitLocalDecl___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_Qq_Qq_SortLocalDecls_visitLocalDecl(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_SortLocalDecls_visitExpr___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_Qq_Qq_SortLocalDecls_visitExpr(x_1, x_2, x_3, x_4, x_5, x_6, x_7);
lean_dec(x_7);
lean_dec_ref(x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec(x_2);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_usize_dec_eq(x_2, x_3);
if (x_5 == 0)
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; size_t x_9; size_t x_10; 
x_6 = lean_array_uget(x_1, x_2);
x_7 = l_Lean_LocalDecl_fvarId(x_6);
x_8 = l_Std_DTreeMap_Internal_Impl_insert___at___00Lean_NameMap_insert_spec__0___redArg(x_7, x_6, x_4);
x_9 = 1;
x_10 = lean_usize_add(x_2, x_9);
x_2 = x_10;
x_4 = x_8;
goto _start;
}
else
{
return x_4;
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
uint8_t x_12; 
x_12 = lean_usize_dec_eq(x_2, x_3);
if (x_12 == 0)
{
lean_object* x_13; lean_object* x_14; 
x_13 = lean_array_uget(x_1, x_2);
x_14 = lp_Qq_Qq_SortLocalDecls_visitLocalDecl(x_13, x_5, x_6, x_7, x_8, x_9, x_10);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; size_t x_16; size_t x_17; 
x_15 = lean_ctor_get(x_14, 0);
lean_inc(x_15);
lean_dec_ref(x_14);
x_16 = 1;
x_17 = lean_usize_add(x_2, x_16);
x_2 = x_17;
x_4 = x_15;
goto _start;
}
else
{
return x_14;
}
}
else
{
lean_object* x_19; 
x_19 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_19, 0, x_4);
return x_19;
}
}
}
static lean_object* _init_lp_Qq_Qq_sortLocalDecls___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_NameSet_empty;
return x_1;
}
}
static lean_object* _init_lp_Qq_Qq_sortLocalDecls___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_Qq_Qq_sortLocalDecls___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_Qq_Qq_sortLocalDecls___closed__1;
x_2 = lp_Qq_Qq_sortLocalDecls___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_sortLocalDecls(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_29; uint8_t x_30; 
x_14 = lean_unsigned_to_nat(0u);
x_15 = lean_array_get_size(x_1);
x_29 = lean_box(1);
x_30 = lean_nat_dec_lt(x_14, x_15);
if (x_30 == 0)
{
x_16 = x_29;
goto block_28;
}
else
{
uint8_t x_31; 
x_31 = lean_nat_dec_le(x_15, x_15);
if (x_31 == 0)
{
x_16 = x_29;
goto block_28;
}
else
{
size_t x_32; size_t x_33; lean_object* x_34; 
x_32 = 0;
x_33 = lean_usize_of_nat(x_15);
x_34 = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(x_1, x_32, x_33, x_29);
x_16 = x_34;
goto block_28;
}
}
block_13:
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_9 = lean_st_ref_get(x_7);
x_10 = lean_st_ref_get(x_7);
lean_dec(x_7);
lean_dec(x_10);
x_11 = lean_ctor_get(x_9, 1);
lean_inc_ref(x_11);
lean_dec(x_9);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
block_28:
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_17 = lp_Qq_Qq_sortLocalDecls___closed__2;
x_18 = lean_st_mk_ref(x_17);
x_19 = lean_nat_dec_lt(x_14, x_15);
if (x_19 == 0)
{
lean_dec(x_16);
x_7 = x_18;
x_8 = lean_box(0);
goto block_13;
}
else
{
uint8_t x_20; 
x_20 = lean_nat_dec_le(x_15, x_15);
if (x_20 == 0)
{
lean_dec(x_16);
x_7 = x_18;
x_8 = lean_box(0);
goto block_13;
}
else
{
lean_object* x_21; size_t x_22; size_t x_23; lean_object* x_24; 
x_21 = lean_box(0);
x_22 = 0;
x_23 = lean_usize_of_nat(x_15);
x_24 = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(x_1, x_22, x_23, x_21, x_16, x_18, x_2, x_3, x_4, x_5);
lean_dec(x_16);
if (lean_obj_tag(x_24) == 0)
{
lean_dec_ref(x_24);
x_7 = x_18;
x_8 = lean_box(0);
goto block_13;
}
else
{
uint8_t x_25; 
lean_dec(x_18);
x_25 = !lean_is_exclusive(x_24);
if (x_25 == 0)
{
return x_24;
}
else
{
lean_object* x_26; lean_object* x_27; 
x_26 = lean_ctor_get(x_24, 0);
lean_inc(x_26);
lean_dec(x_24);
x_27 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_27, 0, x_26);
return x_27;
}
}
}
}
}
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; lean_object* x_7; 
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__1(x_1, x_5, x_6, x_4);
lean_dec_ref(x_1);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10, lean_object* x_11) {
_start:
{
size_t x_12; size_t x_13; lean_object* x_14; 
x_12 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_13 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_14 = lp_Qq___private_Init_Data_Array_Basic_0__Array_foldlMUnsafe_fold___at___00Qq_sortLocalDecls_spec__0(x_1, x_12, x_13, x_4, x_5, x_6, x_7, x_8, x_9, x_10);
lean_dec(x_10);
lean_dec_ref(x_9);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec(x_5);
lean_dec_ref(x_1);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_Qq_Qq_sortLocalDecls___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_Qq_Qq_sortLocalDecls(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_Qq_Qq_SortLocalDecls(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_Qq_Qq_sortLocalDecls___closed__0 = _init_lp_Qq_Qq_sortLocalDecls___closed__0();
lean_mark_persistent(lp_Qq_Qq_sortLocalDecls___closed__0);
lp_Qq_Qq_sortLocalDecls___closed__1 = _init_lp_Qq_Qq_sortLocalDecls___closed__1();
lean_mark_persistent(lp_Qq_Qq_sortLocalDecls___closed__1);
lp_Qq_Qq_sortLocalDecls___closed__2 = _init_lp_Qq_Qq_sortLocalDecls___closed__2();
lean_mark_persistent(lp_Qq_Qq_sortLocalDecls___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
