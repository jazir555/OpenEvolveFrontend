// Lean compiler output
// Module: Mathlib.Tactic.Relation.Rfl
// Imports: public import Init public import Mathlib.Init public meta import Lean.Meta.Tactic.Rfl
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
static lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t l_Array_isEmpty___redArg(lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_instInhabited(lean_object*);
extern lean_object* l_Lean_Meta_Rfl_reflExt;
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
lean_object* l_Lean_Elab_Tactic_getMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__4;
lean_object* l_Lean_Elab_Tactic_replaceMainGoal___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lean_st_ref_get(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__7;
lean_object* l_Lean_Expr_constName_x3f(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Elab_Tactic_withMainContext___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0;
lean_object* l_Lean_ScopedEnvExtension_getState___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__5;
lean_object* l_Lean_Expr_getAppFn(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_MVarId_applyRfl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Meta_DiscrTree_getMatch___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
static lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3;
static lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__6;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_11; 
x_11 = l_Lean_Elab_Tactic_withMainContext___redArg(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; 
x_10 = l_Lean_Elab_Tactic_getMainGoal___redArg(x_2, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_10) == 0)
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_ctor_get(x_10, 0);
lean_inc(x_11);
lean_dec_ref(x_10);
lean_inc(x_8);
lean_inc_ref(x_7);
lean_inc(x_6);
lean_inc_ref(x_5);
x_12 = l_Lean_MVarId_applyRfl(x_11, x_5, x_6, x_7, x_8);
if (lean_obj_tag(x_12) == 0)
{
lean_object* x_13; lean_object* x_14; 
lean_dec_ref(x_12);
x_13 = lean_box(0);
x_14 = l_Lean_Elab_Tactic_replaceMainGoal___redArg(x_13, x_2, x_5, x_6, x_7, x_8);
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
if (lean_obj_tag(x_14) == 0)
{
uint8_t x_15; 
x_15 = !lean_is_exclusive(x_14);
if (x_15 == 0)
{
lean_object* x_16; lean_object* x_17; 
x_16 = lean_ctor_get(x_14, 0);
lean_dec(x_16);
x_17 = lean_box(0);
lean_ctor_set(x_14, 0, x_17);
return x_14;
}
else
{
lean_object* x_18; lean_object* x_19; 
lean_dec(x_14);
x_18 = lean_box(0);
x_19 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
else
{
return x_14;
}
}
else
{
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
return x_12;
}
}
else
{
uint8_t x_20; 
lean_dec(x_8);
lean_dec_ref(x_7);
lean_dec(x_6);
lean_dec_ref(x_5);
x_20 = !lean_is_exclusive(x_10);
if (x_20 == 0)
{
return x_10;
}
else
{
lean_object* x_21; lean_object* x_22; 
x_21 = lean_ctor_get(x_10, 0);
lean_inc(x_21);
lean_dec(x_10);
x_22 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_22, 0, x_21);
return x_22;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Mathlib_Tactic_rflTac___lam__0(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_4);
lean_dec_ref(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_10;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9, lean_object* x_10) {
_start:
{
lean_object* x_11; 
x_11 = lp_mathlib_Mathlib_Tactic_rflTac___lam__1(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9);
return x_11;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_rflTac___lam__0___boxed), 9, 0);
x_11 = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_rflTac___lam__1___boxed), 10, 1);
lean_closure_set(x_11, 0, x_10);
x_12 = l_Lean_Elab_Tactic_withMainContext___redArg(x_11, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_rflTac___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8, lean_object* x_9) {
_start:
{
lean_object* x_10; 
x_10 = lp_mathlib_Mathlib_Tactic_rflTac(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
return x_10;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Eq", 2, 2);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Iff", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("HEq", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__5() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__4;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Meta_Rfl_reflExt;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__7() {
_start:
{
lean_object* x_1; 
x_1 = l_Lean_Meta_DiscrTree_instInhabited(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_11 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1;
x_12 = lean_unsigned_to_nat(3u);
x_13 = l_Lean_Expr_isAppOfArity(x_1, x_11, x_12);
if (x_13 == 0)
{
lean_object* x_14; lean_object* x_15; uint8_t x_16; 
x_14 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3;
x_15 = lean_unsigned_to_nat(2u);
x_16 = l_Lean_Expr_isAppOfArity(x_1, x_14, x_15);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; uint8_t x_19; 
x_17 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__5;
x_18 = lean_unsigned_to_nat(4u);
x_19 = l_Lean_Expr_isAppOfArity(x_1, x_17, x_18);
if (x_19 == 0)
{
if (lean_obj_tag(x_1) == 5)
{
lean_object* x_20; 
x_20 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_20);
if (lean_obj_tag(x_20) == 5)
{
lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; lean_object* x_31; uint8_t x_32; 
x_21 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_21);
lean_dec_ref(x_1);
x_22 = lean_ctor_get(x_20, 0);
lean_inc_ref(x_22);
x_23 = lean_ctor_get(x_20, 1);
lean_inc_ref(x_23);
lean_dec_ref(x_20);
x_24 = lean_st_ref_get(x_5);
x_25 = lean_ctor_get(x_24, 0);
lean_inc_ref(x_25);
lean_dec(x_24);
x_26 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__6;
x_27 = lean_ctor_get(x_26, 1);
lean_inc_ref(x_27);
x_28 = lean_ctor_get(x_27, 0);
lean_inc_ref(x_28);
lean_dec_ref(x_27);
x_29 = lean_ctor_get(x_28, 2);
lean_inc(x_29);
lean_dec_ref(x_28);
x_30 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__7;
x_31 = l_Lean_ScopedEnvExtension_getState___redArg(x_30, x_26, x_25, x_29);
lean_dec(x_29);
x_32 = !lean_is_exclusive(x_26);
if (x_32 == 0)
{
lean_object* x_33; lean_object* x_34; lean_object* x_35; 
x_33 = lean_ctor_get(x_26, 1);
lean_dec(x_33);
x_34 = lean_ctor_get(x_26, 0);
lean_dec(x_34);
lean_inc_ref(x_22);
x_35 = l_Lean_Meta_DiscrTree_getMatch___redArg(x_31, x_22, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_35) == 0)
{
uint8_t x_36; 
x_36 = !lean_is_exclusive(x_35);
if (x_36 == 0)
{
lean_object* x_37; uint8_t x_38; 
x_37 = lean_ctor_get(x_35, 0);
x_38 = l_Array_isEmpty___redArg(x_37);
lean_dec(x_37);
if (x_38 == 0)
{
lean_object* x_39; lean_object* x_40; 
x_39 = l_Lean_Expr_getAppFn(x_22);
lean_dec_ref(x_22);
x_40 = l_Lean_Expr_constName_x3f(x_39);
lean_dec_ref(x_39);
if (lean_obj_tag(x_40) == 0)
{
lean_object* x_41; 
lean_free_object(x_26);
lean_dec_ref(x_23);
lean_dec_ref(x_21);
x_41 = lean_box(0);
lean_ctor_set(x_35, 0, x_41);
return x_35;
}
else
{
uint8_t x_42; 
x_42 = !lean_is_exclusive(x_40);
if (x_42 == 0)
{
lean_object* x_43; lean_object* x_44; 
x_43 = lean_ctor_get(x_40, 0);
lean_ctor_set(x_26, 1, x_21);
lean_ctor_set(x_26, 0, x_23);
x_44 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_44, 0, x_43);
lean_ctor_set(x_44, 1, x_26);
lean_ctor_set(x_40, 0, x_44);
lean_ctor_set(x_35, 0, x_40);
return x_35;
}
else
{
lean_object* x_45; lean_object* x_46; lean_object* x_47; 
x_45 = lean_ctor_get(x_40, 0);
lean_inc(x_45);
lean_dec(x_40);
lean_ctor_set(x_26, 1, x_21);
lean_ctor_set(x_26, 0, x_23);
x_46 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_46, 0, x_45);
lean_ctor_set(x_46, 1, x_26);
x_47 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_47, 0, x_46);
lean_ctor_set(x_35, 0, x_47);
return x_35;
}
}
}
else
{
lean_free_object(x_35);
lean_free_object(x_26);
lean_dec_ref(x_23);
lean_dec_ref(x_22);
lean_dec_ref(x_21);
x_7 = lean_box(0);
goto block_10;
}
}
else
{
lean_object* x_48; uint8_t x_49; 
x_48 = lean_ctor_get(x_35, 0);
lean_inc(x_48);
lean_dec(x_35);
x_49 = l_Array_isEmpty___redArg(x_48);
lean_dec(x_48);
if (x_49 == 0)
{
lean_object* x_50; lean_object* x_51; 
x_50 = l_Lean_Expr_getAppFn(x_22);
lean_dec_ref(x_22);
x_51 = l_Lean_Expr_constName_x3f(x_50);
lean_dec_ref(x_50);
if (lean_obj_tag(x_51) == 0)
{
lean_object* x_52; lean_object* x_53; 
lean_free_object(x_26);
lean_dec_ref(x_23);
lean_dec_ref(x_21);
x_52 = lean_box(0);
x_53 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_53, 0, x_52);
return x_53;
}
else
{
lean_object* x_54; lean_object* x_55; lean_object* x_56; lean_object* x_57; lean_object* x_58; 
x_54 = lean_ctor_get(x_51, 0);
lean_inc(x_54);
if (lean_is_exclusive(x_51)) {
 lean_ctor_release(x_51, 0);
 x_55 = x_51;
} else {
 lean_dec_ref(x_51);
 x_55 = lean_box(0);
}
lean_ctor_set(x_26, 1, x_21);
lean_ctor_set(x_26, 0, x_23);
x_56 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_56, 0, x_54);
lean_ctor_set(x_56, 1, x_26);
if (lean_is_scalar(x_55)) {
 x_57 = lean_alloc_ctor(1, 1, 0);
} else {
 x_57 = x_55;
}
lean_ctor_set(x_57, 0, x_56);
x_58 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_58, 0, x_57);
return x_58;
}
}
else
{
lean_free_object(x_26);
lean_dec_ref(x_23);
lean_dec_ref(x_22);
lean_dec_ref(x_21);
x_7 = lean_box(0);
goto block_10;
}
}
}
else
{
uint8_t x_59; 
lean_free_object(x_26);
lean_dec_ref(x_23);
lean_dec_ref(x_22);
lean_dec_ref(x_21);
x_59 = !lean_is_exclusive(x_35);
if (x_59 == 0)
{
return x_35;
}
else
{
lean_object* x_60; lean_object* x_61; 
x_60 = lean_ctor_get(x_35, 0);
lean_inc(x_60);
lean_dec(x_35);
x_61 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_61, 0, x_60);
return x_61;
}
}
}
else
{
lean_object* x_62; 
lean_dec(x_26);
lean_inc_ref(x_22);
x_62 = l_Lean_Meta_DiscrTree_getMatch___redArg(x_31, x_22, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_62) == 0)
{
lean_object* x_63; lean_object* x_64; uint8_t x_65; 
x_63 = lean_ctor_get(x_62, 0);
lean_inc(x_63);
if (lean_is_exclusive(x_62)) {
 lean_ctor_release(x_62, 0);
 x_64 = x_62;
} else {
 lean_dec_ref(x_62);
 x_64 = lean_box(0);
}
x_65 = l_Array_isEmpty___redArg(x_63);
lean_dec(x_63);
if (x_65 == 0)
{
lean_object* x_66; lean_object* x_67; 
x_66 = l_Lean_Expr_getAppFn(x_22);
lean_dec_ref(x_22);
x_67 = l_Lean_Expr_constName_x3f(x_66);
lean_dec_ref(x_66);
if (lean_obj_tag(x_67) == 0)
{
lean_object* x_68; lean_object* x_69; 
lean_dec_ref(x_23);
lean_dec_ref(x_21);
x_68 = lean_box(0);
if (lean_is_scalar(x_64)) {
 x_69 = lean_alloc_ctor(0, 1, 0);
} else {
 x_69 = x_64;
}
lean_ctor_set(x_69, 0, x_68);
return x_69;
}
else
{
lean_object* x_70; lean_object* x_71; lean_object* x_72; lean_object* x_73; lean_object* x_74; lean_object* x_75; 
x_70 = lean_ctor_get(x_67, 0);
lean_inc(x_70);
if (lean_is_exclusive(x_67)) {
 lean_ctor_release(x_67, 0);
 x_71 = x_67;
} else {
 lean_dec_ref(x_67);
 x_71 = lean_box(0);
}
x_72 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_72, 0, x_23);
lean_ctor_set(x_72, 1, x_21);
x_73 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_73, 0, x_70);
lean_ctor_set(x_73, 1, x_72);
if (lean_is_scalar(x_71)) {
 x_74 = lean_alloc_ctor(1, 1, 0);
} else {
 x_74 = x_71;
}
lean_ctor_set(x_74, 0, x_73);
if (lean_is_scalar(x_64)) {
 x_75 = lean_alloc_ctor(0, 1, 0);
} else {
 x_75 = x_64;
}
lean_ctor_set(x_75, 0, x_74);
return x_75;
}
}
else
{
lean_dec(x_64);
lean_dec_ref(x_23);
lean_dec_ref(x_22);
lean_dec_ref(x_21);
x_7 = lean_box(0);
goto block_10;
}
}
else
{
lean_object* x_76; lean_object* x_77; lean_object* x_78; 
lean_dec_ref(x_23);
lean_dec_ref(x_22);
lean_dec_ref(x_21);
x_76 = lean_ctor_get(x_62, 0);
lean_inc(x_76);
if (lean_is_exclusive(x_62)) {
 lean_ctor_release(x_62, 0);
 x_77 = x_62;
} else {
 lean_dec_ref(x_62);
 x_77 = lean_box(0);
}
if (lean_is_scalar(x_77)) {
 x_78 = lean_alloc_ctor(1, 1, 0);
} else {
 x_78 = x_77;
}
lean_ctor_set(x_78, 0, x_76);
return x_78;
}
}
}
else
{
lean_dec_ref(x_20);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_7 = lean_box(0);
goto block_10;
}
}
else
{
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_7 = lean_box(0);
goto block_10;
}
}
else
{
lean_object* x_79; lean_object* x_80; lean_object* x_81; lean_object* x_82; lean_object* x_83; lean_object* x_84; lean_object* x_85; lean_object* x_86; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_79 = l_Lean_Expr_appFn_x21(x_1);
x_80 = l_Lean_Expr_appFn_x21(x_79);
lean_dec_ref(x_79);
x_81 = l_Lean_Expr_appArg_x21(x_80);
lean_dec_ref(x_80);
x_82 = l_Lean_Expr_appArg_x21(x_1);
lean_dec_ref(x_1);
x_83 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_83, 0, x_81);
lean_ctor_set(x_83, 1, x_82);
x_84 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_84, 0, x_17);
lean_ctor_set(x_84, 1, x_83);
x_85 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_85, 0, x_84);
x_86 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_86, 0, x_85);
return x_86;
}
}
else
{
lean_object* x_87; lean_object* x_88; lean_object* x_89; lean_object* x_90; lean_object* x_91; lean_object* x_92; lean_object* x_93; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_87 = l_Lean_Expr_appFn_x21(x_1);
x_88 = l_Lean_Expr_appArg_x21(x_87);
lean_dec_ref(x_87);
x_89 = l_Lean_Expr_appArg_x21(x_1);
lean_dec_ref(x_1);
x_90 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_90, 0, x_88);
lean_ctor_set(x_90, 1, x_89);
x_91 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_91, 0, x_14);
lean_ctor_set(x_91, 1, x_90);
x_92 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_92, 0, x_91);
x_93 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_93, 0, x_92);
return x_93;
}
}
else
{
lean_object* x_94; lean_object* x_95; lean_object* x_96; lean_object* x_97; lean_object* x_98; lean_object* x_99; lean_object* x_100; 
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
x_94 = l_Lean_Expr_appFn_x21(x_1);
x_95 = l_Lean_Expr_appArg_x21(x_94);
lean_dec_ref(x_94);
x_96 = l_Lean_Expr_appArg_x21(x_1);
lean_dec_ref(x_1);
x_97 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_97, 0, x_95);
lean_ctor_set(x_97, 1, x_96);
x_98 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_98, 0, x_11);
lean_ctor_set(x_98, 1, x_97);
x_99 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_99, 0, x_98);
x_100 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_100, 0, x_99);
return x_100;
}
block_10:
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_box(0);
x_9 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_9, 0, x_8);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Lean_Expr_relSidesIfRefl_x3f(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_Meta_Tactic_Rfl(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Tactic_Relation_Rfl(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Meta_Tactic_Rfl(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0 = _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0();
lean_mark_persistent(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__0);
lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1 = _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1();
lean_mark_persistent(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__1);
lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2 = _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2();
lean_mark_persistent(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__2);
lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3 = _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3();
lean_mark_persistent(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__3);
lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__4 = _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__4();
lean_mark_persistent(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__4);
lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__5 = _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__5();
lean_mark_persistent(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__5);
lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__6 = _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__6();
lean_mark_persistent(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__6);
lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__7 = _init_lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__7();
lean_mark_persistent(lp_mathlib_Lean_Expr_relSidesIfRefl_x3f___closed__7);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
