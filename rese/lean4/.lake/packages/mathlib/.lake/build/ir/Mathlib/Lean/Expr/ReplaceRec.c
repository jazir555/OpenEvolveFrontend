// Lean compiler output
// Module: Mathlib.Lean.Expr.ReplaceRec
// Imports: public import Init public import Lean.Expr public import Mathlib.Util.MemoFix
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
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_replaceRec(lean_object*, lean_object*);
uint8_t lean_usize_dec_eq(size_t, size_t);
lean_object* l_Lean_Expr_mdata___override(lean_object*, lean_object*);
lean_object* l_Lean_Expr_proj___override(lean_object*, lean_object*, lean_object*);
size_t lean_ptr_addr(lean_object*);
lean_object* l_Lean_Expr_forallE___override(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_replaceRec___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Expr_letE___override(lean_object*, lean_object*, lean_object*, lean_object*, uint8_t);
lean_object* l_Lean_Expr_app___override(lean_object*, lean_object*);
lean_object* lp_mathlib___private_Mathlib_Util_MemoFix_0__memoFixImpl___redArg(lean_object*, lean_object*);
uint8_t l_Lean_instBEqBinderInfo_beq(uint8_t, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_replaceRec_spec__0(lean_object*, lean_object*);
lean_object* l_Lean_Expr_lam___override(lean_object*, lean_object*, lean_object*, uint8_t);
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_replaceRec_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
switch (lean_obj_tag(x_2)) {
case 7:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; uint8_t x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; size_t x_14; size_t x_15; uint8_t x_16; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_ctor_get(x_2, 2);
x_6 = lean_ctor_get_uint8(x_2, sizeof(void*)*3 + 8);
lean_inc_ref(x_1);
lean_inc_ref(x_4);
x_7 = lean_apply_1(x_1, x_4);
lean_inc_ref(x_5);
x_8 = lean_apply_1(x_1, x_5);
x_14 = lean_ptr_addr(x_4);
x_15 = lean_ptr_addr(x_7);
x_16 = lean_usize_dec_eq(x_14, x_15);
if (x_16 == 0)
{
x_9 = x_16;
goto block_13;
}
else
{
size_t x_17; size_t x_18; uint8_t x_19; 
x_17 = lean_ptr_addr(x_5);
x_18 = lean_ptr_addr(x_8);
x_19 = lean_usize_dec_eq(x_17, x_18);
x_9 = x_19;
goto block_13;
}
block_13:
{
if (x_9 == 0)
{
lean_object* x_10; 
lean_inc(x_3);
lean_dec_ref(x_2);
x_10 = l_Lean_Expr_forallE___override(x_3, x_7, x_8, x_6);
return x_10;
}
else
{
uint8_t x_11; 
x_11 = l_Lean_instBEqBinderInfo_beq(x_6, x_6);
if (x_11 == 0)
{
lean_object* x_12; 
lean_inc(x_3);
lean_dec_ref(x_2);
x_12 = l_Lean_Expr_forallE___override(x_3, x_7, x_8, x_6);
return x_12;
}
else
{
lean_dec_ref(x_8);
lean_dec_ref(x_7);
return x_2;
}
}
}
}
case 6:
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; uint8_t x_23; lean_object* x_24; lean_object* x_25; uint8_t x_26; size_t x_31; size_t x_32; uint8_t x_33; 
x_20 = lean_ctor_get(x_2, 0);
x_21 = lean_ctor_get(x_2, 1);
x_22 = lean_ctor_get(x_2, 2);
x_23 = lean_ctor_get_uint8(x_2, sizeof(void*)*3 + 8);
lean_inc_ref(x_1);
lean_inc_ref(x_21);
x_24 = lean_apply_1(x_1, x_21);
lean_inc_ref(x_22);
x_25 = lean_apply_1(x_1, x_22);
x_31 = lean_ptr_addr(x_21);
x_32 = lean_ptr_addr(x_24);
x_33 = lean_usize_dec_eq(x_31, x_32);
if (x_33 == 0)
{
x_26 = x_33;
goto block_30;
}
else
{
size_t x_34; size_t x_35; uint8_t x_36; 
x_34 = lean_ptr_addr(x_22);
x_35 = lean_ptr_addr(x_25);
x_36 = lean_usize_dec_eq(x_34, x_35);
x_26 = x_36;
goto block_30;
}
block_30:
{
if (x_26 == 0)
{
lean_object* x_27; 
lean_inc(x_20);
lean_dec_ref(x_2);
x_27 = l_Lean_Expr_lam___override(x_20, x_24, x_25, x_23);
return x_27;
}
else
{
uint8_t x_28; 
x_28 = l_Lean_instBEqBinderInfo_beq(x_23, x_23);
if (x_28 == 0)
{
lean_object* x_29; 
lean_inc(x_20);
lean_dec_ref(x_2);
x_29 = l_Lean_Expr_lam___override(x_20, x_24, x_25, x_23);
return x_29;
}
else
{
lean_dec_ref(x_25);
lean_dec_ref(x_24);
return x_2;
}
}
}
}
case 10:
{
lean_object* x_37; lean_object* x_38; lean_object* x_39; size_t x_40; size_t x_41; uint8_t x_42; 
x_37 = lean_ctor_get(x_2, 0);
x_38 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_38);
x_39 = lean_apply_1(x_1, x_38);
x_40 = lean_ptr_addr(x_38);
x_41 = lean_ptr_addr(x_39);
x_42 = lean_usize_dec_eq(x_40, x_41);
if (x_42 == 0)
{
lean_object* x_43; 
lean_inc(x_37);
lean_dec_ref(x_2);
x_43 = l_Lean_Expr_mdata___override(x_37, x_39);
return x_43;
}
else
{
lean_dec_ref(x_39);
return x_2;
}
}
case 8:
{
lean_object* x_44; lean_object* x_45; lean_object* x_46; lean_object* x_47; uint8_t x_48; lean_object* x_49; lean_object* x_50; lean_object* x_51; uint8_t x_52; size_t x_59; size_t x_60; uint8_t x_61; 
x_44 = lean_ctor_get(x_2, 0);
x_45 = lean_ctor_get(x_2, 1);
x_46 = lean_ctor_get(x_2, 2);
x_47 = lean_ctor_get(x_2, 3);
x_48 = lean_ctor_get_uint8(x_2, sizeof(void*)*4 + 8);
lean_inc_ref(x_1);
lean_inc_ref(x_45);
x_49 = lean_apply_1(x_1, x_45);
lean_inc_ref(x_1);
lean_inc_ref(x_46);
x_50 = lean_apply_1(x_1, x_46);
lean_inc_ref(x_47);
x_51 = lean_apply_1(x_1, x_47);
x_59 = lean_ptr_addr(x_45);
x_60 = lean_ptr_addr(x_49);
x_61 = lean_usize_dec_eq(x_59, x_60);
if (x_61 == 0)
{
x_52 = x_61;
goto block_58;
}
else
{
size_t x_62; size_t x_63; uint8_t x_64; 
x_62 = lean_ptr_addr(x_46);
x_63 = lean_ptr_addr(x_50);
x_64 = lean_usize_dec_eq(x_62, x_63);
x_52 = x_64;
goto block_58;
}
block_58:
{
if (x_52 == 0)
{
lean_object* x_53; 
lean_inc(x_44);
lean_dec_ref(x_2);
x_53 = l_Lean_Expr_letE___override(x_44, x_49, x_50, x_51, x_48);
return x_53;
}
else
{
size_t x_54; size_t x_55; uint8_t x_56; 
x_54 = lean_ptr_addr(x_47);
x_55 = lean_ptr_addr(x_51);
x_56 = lean_usize_dec_eq(x_54, x_55);
if (x_56 == 0)
{
lean_object* x_57; 
lean_inc(x_44);
lean_dec_ref(x_2);
x_57 = l_Lean_Expr_letE___override(x_44, x_49, x_50, x_51, x_48);
return x_57;
}
else
{
lean_dec_ref(x_51);
lean_dec_ref(x_50);
lean_dec_ref(x_49);
return x_2;
}
}
}
}
case 5:
{
lean_object* x_65; lean_object* x_66; lean_object* x_67; lean_object* x_68; uint8_t x_69; size_t x_72; size_t x_73; uint8_t x_74; 
x_65 = lean_ctor_get(x_2, 0);
x_66 = lean_ctor_get(x_2, 1);
lean_inc_ref(x_1);
lean_inc_ref(x_65);
x_67 = lean_apply_1(x_1, x_65);
lean_inc_ref(x_66);
x_68 = lean_apply_1(x_1, x_66);
x_72 = lean_ptr_addr(x_65);
x_73 = lean_ptr_addr(x_67);
x_74 = lean_usize_dec_eq(x_72, x_73);
if (x_74 == 0)
{
x_69 = x_74;
goto block_71;
}
else
{
size_t x_75; size_t x_76; uint8_t x_77; 
x_75 = lean_ptr_addr(x_66);
x_76 = lean_ptr_addr(x_68);
x_77 = lean_usize_dec_eq(x_75, x_76);
x_69 = x_77;
goto block_71;
}
block_71:
{
if (x_69 == 0)
{
lean_object* x_70; 
lean_dec_ref(x_2);
x_70 = l_Lean_Expr_app___override(x_67, x_68);
return x_70;
}
else
{
lean_dec_ref(x_68);
lean_dec_ref(x_67);
return x_2;
}
}
}
case 11:
{
lean_object* x_78; lean_object* x_79; lean_object* x_80; lean_object* x_81; size_t x_82; size_t x_83; uint8_t x_84; 
x_78 = lean_ctor_get(x_2, 0);
x_79 = lean_ctor_get(x_2, 1);
x_80 = lean_ctor_get(x_2, 2);
lean_inc_ref(x_80);
x_81 = lean_apply_1(x_1, x_80);
x_82 = lean_ptr_addr(x_80);
x_83 = lean_ptr_addr(x_81);
x_84 = lean_usize_dec_eq(x_82, x_83);
if (x_84 == 0)
{
lean_object* x_85; 
lean_inc(x_79);
lean_inc(x_78);
lean_dec_ref(x_2);
x_85 = l_Lean_Expr_proj___override(x_78, x_79, x_81);
return x_85;
}
else
{
lean_dec_ref(x_81);
return x_2;
}
}
default: 
{
lean_dec_ref(x_1);
return x_2;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_replaceRec___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
lean_inc_ref(x_3);
lean_inc_ref(x_2);
x_4 = lean_apply_2(x_1, x_2, x_3);
if (lean_obj_tag(x_4) == 0)
{
lean_object* x_5; 
x_5 = lp_mathlib_Lean_Expr_traverseChildren___at___00Lean_Expr_replaceRec_spec__0(x_2, x_3);
return x_5;
}
else
{
lean_object* x_6; 
lean_dec_ref(x_3);
lean_dec_ref(x_2);
x_6 = lean_ctor_get(x_4, 0);
lean_inc(x_6);
lean_dec_ref(x_4);
return x_6;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_replaceRec(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Lean_Expr_replaceRec___lam__0), 3, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lp_mathlib___private_Mathlib_Util_MemoFix_0__memoFixImpl___redArg(x_3, x_2);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Expr(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Util_MemoFix(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Expr_ReplaceRec(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Expr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Util_MemoFix(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
