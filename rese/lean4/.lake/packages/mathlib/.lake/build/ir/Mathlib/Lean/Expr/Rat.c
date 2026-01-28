// Lean compiler output
// Module: Mathlib.Lean.Expr.Rat
// Imports: public import Init public import Mathlib.Init public import Batteries.Tactic.Alias public import Lean.ToExpr
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
lean_object* l_Lean_Expr_const___override(lean_object*, lean_object*);
lean_object* l_Lean_mkNatLit(lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__9;
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__8;
lean_object* l_Lean_mkAppB(lean_object*, lean_object*, lean_object*);
uint8_t l_Lean_Expr_isAppOfArity(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__7;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_rat_x3f(lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__0;
uint8_t lean_int_dec_le(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instToExprRat__mathlib___lam__0(lean_object*);
lean_object* l_Lean_Level_ofNat(lean_object*);
lean_object* l_Lean_Expr_appArg_x21(lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___closed__1;
lean_object* lean_nat_to_int(lean_object*);
static lean_object* lp_mathlib_Lean_Expr_rat_x3f___closed__2;
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__2;
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__3;
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__10;
static lean_object* lp_mathlib_instToExprRat__mathlib___closed__3;
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__6;
lean_object* l_mkRat(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isExplicitNumber(lean_object*);
lean_object* l_Lean_Expr_appFn_x21(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instToExprRat__mathlib;
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__13;
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__11;
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isExplicitNumber___boxed(lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___closed__0;
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
lean_object* l_Lean_mkApp3(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_Name_mkStr2(lean_object*, lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__1;
static lean_object* lp_mathlib_instToExprRat__mathlib___closed__4;
static lean_object* lp_mathlib_Lean_Expr_rat_x3f___closed__1;
lean_object* l_Int_toNat(lean_object*);
lean_object* l_Lean_Expr_int_x3f(lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__12;
static lean_object* lp_mathlib_instToExprRat__mathlib___closed__2;
lean_object* l_Lean_instToExprInt_mkNat(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00Lean_Expr_rat_x3f_spec__0(lean_object*);
lean_object* l_Lean_Name_mkStr1(lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__4;
lean_object* l_Rat_ofInt(lean_object*);
lean_object* lean_int_neg(lean_object*);
static lean_object* lp_mathlib_instToExprRat__mathlib___lam__0___closed__5;
static lean_object* lp_mathlib_Lean_Expr_rat_x3f___closed__0;
lean_object* l_Lean_Expr_nat_x3f(lean_object*);
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Rat", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_instToExprRat__mathlib___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = l_Lean_Level_ofNat(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_instToExprRat__mathlib___closed__2;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instToExprRat__mathlib___closed__3;
x_2 = lp_mathlib_instToExprRat__mathlib___closed__1;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("mkRat", 5, 5);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__0;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__1;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("neg", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Neg", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__6() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__5;
x_2 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__4;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instToExprRat__mathlib___closed__3;
x_2 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__6;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__8() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Int", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__8;
x_2 = l_Lean_Name_mkStr1(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__10() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__9;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__11() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("instNegInt", 10, 10);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__12() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__11;
x_2 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__8;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__13() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__12;
x_3 = l_Lean_Expr_const___override(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instToExprRat__mathlib___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_9; uint8_t x_10; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__2;
x_9 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__3;
x_10 = lean_int_dec_le(x_9, x_2);
if (x_10 == 0)
{
lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_11 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__7;
x_12 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__10;
x_13 = lp_mathlib_instToExprRat__mathlib___lam__0___closed__13;
x_14 = lean_int_neg(x_2);
lean_dec(x_2);
x_15 = l_Int_toNat(x_14);
lean_dec(x_14);
x_16 = l_Lean_instToExprInt_mkNat(x_15);
x_17 = l_Lean_mkApp3(x_11, x_12, x_13, x_16);
x_5 = x_17;
goto block_8;
}
else
{
lean_object* x_18; lean_object* x_19; 
x_18 = l_Int_toNat(x_2);
lean_dec(x_2);
x_19 = l_Lean_instToExprInt_mkNat(x_18);
x_5 = x_19;
goto block_8;
}
block_8:
{
lean_object* x_6; lean_object* x_7; 
x_6 = l_Lean_mkNatLit(x_3);
x_7 = l_Lean_mkAppB(x_4, x_5, x_6);
return x_7;
}
}
}
static lean_object* _init_lp_mathlib_instToExprRat__mathlib() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_instToExprRat__mathlib___lam__0), 1, 0);
x_2 = lp_mathlib_instToExprRat__mathlib___closed__4;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Int_cast___at___00Lean_Expr_rat_x3f_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = l_Rat_ofInt(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_rat_x3f___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("Div", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_rat_x3f___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("div", 3, 3);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Lean_Expr_rat_x3f___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Lean_Expr_rat_x3f___closed__1;
x_2 = lp_mathlib_Lean_Expr_rat_x3f___closed__0;
x_3 = l_Lean_Name_mkStr2(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_rat_x3f(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lp_mathlib_Lean_Expr_rat_x3f___closed__2;
x_3 = lean_unsigned_to_nat(4u);
x_4 = l_Lean_Expr_isAppOfArity(x_1, x_2, x_3);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = l_Lean_Expr_int_x3f(x_1);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; 
x_6 = lean_box(0);
return x_6;
}
else
{
uint8_t x_7; 
x_7 = !lean_is_exclusive(x_5);
if (x_7 == 0)
{
lean_object* x_8; lean_object* x_9; 
x_8 = lean_ctor_get(x_5, 0);
x_9 = l_Rat_ofInt(x_8);
lean_ctor_set(x_5, 0, x_9);
return x_5;
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_10 = lean_ctor_get(x_5, 0);
lean_inc(x_10);
lean_dec(x_5);
x_11 = l_Rat_ofInt(x_10);
x_12 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
}
}
else
{
lean_object* x_13; lean_object* x_14; 
x_13 = l_Lean_Expr_appArg_x21(x_1);
x_14 = l_Lean_Expr_nat_x3f(x_13);
if (lean_obj_tag(x_14) == 0)
{
lean_object* x_15; 
lean_dec_ref(x_1);
x_15 = lean_box(0);
return x_15;
}
else
{
lean_object* x_16; lean_object* x_17; uint8_t x_18; 
x_16 = lean_ctor_get(x_14, 0);
lean_inc(x_16);
lean_dec_ref(x_14);
x_17 = lean_unsigned_to_nat(1u);
x_18 = lean_nat_dec_eq(x_16, x_17);
if (x_18 == 0)
{
if (x_4 == 0)
{
lean_object* x_19; 
lean_dec(x_16);
lean_dec_ref(x_1);
x_19 = lean_box(0);
return x_19;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; 
x_20 = l_Lean_Expr_appFn_x21(x_1);
lean_dec_ref(x_1);
x_21 = l_Lean_Expr_appArg_x21(x_20);
lean_dec_ref(x_20);
x_22 = l_Lean_Expr_int_x3f(x_21);
if (lean_obj_tag(x_22) == 0)
{
lean_object* x_23; 
lean_dec(x_16);
x_23 = lean_box(0);
return x_23;
}
else
{
uint8_t x_24; 
x_24 = !lean_is_exclusive(x_22);
if (x_24 == 0)
{
lean_object* x_25; lean_object* x_26; lean_object* x_27; uint8_t x_28; 
x_25 = lean_ctor_get(x_22, 0);
lean_inc(x_16);
x_26 = l_mkRat(x_25, x_16);
x_27 = lean_ctor_get(x_26, 1);
lean_inc(x_27);
x_28 = lean_nat_dec_eq(x_27, x_16);
lean_dec(x_16);
lean_dec(x_27);
if (x_28 == 0)
{
lean_object* x_29; 
lean_dec_ref(x_26);
lean_free_object(x_22);
x_29 = lean_box(0);
return x_29;
}
else
{
lean_ctor_set(x_22, 0, x_26);
return x_22;
}
}
else
{
lean_object* x_30; lean_object* x_31; lean_object* x_32; uint8_t x_33; 
x_30 = lean_ctor_get(x_22, 0);
lean_inc(x_30);
lean_dec(x_22);
lean_inc(x_16);
x_31 = l_mkRat(x_30, x_16);
x_32 = lean_ctor_get(x_31, 1);
lean_inc(x_32);
x_33 = lean_nat_dec_eq(x_32, x_16);
lean_dec(x_16);
lean_dec(x_32);
if (x_33 == 0)
{
lean_object* x_34; 
lean_dec_ref(x_31);
x_34 = lean_box(0);
return x_34;
}
else
{
lean_object* x_35; 
x_35 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_35, 0, x_31);
return x_35;
}
}
}
}
}
else
{
lean_object* x_36; 
lean_dec(x_16);
lean_dec_ref(x_1);
x_36 = lean_box(0);
return x_36;
}
}
}
}
}
LEAN_EXPORT uint8_t lp_mathlib_Lean_Expr_isExplicitNumber(lean_object* x_1) {
_start:
{
switch (lean_obj_tag(x_1)) {
case 9:
{
uint8_t x_2; 
lean_dec_ref(x_1);
x_2 = 1;
return x_2;
}
case 10:
{
lean_object* x_3; 
x_3 = lean_ctor_get(x_1, 1);
lean_inc_ref(x_3);
lean_dec_ref(x_1);
x_1 = x_3;
goto _start;
}
default: 
{
lean_object* x_5; 
x_5 = lp_mathlib_Lean_Expr_rat_x3f(x_1);
if (lean_obj_tag(x_5) == 0)
{
uint8_t x_6; 
x_6 = 0;
return x_6;
}
else
{
uint8_t x_7; 
lean_dec_ref(x_5);
x_7 = 1;
return x_7;
}
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Lean_Expr_isExplicitNumber___boxed(lean_object* x_1) {
_start:
{
uint8_t x_2; lean_object* x_3; 
x_2 = lp_mathlib_Lean_Expr_isExplicitNumber(x_1);
x_3 = lean_box(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Tactic_Alias(uint8_t builtin);
lean_object* initialize_Lean_ToExpr(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Lean_Expr_Rat(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Tactic_Alias(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_ToExpr(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_instToExprRat__mathlib___closed__0 = _init_lp_mathlib_instToExprRat__mathlib___closed__0();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___closed__0);
lp_mathlib_instToExprRat__mathlib___closed__1 = _init_lp_mathlib_instToExprRat__mathlib___closed__1();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___closed__1);
lp_mathlib_instToExprRat__mathlib___closed__2 = _init_lp_mathlib_instToExprRat__mathlib___closed__2();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___closed__2);
lp_mathlib_instToExprRat__mathlib___closed__3 = _init_lp_mathlib_instToExprRat__mathlib___closed__3();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___closed__3);
lp_mathlib_instToExprRat__mathlib___closed__4 = _init_lp_mathlib_instToExprRat__mathlib___closed__4();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___closed__4);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__0 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__0);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__1 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__1);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__2 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__2);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__3 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__3();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__3);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__5 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__5();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__5);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__4 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__4();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__4);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__6 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__6();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__6);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__7 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__7();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__7);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__8 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__8();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__8);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__9 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__9();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__9);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__10 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__10();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__10);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__11 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__11();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__11);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__12 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__12();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__12);
lp_mathlib_instToExprRat__mathlib___lam__0___closed__13 = _init_lp_mathlib_instToExprRat__mathlib___lam__0___closed__13();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib___lam__0___closed__13);
lp_mathlib_instToExprRat__mathlib = _init_lp_mathlib_instToExprRat__mathlib();
lean_mark_persistent(lp_mathlib_instToExprRat__mathlib);
lp_mathlib_Lean_Expr_rat_x3f___closed__0 = _init_lp_mathlib_Lean_Expr_rat_x3f___closed__0();
lean_mark_persistent(lp_mathlib_Lean_Expr_rat_x3f___closed__0);
lp_mathlib_Lean_Expr_rat_x3f___closed__1 = _init_lp_mathlib_Lean_Expr_rat_x3f___closed__1();
lean_mark_persistent(lp_mathlib_Lean_Expr_rat_x3f___closed__1);
lp_mathlib_Lean_Expr_rat_x3f___closed__2 = _init_lp_mathlib_Lean_Expr_rat_x3f___closed__2();
lean_mark_persistent(lp_mathlib_Lean_Expr_rat_x3f___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
