// Lean compiler output
// Module: Mathlib.Data.Fintype.Units
// Imports: public import Init public import Mathlib.Algebra.Ring.Int.Units public import Mathlib.Data.Fintype.Prod public import Mathlib.Data.Fintype.Sum public import Mathlib.SetTheory.Cardinal.Finite public import Mathlib.Algebra.GroupWithZero.Units.Equiv
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
lean_object* lp_mathlib_unitsEquivProdSubtype(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_UnitsInt_fintype___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UnitsInt_fintype___lam__0___boxed(lean_object*, lean_object*);
uint8_t lp_mathlib_Units_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_UnitsInt_fintype___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lean_nat_to_int(lean_object*);
static lean_object* lp_mathlib_UnitsInt_fintype___closed__4;
static lean_object* lp_mathlib_UnitsInt_fintype___closed__0;
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Monoid_toMulOneClass___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_UnitsInt_fintype;
lean_object* l_Int_instDecidableEq___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_product___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_fintype___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_UnitsInt_fintype___closed__3;
lean_object* lp_mathlib_Multiset_ndinsert___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lean_int_neg(lean_object*);
static lean_object* lp_mathlib_UnitsInt_fintype___closed__1;
LEAN_EXPORT uint8_t lp_mathlib_UnitsInt_fintype___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; uint8_t x_4; 
x_3 = lean_alloc_closure((void*)(l_Int_instDecidableEq___boxed), 2, 0);
x_4 = lp_mathlib_Units_instDecidableEq___redArg(x_3, x_1, x_2);
return x_4;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(1u);
x_2 = lean_nat_to_int(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_UnitsInt_fintype___closed__0;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_UnitsInt_fintype___closed__0;
x_2 = lean_int_neg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_UnitsInt_fintype___closed__2;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_UnitsInt_fintype___closed__3;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_UnitsInt_fintype___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_mathlib_UnitsInt_fintype___lam__0(x_1, x_2);
x_4 = lean_box(x_3);
return x_4;
}
}
static lean_object* _init_lp_mathlib_UnitsInt_fintype() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_UnitsInt_fintype___lam__0___boxed), 2, 0);
x_2 = lp_mathlib_UnitsInt_fintype___closed__1;
x_3 = lp_mathlib_UnitsInt_fintype___closed__4;
x_4 = lp_mathlib_Multiset_ndinsert___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT uint8_t lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; uint8_t x_9; 
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
lean_inc(x_1);
lean_inc(x_6);
lean_inc(x_5);
x_7 = lean_apply_2(x_1, x_5, x_6);
lean_inc_ref(x_2);
lean_inc(x_3);
x_8 = lean_apply_2(x_2, x_7, x_3);
x_9 = lean_unbox(x_8);
if (x_9 == 0)
{
uint8_t x_10; 
lean_dec(x_6);
lean_dec(x_5);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec(x_1);
x_10 = lean_unbox(x_8);
return x_10;
}
else
{
lean_object* x_11; lean_object* x_12; uint8_t x_13; 
x_11 = lean_apply_2(x_1, x_6, x_5);
x_12 = lean_apply_2(x_2, x_11, x_3);
x_13 = lean_unbox(x_12);
return x_13;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_4 = lp_mathlib_Monoid_toMulOneClass___redArg(x_1);
x_5 = lean_ctor_get(x_4, 0);
lean_inc(x_5);
x_6 = lean_ctor_get(x_4, 1);
lean_inc(x_6);
lean_dec_ref(x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___lam__0___boxed), 4, 3);
lean_closure_set(x_7, 0, x_6);
lean_closure_set(x_7, 1, x_3);
lean_closure_set(x_7, 2, x_5);
lean_inc(x_2);
x_8 = lp_mathlib_Multiset_product___redArg(x_2, x_2);
x_9 = lp_mathlib_Subtype_fintype___redArg(x_7, x_8);
x_10 = lp_mathlib_unitsEquivProdSubtype(lean_box(0), x_1);
x_11 = lp_mathlib_Equiv_symm___redArg(x_10);
x_12 = lp_mathlib_Fintype_ofEquiv___redArg(x_9, x_11);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instFintypeUnitsOfDecidableEq___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_instFintypeUnitsOfDecidableEq(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeUnitsOfDecidableEq___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_instFintypeUnitsOfDecidableEq___redArg(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Sum(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Units(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Int_Units(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Sum(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_SetTheory_Cardinal_Finite(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_GroupWithZero_Units_Equiv(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_UnitsInt_fintype___closed__0 = _init_lp_mathlib_UnitsInt_fintype___closed__0();
lean_mark_persistent(lp_mathlib_UnitsInt_fintype___closed__0);
lp_mathlib_UnitsInt_fintype___closed__1 = _init_lp_mathlib_UnitsInt_fintype___closed__1();
lean_mark_persistent(lp_mathlib_UnitsInt_fintype___closed__1);
lp_mathlib_UnitsInt_fintype___closed__2 = _init_lp_mathlib_UnitsInt_fintype___closed__2();
lean_mark_persistent(lp_mathlib_UnitsInt_fintype___closed__2);
lp_mathlib_UnitsInt_fintype___closed__3 = _init_lp_mathlib_UnitsInt_fintype___closed__3();
lean_mark_persistent(lp_mathlib_UnitsInt_fintype___closed__3);
lp_mathlib_UnitsInt_fintype___closed__4 = _init_lp_mathlib_UnitsInt_fintype___closed__4();
lean_mark_persistent(lp_mathlib_UnitsInt_fintype___closed__4);
lp_mathlib_UnitsInt_fintype = _init_lp_mathlib_UnitsInt_fintype();
lean_mark_persistent(lp_mathlib_UnitsInt_fintype);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
