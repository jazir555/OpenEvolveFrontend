// Lean compiler output
// Module: Mathlib.Data.Fintype.Option
// Imports: public import Init public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Data.Finset.Option
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
lean_object* l_List_lengthTR___redArg(lean_object*);
lean_object* lp_mathlib_Finset_eraseNone(lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2;
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOptionEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
lean_object* l_Nat_recCompiled___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_truncEquivFin___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOption(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_truncEquivOfCardEq___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
uint8_t l_Option_instDecidableEq___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Finset_insertNone___lam__0(lean_object*);
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4;
static lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3;
LEAN_EXPORT lean_object* lp_mathlib_instFintypeOption___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOptionEquiv(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeOption(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Nat_decEq___boxed(lean_object*, lean_object*);
lean_object* l_List_finRange(lean_object*);
lean_object* lp_mathlib_ULift_fintype___redArg(lean_object*);
lean_object* l_instDecidableEqPEmpty___boxed(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOption___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instFintypeOption(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_insertNone___lam__0(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instFintypeOption___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Finset_insertNone___lam__0(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOption(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Finset_eraseNone(lean_box(0));
x_4 = lean_apply_1(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOption___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_mathlib_Finset_eraseNone(lean_box(0));
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOptionEquiv___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Fintype_ofEquiv___redArg(x_1, x_2);
x_4 = lp_mathlib_Finset_eraseNone(lean_box(0));
x_5 = lean_apply_1(x_4, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypeOfOptionEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_fintypeOfOptionEquiv___redArg(x_3, x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_decEq___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = l_Option_instDecidableEq___redArg(x_1, x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
lean_inc(x_6);
x_8 = l_List_finRange(x_6);
x_9 = lp_mathlib_ULift_fintype___redArg(x_8);
lean_inc(x_9);
x_10 = lp_mathlib_Finset_insertNone___lam__0(x_9);
x_11 = lean_unsigned_to_nat(1u);
x_12 = lean_nat_add(x_6, x_11);
lean_dec(x_6);
x_13 = l_List_finRange(x_12);
x_14 = lp_mathlib_ULift_fintype___redArg(x_13);
x_15 = lp_mathlib_Fintype_truncEquivOfCardEq___redArg(x_10, x_14, x_1, x_2);
x_16 = lean_apply_4(x_3, lean_box(0), x_9, x_4, x_7);
x_17 = lean_apply_4(x_5, lean_box(0), lean_box(0), x_15, x_16);
return x_17;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = l_List_finRange(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1;
x_2 = lp_mathlib_ULift_fintype___redArg(x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_1 = lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0;
x_2 = lean_alloc_closure((void*)(l_instDecidableEqPEmpty___boxed), 2, 0);
x_3 = lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2;
x_4 = lean_box(0);
x_5 = lp_mathlib_Fintype_truncEquivOfCardEq___redArg(x_4, x_3, x_2, x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_ulift(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; 
x_6 = lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0;
x_7 = lean_alloc_closure((void*)(lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__0___boxed), 3, 1);
lean_closure_set(x_7, 0, x_6);
lean_inc(x_1);
x_8 = lean_alloc_closure((void*)(lp_mathlib_Fintype_truncRecEmptyOption___redArg___lam__1), 7, 5);
lean_closure_set(x_8, 0, x_7);
lean_closure_set(x_8, 1, x_6);
lean_closure_set(x_8, 2, x_3);
lean_closure_set(x_8, 3, x_6);
lean_closure_set(x_8, 4, x_1);
x_9 = l_List_lengthTR___redArg(x_4);
x_10 = lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3;
lean_inc(x_1);
x_11 = lean_apply_4(x_1, lean_box(0), lean_box(0), x_10, x_2);
x_12 = l_Nat_recCompiled___redArg(x_11, x_8, x_9);
lean_dec(x_9);
lean_dec(x_11);
x_13 = lp_mathlib_Fintype_truncEquivFin___redArg(x_5, x_4);
x_14 = lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4;
x_15 = lp_mathlib_Equiv_symm___redArg(x_13);
x_16 = lp_mathlib_Equiv_trans___redArg(x_14, x_15);
x_17 = lean_apply_4(x_1, lean_box(0), lean_box(0), x_16, x_12);
return x_17;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Fintype_truncRecEmptyOption(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Fintype_truncRecEmptyOption___redArg(x_2, x_3, x_4, x_6, x_7);
return x_8;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Option(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Option(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Option(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0 = _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__0);
lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1 = _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__1);
lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2 = _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2();
lean_mark_persistent(lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__2);
lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3 = _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3();
lean_mark_persistent(lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__3);
lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4 = _init_lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4();
lean_mark_persistent(lp_mathlib_Fintype_truncRecEmptyOption___redArg___closed__4);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
