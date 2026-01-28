// Lean compiler output
// Module: Mathlib.Data.Fintype.Perm
// Imports: public import Init public import Mathlib.Algebra.BigOperators.Group.List.Defs public import Mathlib.Algebra.Group.End public import Mathlib.Algebra.Group.Nat.Defs public import Mathlib.Data.Fintype.EquivFin public import Mathlib.Data.Nat.Factorial.Basic
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
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_permsOfList___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype___redArg___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_List_mapTR_loop___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_equivCongr___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Perm_0__permsOfList_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_permsOfList___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypePerm(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_ofEquiv___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Fintype_truncEquivFin___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_instFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype___redArg___lam__2(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_permsOfList___redArg___closed__1;
static lean_object* lp_mathlib_permsOfList___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instFintype___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_instFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* l_List_appendTR___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Perm_0__permsOfList_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_permsOfList___redArg___closed__0;
lean_object* l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_eq(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_swap___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_permsOfList___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lp_mathlib_Fintype_decidableForallFintype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_permsOfList(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_permsOfFinset(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulEquiv_instFintype___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_refl(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_permsOfFinset___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_fintypePerm___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib_MulEquiv_instFintype___redArg___lam__1(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_permsOfList___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_refl(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_permsOfList___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lp_mathlib_permsOfList___redArg___closed__0;
x_3 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_permsOfList___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lp_mathlib_Equiv_swap___redArg(x_1, x_2, x_3);
x_6 = lp_mathlib_Equiv_trans___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_permsOfList___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_permsOfList___redArg___lam__0), 4, 3);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_2);
lean_closure_set(x_5, 2, x_4);
x_6 = lean_box(0);
x_7 = l_List_mapTR_loop___redArg(x_5, x_3, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_permsOfList___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_permsOfList___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; 
lean_dec_ref(x_1);
x_3 = lp_mathlib_permsOfList___redArg___closed__1;
return x_3;
}
else
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lean_ctor_get(x_2, 0);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 1);
lean_inc(x_5);
lean_dec_ref(x_2);
lean_inc(x_5);
lean_inc_ref(x_1);
x_6 = lp_mathlib_permsOfList___redArg(x_1, x_5);
lean_inc(x_6);
x_7 = lean_alloc_closure((void*)(lp_mathlib_permsOfList___redArg___lam__1), 4, 3);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_4);
lean_closure_set(x_7, 2, x_6);
x_8 = lp_mathlib_permsOfList___redArg___closed__2;
x_9 = l___private_Init_Data_List_Impl_0__List_flatMapTR_go(lean_box(0), lean_box(0), x_7, x_5, x_8);
x_10 = l_List_appendTR___redArg(x_6, x_9);
return x_10;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_permsOfList(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_permsOfList___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Perm_0__permsOfList_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_3);
x_4 = lean_box(0);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_dec(x_2);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_apply_2(x_3, x_6, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Data_Fintype_Perm_0__permsOfList_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib___private_Mathlib_Data_Fintype_Perm_0__permsOfList_match__1_splitter___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_permsOfFinset(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_permsOfList___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_permsOfFinset___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_permsOfList___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypePerm(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_permsOfList___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_fintypePerm___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_permsOfList___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_instFintype___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_5 = l_List_lengthTR___redArg(x_4);
x_6 = l_List_lengthTR___redArg(x_3);
x_7 = lean_nat_dec_eq(x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
if (x_7 == 0)
{
lean_object* x_8; 
lean_dec(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
x_8 = lean_box(0);
return x_8;
}
else
{
lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
lean_inc(x_3);
lean_inc_ref(x_1);
x_9 = lp_mathlib_Fintype_truncEquivFin___redArg(x_1, x_3);
x_10 = lp_mathlib_Fintype_truncEquivFin___redArg(x_2, x_4);
x_11 = lp_mathlib_permsOfList___redArg(x_1, x_3);
x_12 = lp_mathlib_permsOfList___redArg___closed__0;
x_13 = lp_mathlib_Equiv_symm___redArg(x_10);
x_14 = lp_mathlib_Equiv_trans___redArg(x_9, x_13);
x_15 = lp_mathlib_Equiv_equivCongr___redArg(x_12, x_14);
x_16 = lp_mathlib_Fintype_ofEquiv___redArg(x_11, x_15);
return x_16;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_instFintype(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_instFintype___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulEquiv_instFintype___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; uint8_t x_14; 
x_7 = lean_ctor_get(x_1, 0);
lean_inc(x_7);
lean_dec_ref(x_1);
lean_inc(x_6);
lean_inc(x_3);
x_8 = lean_apply_2(x_2, x_3, x_6);
lean_inc(x_7);
x_9 = lean_apply_1(x_7, x_8);
lean_inc(x_7);
x_10 = lean_apply_1(x_7, x_3);
x_11 = lean_apply_1(x_7, x_6);
x_12 = lean_apply_2(x_4, x_10, x_11);
x_13 = lean_apply_2(x_5, x_9, x_12);
x_14 = lean_unbox(x_13);
return x_14;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lp_mathlib_MulEquiv_instFintype___redArg___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT uint8_t lp_mathlib_MulEquiv_instFintype___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; uint8_t x_8; 
x_7 = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_instFintype___redArg___lam__0___boxed), 6, 5);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_6);
lean_closure_set(x_7, 3, x_3);
lean_closure_set(x_7, 4, x_4);
x_8 = lp_mathlib_Fintype_decidableForallFintype___redArg(x_7, x_5);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype___redArg___lam__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
uint8_t x_7; lean_object* x_8; 
x_7 = lp_mathlib_MulEquiv_instFintype___redArg___lam__1(x_1, x_2, x_3, x_4, x_5, x_6);
x_8 = lean_box(x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype___redArg___lam__2(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; uint8_t x_7; 
lean_inc(x_4);
lean_inc_ref(x_5);
x_6 = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_instFintype___redArg___lam__1___boxed), 6, 5);
lean_closure_set(x_6, 0, x_5);
lean_closure_set(x_6, 1, x_1);
lean_closure_set(x_6, 2, x_2);
lean_closure_set(x_6, 3, x_3);
lean_closure_set(x_6, 4, x_4);
x_7 = lp_mathlib_Fintype_decidableForallFintype___redArg(x_6, x_4);
if (x_7 == 0)
{
lean_object* x_8; 
lean_dec_ref(x_5);
x_8 = lean_box(0);
return x_8;
}
else
{
lean_object* x_9; 
x_9 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_9, 0, x_5);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_5);
lean_inc_ref(x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_instFintype___redArg___lam__2), 5, 4);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_4);
lean_closure_set(x_7, 3, x_5);
x_8 = lp_mathlib_Equiv_instFintype___redArg(x_3, x_4, x_5, x_6);
x_9 = lp_mathlib_Multiset_filterMap___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_instFintype(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_MulEquiv_instFintype___redArg(x_3, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instFintype___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
lean_inc(x_5);
lean_inc_ref(x_4);
x_7 = lean_alloc_closure((void*)(lp_mathlib_MulEquiv_instFintype___redArg___lam__2), 5, 4);
lean_closure_set(x_7, 0, x_1);
lean_closure_set(x_7, 1, x_2);
lean_closure_set(x_7, 2, x_4);
lean_closure_set(x_7, 3, x_5);
x_8 = lp_mathlib_Equiv_instFintype___redArg(x_3, x_4, x_5, x_6);
x_9 = lp_mathlib_Multiset_filterMap___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_instFintype(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_AddEquiv_instFintype___redArg(x_3, x_4, x_5, x_6, x_7, x_8);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_End(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_EquivFin(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Fintype_Perm(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_BigOperators_Group_List_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_End(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Group_Nat_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_EquivFin(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Nat_Factorial_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_permsOfList___redArg___closed__0 = _init_lp_mathlib_permsOfList___redArg___closed__0();
lean_mark_persistent(lp_mathlib_permsOfList___redArg___closed__0);
lp_mathlib_permsOfList___redArg___closed__1 = _init_lp_mathlib_permsOfList___redArg___closed__1();
lean_mark_persistent(lp_mathlib_permsOfList___redArg___closed__1);
lp_mathlib_permsOfList___redArg___closed__2 = _init_lp_mathlib_permsOfList___redArg___closed__2();
lean_mark_persistent(lp_mathlib_permsOfList___redArg___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
