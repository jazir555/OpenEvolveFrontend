// Lean compiler output
// Module: Mathlib.Logic.Equiv.Multiset
// Imports: public import Init public import Mathlib.Data.Multiset.Sort public import Mathlib.Logic.Equiv.List
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
lean_object* lp_mathlib_List_encodable___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_lower(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__Denumerable_lower_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_multiset___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__Denumerable_lower_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_lower___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_encodeMultiset___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___redArg(lean_object*, lean_object*, lean_object*);
extern lean_object* lp_mathlib_Nat_encodable;
LEAN_EXPORT lean_object* lp_mathlib_decodeMultiset___redArg___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_sort___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_encodable(lean_object*, lean_object*);
lean_object* lp_mathlib_Denumerable_mk_x27___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_decodeMultiset___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_multiset___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_multiset___redArg___lam__1(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Denumerable_multiset___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_encodeMultiset(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Denumerable_ofNat___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_raise___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Denumerable_multiset___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_decodeMultiset(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_raise(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___redArg___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_multiset(lean_object*, lean_object*);
lean_object* l_Nat_decLe___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Encodable_decodeList___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Encodable_encodeList___redArg(lean_object*, lean_object*);
lean_object* lean_nat_sub(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_decodeMultiset___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Multiset_encodable___redArg(lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle(lean_object*, lean_object*, lean_object*, lean_object*);
uint8_t lean_nat_dec_le(lean_object*, lean_object*);
lean_object* lean_nat_add(lean_object*, lean_object*);
lean_object* lp_mathlib_Denumerable_ofNat(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint8_t x_7; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
lean_inc_ref(x_4);
x_5 = lean_apply_1(x_4, x_2);
x_6 = lean_apply_1(x_4, x_3);
x_7 = lean_nat_dec_le(x_5, x_6);
lean_dec(x_6);
lean_dec(x_5);
return x_7;
}
}
LEAN_EXPORT uint8_t lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; lean_object* x_6; 
x_5 = lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle(x_1, x_2, x_3, x_4);
x_6 = lean_box(x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___redArg___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___redArg(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_encodeMultiset___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__decidable__enle___boxed), 4, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lp_mathlib_Multiset_sort___redArg(x_2, x_3);
x_5 = lp_mathlib_Encodable_encodeList___redArg(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_encodeMultiset(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_encodeMultiset___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_decodeMultiset___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Encodable_decodeList___redArg(x_1, x_2);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
else
{
uint8_t x_5; 
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
return x_3;
}
else
{
lean_object* x_6; lean_object* x_7; 
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec(x_3);
x_7 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_7, 0, x_6);
return x_7;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_decodeMultiset(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_decodeMultiset___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_decodeMultiset___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_decodeMultiset(x_1, x_2, x_3);
lean_dec(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_decodeMultiset___redArg___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_decodeMultiset___redArg(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_encodable___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_encodeMultiset), 3, 2);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_decodeMultiset___boxed), 3, 2);
lean_closure_set(x_3, 0, lean_box(0));
lean_closure_set(x_3, 1, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Multiset_encodable(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_encodable___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_lower(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return x_1;
}
else
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_nat_sub(x_4, x_2);
x_7 = lp_mathlib_Denumerable_lower(x_5, x_4);
lean_dec(x_4);
lean_ctor_set(x_1, 1, x_7);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_1);
x_10 = lean_nat_sub(x_8, x_2);
x_11 = lp_mathlib_Denumerable_lower(x_9, x_8);
lean_dec(x_8);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_lower___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Denumerable_lower(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_raise(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
return x_1;
}
else
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_1);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_nat_add(x_4, x_2);
lean_dec(x_4);
x_7 = lp_mathlib_Denumerable_raise(x_5, x_6);
lean_ctor_set(x_1, 1, x_7);
lean_ctor_set(x_1, 0, x_6);
return x_1;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_8 = lean_ctor_get(x_1, 0);
x_9 = lean_ctor_get(x_1, 1);
lean_inc(x_9);
lean_inc(x_8);
lean_dec(x_1);
x_10 = lean_nat_add(x_8, x_2);
lean_dec(x_8);
x_11 = lp_mathlib_Denumerable_raise(x_9, x_10);
x_12 = lean_alloc_ctor(1, 2, 0);
lean_ctor_set(x_12, 0, x_10);
lean_ctor_set(x_12, 1, x_11);
return x_12;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_raise___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Denumerable_raise(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__Denumerable_lower_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_5; 
lean_dec(x_4);
x_5 = lean_apply_1(x_3, x_2);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; 
lean_dec(x_3);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
x_7 = lean_ctor_get(x_1, 1);
lean_inc(x_7);
lean_dec_ref(x_1);
x_8 = lean_apply_3(x_4, x_6, x_7, x_2);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__Denumerable_lower_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib___private_Mathlib_Logic_Equiv_Multiset_0__Denumerable_lower_match__1_splitter___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_Denumerable_multiset___redArg___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Nat_decLe___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_multiset___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; 
x_4 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_4);
lean_dec_ref(x_1);
x_5 = lp_mathlib_Multiset_map___redArg(x_4, x_3);
x_6 = lp_mathlib_Denumerable_multiset___redArg___lam__0___closed__0;
x_7 = lp_mathlib_Multiset_sort___redArg(x_5, x_6);
x_8 = lean_unsigned_to_nat(0u);
x_9 = lp_mathlib_Denumerable_lower(x_7, x_8);
x_10 = lp_mathlib_Encodable_encodeList___redArg(x_2, x_9);
return x_10;
}
}
static lean_object* _init_lp_mathlib_Denumerable_multiset___redArg___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Nat_encodable;
x_2 = lp_mathlib_List_encodable___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_multiset___redArg___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Denumerable_ofNat), 3, 2);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, x_1);
x_5 = lp_mathlib_Denumerable_ofNat___redArg(x_2, x_3);
x_6 = lean_unsigned_to_nat(0u);
x_7 = lp_mathlib_Denumerable_raise(x_5, x_6);
x_8 = lp_mathlib_Multiset_map___redArg(x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_multiset___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lp_mathlib_Nat_encodable;
lean_inc_ref(x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Denumerable_multiset___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lp_mathlib_Denumerable_multiset___redArg___closed__0;
x_5 = lean_alloc_closure((void*)(lp_mathlib_Denumerable_multiset___redArg___lam__1), 3, 2);
lean_closure_set(x_5, 0, x_1);
lean_closure_set(x_5, 1, x_4);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_5);
x_7 = lp_mathlib_Denumerable_mk_x27___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Denumerable_multiset(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Denumerable_multiset___redArg(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Sort(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_List(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Multiset(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Sort(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_List(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Denumerable_multiset___redArg___lam__0___closed__0 = _init_lp_mathlib_Denumerable_multiset___redArg___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Denumerable_multiset___redArg___lam__0___closed__0);
lp_mathlib_Denumerable_multiset___redArg___closed__0 = _init_lp_mathlib_Denumerable_multiset___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Denumerable_multiset___redArg___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
