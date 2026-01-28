// Lean compiler output
// Module: Mathlib.Data.Finset.Image
// Imports: public import Init public import Mathlib.Algebra.NeZero public import Mathlib.Data.Finset.Attach public import Mathlib.Data.Finset.Disjoint public import Mathlib.Data.Finset.Erase public import Mathlib.Data.Finset.Filter public import Mathlib.Data.Finset.Range public import Mathlib.Data.Finset.Lattice.Lemmas public import Mathlib.Data.Finset.SDiff public import Mathlib.Data.Fintype.Defs
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__0(lean_object*);
lean_object* lp_mathlib_Multiset_filter___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_map(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_map___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filterMap(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_List_dedup___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm(lean_object*, lean_object*);
lean_object* lp_mathlib_Subtype_impEmbedding___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_map___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__2(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__1(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_toEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_filterMap___redArg(lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_attach___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_image(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_mapEmbedding(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Multiset_filterMap___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_mapEmbedding___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_image___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_map___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_map___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Finset_map___redArg___lam__0), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lp_mathlib_Multiset_map___redArg(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_map(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_map___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mapEmbedding(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Finset_map), 4, 3);
lean_closure_set(x_4, 0, lean_box(0));
lean_closure_set(x_4, 1, lean_box(0));
lean_closure_set(x_4, 2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_mapEmbedding___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Finset_map), 4, 3);
lean_closure_set(x_2, 0, lean_box(0));
lean_closure_set(x_2, 1, lean_box(0));
lean_closure_set(x_2, 2, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_image___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lp_mathlib_Multiset_map___redArg(x_2, x_3);
x_5 = lp_mathlib_List_dedup___redArg(x_1, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_image(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Finset_image___redArg(x_3, x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filterMap(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Multiset_filterMap___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_filterMap___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Multiset_filterMap___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Finset_subtype___redArg___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Finset_subtype___redArg___lam__0___boxed), 1, 0);
x_4 = lp_mathlib_Multiset_filter___redArg(x_1, x_2);
x_5 = lp_mathlib_Multiset_attach___redArg(x_4);
x_6 = lp_mathlib_Finset_map___redArg(x_3, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_subtype(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Finset_subtype___redArg(x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lp_mathlib_Equiv_toEmbedding___redArg(x_1);
x_4 = lp_mathlib_Finset_map___redArg(x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Equiv_symm___redArg(x_1);
x_4 = lp_mathlib_Equiv_toEmbedding___redArg(x_3);
x_5 = lp_mathlib_Finset_map___redArg(x_4, x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
lean_inc_ref(x_1);
x_2 = lean_alloc_closure((void*)(lp_mathlib_Equiv_finsetCongr___redArg___lam__0), 2, 1);
lean_closure_set(x_2, 0, x_1);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_finsetCongr___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetCongr(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_finsetCongr___redArg(x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_map___redArg(x_1, x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Subtype_impEmbedding___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__2(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0;
x_3 = lp_mathlib_Multiset_attach___redArg(x_1);
x_4 = lp_mathlib_Finset_map___redArg(x_2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_finsetSubtypeComm___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_finsetSubtypeComm(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_finsetSubtypeComm___lam__0___boxed), 1, 0);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_finsetSubtypeComm___lam__1), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_finsetSubtypeComm___lam__2), 1, 0);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_NeZero(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Attach(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Erase(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Fintype_Defs(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Image(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_NeZero(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Attach(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Disjoint(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Erase(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Filter(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Range(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_Lattice_Lemmas(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Fintype_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0 = _init_lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_finsetSubtypeComm___lam__2___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
