// Lean compiler output
// Module: Mathlib.Data.Finset.Basic
// Imports: public import Init public import Mathlib.Data.Finset.Attach public import Mathlib.Data.Finset.Disjoint public import Mathlib.Data.Finset.Erase public import Mathlib.Data.Finset.Filter public import Mathlib.Data.Finset.Range public import Mathlib.Data.Finset.SDiff public import Mathlib.Data.Multiset.Basic public import Mathlib.Logic.Equiv.Set public import Mathlib.Order.Directed public import Mathlib.Order.Interval.Set.Defs public import Mathlib.Data.Set.SymmDiff
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
static lean_object* lp_mathlib_Equiv_Finset_union___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___redArg___lam__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_choose(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_chooseX___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_choose___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_chooseX(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___boxed(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sumPiEquivProdPi(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0;
static lean_object* lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1;
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Finset_union___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_List_chooseX___redArg(lean_object*, lean_object*);
uint8_t lp_mathlib_Multiset_decidableMem___redArg(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_Set_union_x27___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_piCongrLeft___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Finset_chooseX(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_List_chooseX___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_chooseX___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_List_chooseX___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_choose(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_List_chooseX___redArg(x_3, x_4);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_choose___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_List_chooseX___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT uint8_t lp_mathlib_Equiv_Finset_union___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; 
x_4 = lp_mathlib_Multiset_decidableMem___redArg(x_1, x_3, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
uint8_t x_4; lean_object* x_5; 
x_4 = lp_mathlib_Equiv_Finset_union___redArg___lam__0(x_1, x_2, x_3);
x_5 = lean_box(x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Equiv_Finset_union___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_Finset_union___redArg___lam__0___boxed), 3, 2);
lean_closure_set(x_3, 0, x_1);
lean_closure_set(x_3, 1, x_2);
x_4 = lp_mathlib_Equiv_Finset_union___redArg___closed__0;
x_5 = lp_mathlib_Equiv_Set_union_x27___redArg(x_3);
x_6 = lp_mathlib_Equiv_trans___redArg(x_4, x_5);
x_7 = lp_mathlib_Equiv_symm___redArg(x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_Finset_union___redArg(x_2, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_Finset_union___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Equiv_Finset_union(x_1, x_2, x_3, x_4, x_5);
lean_dec(x_4);
return x_6;
}
}
static lean_object* _init_lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumPiEquivProdPi(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0;
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_3 = lp_mathlib_Equiv_Finset_union___redArg(x_1, x_2);
x_4 = lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1;
x_5 = lp_mathlib_Equiv_piCongrLeft___redArg(x_3);
x_6 = lp_mathlib_Equiv_trans___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_piFinsetUnion___redArg(x_2, x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_piFinsetUnion___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Equiv_piFinsetUnion(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___lam__0(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Finset_equivToSet___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_Finset_equivToSet___lam__0___boxed), 1, 0);
lean_inc_ref(x_3);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Finset_equivToSet___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Finset_equivToSet(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Attach(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Disjoint(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Erase(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Filter(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_Range(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Finset_SDiff(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Multiset_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Set(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Directed(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_Interval_Set_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_SymmDiff(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Finset_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
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
res = initialize_mathlib_Mathlib_Data_Finset_SDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Multiset_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Equiv_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Directed(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_Interval_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_SymmDiff(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_Finset_union___redArg___closed__0 = _init_lp_mathlib_Equiv_Finset_union___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_Finset_union___redArg___closed__0);
lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0 = _init_lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_piFinsetUnion___redArg___closed__0);
lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1 = _init_lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_piFinsetUnion___redArg___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
