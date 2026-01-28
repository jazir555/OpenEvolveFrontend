// Lean compiler output
// Module: Mathlib.Control.Traversable.Basic
// Imports: public import Init public import Mathlib.Data.Option.Defs public import Mathlib.Control.Functor public import Batteries.Data.List.Basic public import Mathlib.Control.Basic
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
extern lean_object* l_List_instFunctor;
static lean_object* lp_mathlib_instTraversableId___closed__2;
static lean_object* lp_mathlib_instTraversableId___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instTraversableSum___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instInhabited(lean_object*, lean_object*);
static lean_object* lp_mathlib_ApplicativeTransformation_instInhabited___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_batteries_List_traverse___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_instFunctorOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instTraversableSum___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instTraversableOption___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse___redArg___lam__0(lean_object*);
lean_object* l_id___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_sequence___redArg___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_sequence___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instTraversableOption___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instTraversableOption;
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Option_map(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Option_traverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTraversableList;
static lean_object* lp_mathlib_instTraversableOption___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_sequence(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__1___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_instTraversableOption___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId;
static lean_object* lp_mathlib_instTraversableId___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_instTraversableList___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Id_instMonad___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Sum_instMonad__mathlib(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instInhabited___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_instTraversableSum(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_apply_2(x_1, lean_box(0), x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___lam__0), 3, 0);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instCoeFunForallForall___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_ApplicativeTransformation_instCoeFunForallForall(x_1, x_2, x_3, x_4);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ApplicativeTransformation_idTransformation___lam__0(x_1, x_2);
lean_dec(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ApplicativeTransformation_idTransformation___lam__0___boxed), 2, 0);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_idTransformation___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ApplicativeTransformation_idTransformation(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
static lean_object* _init_lp_mathlib_ApplicativeTransformation_instInhabited___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ApplicativeTransformation_idTransformation___lam__0___boxed), 2, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instInhabited(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ApplicativeTransformation_instInhabited___closed__0;
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_instInhabited___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_ApplicativeTransformation_instInhabited(x_1, x_2);
lean_dec_ref(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_apply_2(x_1, lean_box(0), x_4);
x_6 = lean_apply_2(x_2, lean_box(0), x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_ApplicativeTransformation_comp___redArg___lam__0), 4, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ApplicativeTransformation_comp___redArg(x_7, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ApplicativeTransformation_comp___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_ApplicativeTransformation_comp(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec_ref(x_6);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
static lean_object* _init_lp_mathlib_sequence___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_id___boxed), 2, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sequence___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_ctor_get(x_2, 1);
lean_inc(x_4);
lean_dec_ref(x_2);
x_5 = lp_mathlib_sequence___redArg___closed__0;
x_6 = lean_apply_6(x_4, lean_box(0), x_1, lean_box(0), lean_box(0), x_5, x_3);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_sequence(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_sequence___redArg(x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lean_apply_1(x_5, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_instTraversableId___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instTraversableId___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Id_instMonad___lam__1___boxed), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instTraversableId___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instTraversableId___closed__1;
x_2 = lp_mathlib_instTraversableId___closed__0;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableId___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_instTraversableId___lam__0(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec_ref(x_2);
return x_7;
}
}
static lean_object* _init_lp_mathlib_instTraversableId() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_instTraversableId___lam__0___boxed), 6, 0);
x_2 = lp_mathlib_instTraversableId___closed__2;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableOption___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Option_traverse___redArg(x_2, x_5, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_instTraversableOption___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_instFunctorOption___lam__0), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instTraversableOption___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Option_map), 4, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_instTraversableOption___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_instTraversableOption___closed__0;
x_2 = lp_mathlib_instTraversableOption___closed__1;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
static lean_object* _init_lp_mathlib_instTraversableOption() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_instTraversableOption___lam__0), 6, 0);
x_2 = lp_mathlib_instTraversableOption___closed__2;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableList___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_batteries_List_traverse___redArg(x_2, x_5, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_instTraversableList() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_instTraversableList___lam__0), 6, 0);
x_2 = l_List_instFunctor;
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse___redArg___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; uint8_t x_5; 
lean_dec(x_2);
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = !lean_is_exclusive(x_3);
if (x_5 == 0)
{
lean_object* x_6; 
x_6 = lean_apply_2(x_4, lean_box(0), x_3);
return x_6;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_3, 0);
lean_inc(x_7);
lean_dec(x_3);
x_8 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_8, 0, x_7);
x_9 = lean_apply_2(x_4, lean_box(0), x_8);
return x_9;
}
}
else
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_10 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_10);
lean_dec_ref(x_1);
x_11 = lean_ctor_get(x_3, 0);
lean_inc(x_11);
lean_dec_ref(x_3);
x_12 = lean_ctor_get(x_10, 0);
lean_inc(x_12);
lean_dec_ref(x_10);
x_13 = lean_alloc_closure((void*)(lp_mathlib_Sum_traverse___redArg___lam__0), 1, 0);
x_14 = lean_apply_1(x_2, x_11);
x_15 = lean_apply_4(x_12, lean_box(0), lean_box(0), x_13, x_14);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Sum_traverse(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7) {
_start:
{
lean_object* x_8; 
x_8 = lp_mathlib_Sum_traverse___redArg(x_3, x_6, x_7);
return x_8;
}
}
static lean_object* _init_lp_mathlib_instTraversableSum___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Sum_instMonad__mathlib(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_instTraversableSum___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Sum_traverse), 7, 1);
lean_closure_set(x_1, 0, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_instTraversableSum(lean_object* x_1) {
_start:
{
lean_object* x_2; uint8_t x_3; 
x_2 = lp_mathlib_instTraversableSum___closed__0;
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_2, 0);
x_5 = lean_ctor_get(x_2, 1);
lean_dec(x_5);
x_6 = lean_ctor_get(x_4, 0);
lean_inc_ref(x_6);
lean_dec_ref(x_4);
x_7 = lp_mathlib_instTraversableSum___closed__1;
lean_ctor_set(x_2, 1, x_7);
lean_ctor_set(x_2, 0, x_6);
return x_2;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; 
x_8 = lean_ctor_get(x_2, 0);
lean_inc(x_8);
lean_dec(x_2);
x_9 = lean_ctor_get(x_8, 0);
lean_inc_ref(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_instTraversableSum___closed__1;
x_11 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_11, 0, x_9);
lean_ctor_set(x_11, 1, x_10);
return x_11;
}
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Option_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Functor(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Data_List_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Control_Traversable_Basic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Option_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Functor(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Data_List_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ApplicativeTransformation_instInhabited___closed__0 = _init_lp_mathlib_ApplicativeTransformation_instInhabited___closed__0();
lean_mark_persistent(lp_mathlib_ApplicativeTransformation_instInhabited___closed__0);
lp_mathlib_sequence___redArg___closed__0 = _init_lp_mathlib_sequence___redArg___closed__0();
lean_mark_persistent(lp_mathlib_sequence___redArg___closed__0);
lp_mathlib_instTraversableId___closed__0 = _init_lp_mathlib_instTraversableId___closed__0();
lean_mark_persistent(lp_mathlib_instTraversableId___closed__0);
lp_mathlib_instTraversableId___closed__1 = _init_lp_mathlib_instTraversableId___closed__1();
lean_mark_persistent(lp_mathlib_instTraversableId___closed__1);
lp_mathlib_instTraversableId___closed__2 = _init_lp_mathlib_instTraversableId___closed__2();
lean_mark_persistent(lp_mathlib_instTraversableId___closed__2);
lp_mathlib_instTraversableId = _init_lp_mathlib_instTraversableId();
lean_mark_persistent(lp_mathlib_instTraversableId);
lp_mathlib_instTraversableOption___closed__0 = _init_lp_mathlib_instTraversableOption___closed__0();
lean_mark_persistent(lp_mathlib_instTraversableOption___closed__0);
lp_mathlib_instTraversableOption___closed__1 = _init_lp_mathlib_instTraversableOption___closed__1();
lean_mark_persistent(lp_mathlib_instTraversableOption___closed__1);
lp_mathlib_instTraversableOption___closed__2 = _init_lp_mathlib_instTraversableOption___closed__2();
lean_mark_persistent(lp_mathlib_instTraversableOption___closed__2);
lp_mathlib_instTraversableOption = _init_lp_mathlib_instTraversableOption();
lean_mark_persistent(lp_mathlib_instTraversableOption);
lp_mathlib_instTraversableList = _init_lp_mathlib_instTraversableList();
lean_mark_persistent(lp_mathlib_instTraversableList);
lp_mathlib_instTraversableSum___closed__0 = _init_lp_mathlib_instTraversableSum___closed__0();
lean_mark_persistent(lp_mathlib_instTraversableSum___closed__0);
lp_mathlib_instTraversableSum___closed__1 = _init_lp_mathlib_instTraversableSum___closed__1();
lean_mark_persistent(lp_mathlib_instTraversableSum___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
