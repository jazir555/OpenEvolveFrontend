// Lean compiler output
// Module: Mathlib.Logic.Equiv.Embedding
// Imports: public import Init public import Mathlib.Logic.Embedding.Set
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
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__2(lean_object*);
static lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__2;
static lean_object* lp_mathlib_Equiv_codRestrict___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_codRestrict___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_trans___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg(lean_object*);
lean_object* lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__1(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__1;
static lean_object* lp_mathlib_Equiv_codRestrict___closed__0;
static lean_object* lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_inl___lam__0(lean_object*);
lean_object* lp_mathlib_Function_Embedding_inr___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0(lean_object*);
lean_object* lp_mathlib_Function_Embedding_subtype___lam__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Embedding_0__Equiv_sumEmbeddingEquivProdEmbeddingDisjoint_match__1_splitter___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_codRestrict(lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Function_Embedding_codRestrict___redArg(lean_object*);
static lean_object* lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__0;
static lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__0;
lean_object* lp_mathlib_Equiv_subtypeEquivProp(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___boxed(lean_object*);
static lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__1;
static lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__0;
lean_object* lp_mathlib_Function_Embedding_trans___redArg(lean_object*, lean_object*);
static lean_object* lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Embedding_0__Equiv_sumEmbeddingEquivProdEmbeddingDisjoint_match__1_splitter(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_sigmaCongrRight___redArg(lean_object*);
static lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___closed__0;
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Embedding_0__Equiv_sumEmbeddingEquivProdEmbeddingDisjoint_match__1_splitter___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_3);
x_4 = lean_ctor_get(x_1, 0);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_2);
x_6 = lean_ctor_get(x_1, 0);
lean_inc(x_6);
lean_dec_ref(x_1);
x_7 = lean_apply_1(x_3, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib___private_Mathlib_Logic_Equiv_Embedding_0__Equiv_sumEmbeddingEquivProdEmbeddingDisjoint_match__1_splitter(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib___private_Mathlib_Logic_Equiv_Embedding_0__Equiv_sumEmbeddingEquivProdEmbeddingDisjoint_match__1_splitter___redArg(x_4, x_5, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_inl___lam__0), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_inr___lam__0), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__0;
lean_inc(x_1);
x_3 = lp_mathlib_Function_Embedding_trans___redArg(x_2, x_1);
x_4 = lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__1;
x_5 = lp_mathlib_Function_Embedding_trans___redArg(x_4, x_1);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_3);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_4; lean_object* x_5; 
lean_dec(x_2);
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
lean_dec_ref(x_3);
x_5 = lean_apply_1(x_1, x_4);
return x_5;
}
else
{
lean_object* x_6; lean_object* x_7; 
lean_dec(x_1);
x_6 = lean_ctor_get(x_3, 0);
lean_inc(x_6);
lean_dec_ref(x_3);
x_7 = lean_apply_1(x_2, x_6);
return x_7;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__2(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__1), 3, 2);
lean_closure_set(x_4, 0, x_2);
lean_closure_set(x_4, 1, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0), 1, 0);
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__2), 1, 0);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_Equiv_codRestrict___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_codRestrict___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Equiv_codRestrict___lam__0___closed__0;
x_4 = lp_mathlib_Function_Embedding_trans___redArg(x_1, x_3);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Equiv_codRestrict___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_Function_Embedding_codRestrict___redArg), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_codRestrict(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_4 = lp_mathlib_Equiv_codRestrict___closed__0;
x_5 = lean_alloc_closure((void*)(lp_mathlib_Equiv_codRestrict___lam__0), 2, 0);
x_6 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_6, 0, x_4);
lean_ctor_set(x_6, 1, x_5);
return x_6;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_subtypeEquivProp(lean_box(0), lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_codRestrict(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__1;
x_2 = lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__0;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__2;
return x_2;
}
}
static lean_object* _init_lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_subtypeProdEquivSigmaSubtype(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___boxed), 1, 0);
x_5 = lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___closed__0;
x_6 = lp_mathlib_Equiv_sigmaCongrRight___redArg(x_4);
x_7 = lp_mathlib_Equiv_trans___redArg(x_5, x_6);
return x_7;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted(lean_box(0), lean_box(0), lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__1;
x_2 = lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__0;
x_3 = lp_mathlib_Equiv_trans___redArg(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__2;
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__0(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__1(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_apply_1(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__0___boxed), 2, 0);
x_3 = lean_alloc_closure((void*)(lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg___lam__1), 2, 1);
lean_closure_set(x_3, 0, x_1);
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_3);
lean_ctor_set(x_4, 1, x_2);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Equiv_uniqueEmbeddingEquivResult(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_Equiv_uniqueEmbeddingEquivResult___redArg(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Logic_Embedding_Set(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Logic_Equiv_Embedding(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Logic_Embedding_Set(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__0 = _init_lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__0);
lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__1 = _init_lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_sumEmbeddingEquivProdEmbeddingDisjoint___lam__0___closed__1);
lp_mathlib_Equiv_codRestrict___lam__0___closed__0 = _init_lp_mathlib_Equiv_codRestrict___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_codRestrict___lam__0___closed__0);
lp_mathlib_Equiv_codRestrict___closed__0 = _init_lp_mathlib_Equiv_codRestrict___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_codRestrict___closed__0);
lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__0 = _init_lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__0);
lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__1 = _init_lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__1);
lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__2 = _init_lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__2();
lean_mark_persistent(lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___lam__0___closed__2);
lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___closed__0 = _init_lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_prodEmbeddingDisjointEquivSigmaEmbeddingRestricted___closed__0);
lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__0 = _init_lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__0();
lean_mark_persistent(lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__0);
lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__1 = _init_lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__1();
lean_mark_persistent(lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__1);
lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__2 = _init_lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__2();
lean_mark_persistent(lp_mathlib_Equiv_sumEmbeddingEquivSigmaEmbeddingRestricted___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
