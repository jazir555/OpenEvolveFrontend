// Lean compiler output
// Module: Mathlib.AlgebraicTopology.SimplicialSet.NonDegenerateSimplices
// Imports: public import Init public import Mathlib.AlgebraicTopology.SimplicialSet.Degenerate public import Mathlib.AlgebraicTopology.SimplicialSet.Simplices
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
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_instPreorder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_orderEmbeddingN___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_instPreorder___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_mk(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_SSet_S_cast___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_instPartialOrder(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_cast___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subfunctor_ofSection___at___00SSet_orderEmbeddingN_spec__0___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_orderEmbeddingN(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_cast___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_mk___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_instPartialOrder___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_orderEmbeddingN___lam__0___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_mk___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subfunctor_ofSection___at___00SSet_orderEmbeddingN_spec__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_cast(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_SSet_N_instPreorder___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_mk(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_5, 0, x_2);
lean_ctor_set(x_5, 1, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_mk___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_1);
lean_ctor_set(x_3, 1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_mk___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SSet_N_mk(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_SSet_N_instPreorder___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lean_box(0);
x_2 = lean_box(0);
x_3 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_3, 0, x_2);
lean_ctor_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_instPreorder(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SSet_N_instPreorder___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_instPreorder___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SSet_N_instPreorder(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_instPartialOrder(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SSet_N_instPreorder(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_instPartialOrder___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_SSet_N_instPartialOrder(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_cast(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SSet_S_cast___redArg(x_2, x_3);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_cast___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SSet_S_cast___redArg(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_N_cast___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_SSet_N_cast(x_1, x_2, x_3, x_4);
lean_dec_ref(x_1);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subfunctor_ofSection___at___00SSet_orderEmbeddingN_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_box(0);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_orderEmbeddingN___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lean_ctor_get(x_2, 0);
x_4 = lean_ctor_get(x_2, 1);
x_5 = lp_mathlib_CategoryTheory_Subfunctor_ofSection___at___00SSet_orderEmbeddingN_spec__0(x_1, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_orderEmbeddingN___lam__0___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_SSet_orderEmbeddingN___lam__0(x_1, x_2);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_SSet_orderEmbeddingN(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_SSet_orderEmbeddingN___lam__0___boxed), 2, 1);
lean_closure_set(x_2, 0, x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CategoryTheory_Subfunctor_ofSection___at___00SSet_orderEmbeddingN_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_CategoryTheory_Subfunctor_ofSection___at___00SSet_orderEmbeddingN_spec__0(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_Degenerate(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_Simplices(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_NonDegenerateSimplices(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_Degenerate(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_AlgebraicTopology_SimplicialSet_Simplices(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_SSet_N_instPreorder___closed__0 = _init_lp_mathlib_SSet_N_instPreorder___closed__0();
lean_mark_persistent(lp_mathlib_SSet_N_instPreorder___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
