// Lean compiler output
// Module: Mathlib.LinearAlgebra.Projection
// Imports: public import Init public import Mathlib.LinearAlgebra.Quotient.Basic public import Mathlib.LinearAlgebra.Prod public import Mathlib.Algebra.Module.Submodule.Invariant public import Mathlib.LinearAlgebra.GeneralLinearGroup.Basic public import Mathlib.Algebra.Ring.Idempotent
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
lean_object* lp_mathlib_SMulMemClass_subtype___lam__0___boxed(lean_object*);
lean_object* lp_mathlib_LinearMap_comp___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0(lean_object*, lean_object*);
lean_object* lp_mathlib_LinearMap_codRestrict___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_SMulMemClass_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_3 = lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0;
x_4 = lp_mathlib_LinearMap_comp___redArg(x_3, x_1);
x_5 = lean_apply_1(x_4, x_2);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_LinearMap_codRestrict___redArg), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0;
x_8 = lean_alloc_closure((void*)(lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0), 2, 0);
x_9 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_9, 0, x_7);
lean_ctor_set(x_9, 1, x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_isIdempotentElemEquiv___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Submodule_isIdempotentElemEquiv(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_LinearMap_codRestrict___redArg(x_7);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_LinearMap_codRestrict___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_LinearMap_IsProj_codRestrict___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6, lean_object* x_7, lean_object* x_8) {
_start:
{
lean_object* x_9; 
x_9 = lp_mathlib_LinearMap_IsProj_codRestrict(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8);
lean_dec(x_5);
lean_dec_ref(x_4);
lean_dec_ref(x_2);
return x_9;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Prod(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_Invariant(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Ring_Idempotent(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_LinearAlgebra_Projection(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Quotient_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_Prod(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_Invariant(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_LinearAlgebra_GeneralLinearGroup_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Ring_Idempotent(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0 = _init_lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Submodule_isIdempotentElemEquiv___lam__0___closed__0);
lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0 = _init_lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0();
lean_mark_persistent(lp_mathlib_Submodule_isIdempotentElemEquiv___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
