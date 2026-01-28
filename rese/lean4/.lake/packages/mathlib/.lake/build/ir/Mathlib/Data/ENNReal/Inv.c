// Lean compiler output
// Module: Mathlib.Data.ENNReal.Inv
// Imports: public import Init public import Mathlib.Data.ENNReal.Operations
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
lean_object* lp_mathlib_ENNReal_toNNReal(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENNReal_orderIsoIicCoe___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENNReal_orderIsoIicCoe(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
static lean_object* lp_mathlib_ENNReal_orderIsoIicCoe___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_ENNReal_orderIsoIicCoe___lam__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_ENNReal_orderIsoIicCoe___lam__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_mathlib_ENNReal_orderIsoIicCoe___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_ENNReal_toNNReal), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENNReal_orderIsoIicCoe(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; 
x_2 = lean_alloc_closure((void*)(lp_mathlib_ENNReal_orderIsoIicCoe___lam__0), 1, 0);
x_3 = lp_mathlib_ENNReal_orderIsoIicCoe___closed__0;
x_4 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_3);
x_5 = lp_mathlib_Equiv_symm___redArg(x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_ENNReal_orderIsoIicCoe___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_ENNReal_orderIsoIicCoe(x_1);
lean_dec(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_ENNReal_Operations(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_ENNReal_Inv(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_ENNReal_Operations(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_ENNReal_orderIsoIicCoe___closed__0 = _init_lp_mathlib_ENNReal_orderIsoIicCoe___closed__0();
lean_mark_persistent(lp_mathlib_ENNReal_orderIsoIicCoe___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
