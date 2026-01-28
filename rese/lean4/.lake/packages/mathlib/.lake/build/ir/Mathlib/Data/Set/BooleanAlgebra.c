// Lean compiler output
// Module: Mathlib.Data.Set.BooleanAlgebra
// Imports: public import Init public import Mathlib.Order.CompleteBooleanAlgebra
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
LEAN_EXPORT lean_object* lp_mathlib_Set_instOrderTop(lean_object*);
lean_object* lp_mathlib_Set_instBooleanAlgebra(lean_object*);
static lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0;
static lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra(lean_object*);
static lean_object* _init_lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Set_instBooleanAlgebra(lean_box(0));
return x_1;
}
}
static lean_object* _init_lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1, 0, lean_box(0));
lean_ctor_set(x_1, 1, lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instCompleteAtomicBooleanAlgebra(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; 
x_2 = lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0;
x_3 = lean_ctor_get(x_2, 0);
lean_inc_ref(x_3);
x_4 = lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1;
x_5 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_5, 0, x_3);
lean_ctor_set(x_5, 1, lean_box(0));
lean_ctor_set(x_5, 2, lean_box(0));
lean_ctor_set(x_5, 3, x_4);
x_6 = lean_alloc_ctor(0, 4, 0);
lean_ctor_set(x_6, 0, x_5);
lean_ctor_set(x_6, 1, lean_box(0));
lean_ctor_set(x_6, 2, lean_box(0));
lean_ctor_set(x_6, 3, lean_box(0));
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Set_instOrderTop(lean_object* x_1) {
_start:
{
return lean_box(0);
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_BooleanAlgebra(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Order_CompleteBooleanAlgebra(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0 = _init_lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0();
lean_mark_persistent(lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__0);
lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1 = _init_lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1();
lean_mark_persistent(lp_mathlib_Set_instCompleteAtomicBooleanAlgebra___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
