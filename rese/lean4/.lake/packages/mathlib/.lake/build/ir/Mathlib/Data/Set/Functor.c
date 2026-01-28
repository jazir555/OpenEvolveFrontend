// Lean compiler output
// Module: Mathlib.Data.Set.Functor
// Imports: public import Init public import Batteries.Control.AlternativeMonad public import Mathlib.Control.Basic public import Mathlib.Data.Set.Defs public import Mathlib.Data.Set.Lattice.Image public import Mathlib.Data.Set.Notation
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
static lean_object* lp_mathlib_Set_monad___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Set_instAlternative;
LEAN_EXPORT lean_object* lp_mathlib_instAlternativeMonadSetM;
static lean_object* lp_mathlib_Set_instAlternative___closed__0;
static lean_object* lp_mathlib_Set_instAlternative___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Set_monad;
static lean_object* _init_lp_mathlib_Set_instAlternative___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_1, 0, lean_box(0));
lean_ctor_set(x_1, 1, lean_box(0));
x_2 = lean_alloc_ctor(0, 5, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
lean_ctor_set(x_2, 2, lean_box(0));
lean_ctor_set(x_2, 3, lean_box(0));
lean_ctor_set(x_2, 4, lean_box(0));
return x_2;
}
}
static lean_object* _init_lp_mathlib_Set_instAlternative___closed__1() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Set_instAlternative___closed__0;
x_2 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
lean_ctor_set(x_2, 2, lean_box(0));
return x_2;
}
}
static lean_object* _init_lp_mathlib_Set_instAlternative() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Set_instAlternative___closed__1;
return x_1;
}
}
static lean_object* _init_lp_mathlib_Set_monad___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Set_instAlternative;
x_2 = lean_alloc_ctor(0, 2, 0);
lean_ctor_set(x_2, 0, x_1);
lean_ctor_set(x_2, 1, lean_box(0));
return x_2;
}
}
static lean_object* _init_lp_mathlib_Set_monad() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Set_monad___closed__0;
return x_1;
}
}
static lean_object* _init_lp_mathlib_instAlternativeMonadSetM() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Set_monad;
return x_1;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_batteries_Batteries_Control_AlternativeMonad(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Control_Basic(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Defs(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Lattice_Image(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Data_Set_Notation(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Data_Set_Functor(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_batteries_Batteries_Control_AlternativeMonad(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Control_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Defs(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Lattice_Image(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Data_Set_Notation(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Set_instAlternative___closed__0 = _init_lp_mathlib_Set_instAlternative___closed__0();
lean_mark_persistent(lp_mathlib_Set_instAlternative___closed__0);
lp_mathlib_Set_instAlternative___closed__1 = _init_lp_mathlib_Set_instAlternative___closed__1();
lean_mark_persistent(lp_mathlib_Set_instAlternative___closed__1);
lp_mathlib_Set_instAlternative = _init_lp_mathlib_Set_instAlternative();
lean_mark_persistent(lp_mathlib_Set_instAlternative);
lp_mathlib_Set_monad___closed__0 = _init_lp_mathlib_Set_monad___closed__0();
lean_mark_persistent(lp_mathlib_Set_monad___closed__0);
lp_mathlib_Set_monad = _init_lp_mathlib_Set_monad();
lean_mark_persistent(lp_mathlib_Set_monad);
lp_mathlib_instAlternativeMonadSetM = _init_lp_mathlib_instAlternativeMonadSetM();
lean_mark_persistent(lp_mathlib_instAlternativeMonadSetM);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
