// Lean compiler output
// Module: Mathlib.Analysis.Normed.Group.Submodule
// Imports: public import Init public import Mathlib.Algebra.Module.Submodule.LinearMap public import Mathlib.Analysis.Normed.Group.Basic
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
LEAN_EXPORT lean_object* lp_mathlib_Submodule_normedAddCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_AddSubgroupClass_toAddGroup___redArg(lean_object*);
lean_object* lp_mathlib_LinearMap_toAddMonoidHom___redArg(lean_object*);
lean_object* lp_mathlib_SeminormedAddCommGroup_toSeminormedAddGroup___redArg(lean_object*);
static lean_object* lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__1;
lean_object* lp_mathlib_SeminormedAddCommGroup_induced___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__2;
LEAN_EXPORT lean_object* lp_mathlib_Submodule_normedAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_seminormedAddCommGroup___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_NormedAddCommGroup_toSeminormedAddCommGroup___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_seminormedAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_normedAddCommGroup(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Submodule_seminormedAddCommGroup___redArg(lean_object*);
static lean_object* lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__0;
lean_object* lp_mathlib_MonoidHom_instFunLike___lam__0(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_instFunLike___lam__0), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_SMulMemClass_subtype___lam__0___boxed), 1, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__1;
x_2 = lp_mathlib_LinearMap_toAddMonoidHom___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_seminormedAddCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_2 = lean_ctor_get(x_1, 1);
x_3 = lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__0;
lean_inc_ref(x_2);
x_4 = lp_mathlib_AddSubgroupClass_toAddGroup___redArg(x_2);
x_5 = lp_mathlib_SeminormedAddCommGroup_toSeminormedAddGroup___redArg(x_1);
x_6 = lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__2;
x_7 = lp_mathlib_SeminormedAddCommGroup_induced___redArg(x_3, x_4, x_5, x_6);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_seminormedAddCommGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Submodule_seminormedAddCommGroup___redArg(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_seminormedAddCommGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Submodule_seminormedAddCommGroup(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_normedAddCommGroup___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; uint8_t x_4; 
x_2 = lp_mathlib_NormedAddCommGroup_toSeminormedAddCommGroup___redArg(x_1);
x_3 = lp_mathlib_Submodule_seminormedAddCommGroup___redArg(x_2);
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
return x_3;
}
else
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_5 = lean_ctor_get(x_3, 0);
x_6 = lean_ctor_get(x_3, 1);
x_7 = lean_ctor_get(x_3, 2);
lean_inc(x_7);
lean_inc(x_6);
lean_inc(x_5);
lean_dec(x_3);
x_8 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_8, 0, x_5);
lean_ctor_set(x_8, 1, x_6);
lean_ctor_set(x_8, 2, x_7);
return x_8;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_normedAddCommGroup(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Submodule_normedAddCommGroup___redArg(x_4);
return x_7;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Submodule_normedAddCommGroup___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_mathlib_Submodule_normedAddCommGroup(x_1, x_2, x_3, x_4, x_5, x_6);
lean_dec(x_5);
lean_dec_ref(x_3);
return x_7;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_Submodule_LinearMap(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Basic(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Analysis_Normed_Group_Submodule(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_Submodule_LinearMap(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Analysis_Normed_Group_Basic(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__0 = _init_lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__0);
lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__1 = _init_lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__1);
lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__2 = _init_lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__2();
lean_mark_persistent(lp_mathlib_Submodule_seminormedAddCommGroup___redArg___closed__2);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
