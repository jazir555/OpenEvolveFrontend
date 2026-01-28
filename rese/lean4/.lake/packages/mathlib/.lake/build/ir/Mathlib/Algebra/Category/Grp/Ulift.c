// Lean compiler output
// Module: Mathlib.Algebra.Category.Grp.Ulift
// Imports: public import Init public import Mathlib.Algebra.Category.Grp.LargeColimits public import Mathlib.Algebra.Category.Grp.Limits public import Mathlib.Algebra.Module.CharacterModule public import Mathlib.CategoryTheory.Limits.Preserves.Ulift
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
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_uliftFunctorFullyFaithful;
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful;
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_uliftFunctorFullyFaithful___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_mathlib_Equiv_ulift(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00GrpCat_uliftFunctorFullyFaithful_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg___boxed(lean_object*);
lean_object* lp_mathlib_Equiv_symm___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00GrpCat_uliftFunctorFullyFaithful_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__2___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful;
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful;
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_uliftFunctorFullyFaithful___lam__0(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__2___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00GrpCat_uliftFunctorFullyFaithful_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__2(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__0___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__0(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1___redArg___boxed(lean_object*);
static lean_object* lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg___boxed(lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3___boxed(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg(lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lp_mathlib_Equiv_ulift(lean_box(0));
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_apply_1(x_1, x_3);
x_5 = lean_apply_1(x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_uliftFunctorFullyFaithful___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_5 = lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0(x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0(x_1);
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg(x_3, x_9);
x_11 = lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5___redArg(x_6, x_10);
x_12 = lean_apply_1(x_11, x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_GrpCat_uliftFunctorFullyFaithful___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_GrpCat_uliftFunctorFullyFaithful___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_GrpCat_uliftFunctorFullyFaithful() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_GrpCat_uliftFunctorFullyFaithful___lam__0___boxed), 4, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00GrpCat_uliftFunctorFullyFaithful_spec__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_symm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00GrpCat_uliftFunctorFullyFaithful_spec__2___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__5(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00GrpCat_uliftFunctorFullyFaithful_spec__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulEquiv_symm___at___00GrpCat_uliftFunctorFullyFaithful_spec__2(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__1___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_toMonoidHom___at___00GrpCat_uliftFunctorFullyFaithful_spec__3___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_symm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__2___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_5 = lp_mathlib_AddEquiv_ulift___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__0(x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_AddEquiv_ulift___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__0(x_1);
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(x_3, x_9);
x_11 = lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(x_6, x_10);
x_12 = lean_apply_1(x_11, x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful___lam__0___boxed), 4, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddGrpCat_ofHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__6___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddEquiv_symm___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__2(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddEquiv_ulift___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__4(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddMonoidHom_comp___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__5(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_5 = lp_mathlib_MulEquiv_ulift___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__0(x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_MulEquiv_ulift___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__0(x_1);
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(x_3, x_9);
x_11 = lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(x_6, x_10);
x_12 = lean_apply_1(x_11, x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful___lam__0___boxed), 4, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_symm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__2___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__4(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_MonoidHom_comp___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__5(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_symm___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_MulEquiv_symm___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__2(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_ulift___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_ulift___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_toMonoidHom___at___00CommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_inc(x_3);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6___redArg(lean_object* x_1) {
_start:
{
lean_inc(x_1);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__0(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0;
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lean_alloc_closure((void*)(lp_mathlib_MonoidHom_comp___at___00GrpCat_uliftFunctorFullyFaithful_spec__4___redArg___lam__0), 3, 2);
lean_closure_set(x_3, 0, x_2);
lean_closure_set(x_3, 1, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; 
x_5 = lp_mathlib_AddEquiv_ulift___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__0(x_2);
x_6 = lean_ctor_get(x_5, 0);
lean_inc(x_6);
lean_dec_ref(x_5);
x_7 = lp_mathlib_AddEquiv_ulift___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__0(x_1);
x_8 = lp_mathlib_Equiv_symm___redArg(x_7);
x_9 = lean_ctor_get(x_8, 0);
lean_inc(x_9);
lean_dec_ref(x_8);
x_10 = lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(x_3, x_9);
x_11 = lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(x_6, x_10);
x_12 = lean_apply_1(x_11, x_4);
return x_12;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful___lam__0(x_1, x_2, x_3, x_4);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
}
static lean_object* _init_lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful___lam__0___boxed), 4, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__2(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_Equiv_symm___redArg(x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__2___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_Equiv_symm___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5___redArg(x_4, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lean_ctor_get(x_3, 0);
lean_inc(x_4);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc(x_2);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6(x_1, x_2, x_3);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddCommGrpCat_ofHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__6___redArg(x_1);
lean_dec(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_symm___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__2___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_mathlib_AddEquiv_symm___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__2(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_ulift___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__0___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddEquiv_ulift___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__0(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__4(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_AddMonoidHom_comp___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__5(x_1, x_2, x_3, x_4, x_5);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__1___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3(x_1, x_2, x_3);
lean_dec_ref(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_mathlib_AddEquiv_toAddMonoidHom___at___00AddCommGrpCat_uliftFunctorFullyFaithful_spec__3___redArg(x_1);
lean_dec_ref(x_1);
return x_2;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_LargeColimits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_Limits(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Algebra_Module_CharacterModule(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Ulift(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Algebra_Category_Grp_Ulift(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Grp_LargeColimits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Category_Grp_Limits(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Algebra_Module_CharacterModule(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_CategoryTheory_Limits_Preserves_Ulift(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0 = _init_lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0();
lean_mark_persistent(lp_mathlib_MulEquiv_ulift___at___00GrpCat_uliftFunctorFullyFaithful_spec__0___closed__0);
lp_mathlib_GrpCat_uliftFunctorFullyFaithful = _init_lp_mathlib_GrpCat_uliftFunctorFullyFaithful();
lean_mark_persistent(lp_mathlib_GrpCat_uliftFunctorFullyFaithful);
lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful = _init_lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful();
lean_mark_persistent(lp_mathlib_AddGrpCat_uliftFunctorFullyFaithful);
lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful = _init_lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful();
lean_mark_persistent(lp_mathlib_CommGrpCat_uliftFunctorFullyFaithful);
lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful = _init_lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful();
lean_mark_persistent(lp_mathlib_AddCommGrpCat_uliftFunctorFullyFaithful);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
