// Lean compiler output
// Module: ProofWidgets.Demos.ExprPresentation
// Imports: public import Init public meta import ProofWidgets.Component.Panel.SelectionPanel public meta import ProofWidgets.Component.Panel.GoalTypePanel
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
static lean_object* lp_proofwidgets_presenter___lam__0___closed__2;
static lean_object* lp_proofwidgets_presenter___lam__0___closed__8;
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
lean_object* l_Lean_Widget_ppExprTagged(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
extern lean_object* lp_proofwidgets_ProofWidgets_InteractiveCode;
LEAN_EXPORT lean_object* lp_proofwidgets_presenter___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00presenter_spec__0___boxed(lean_object*, lean_object*, lean_object*);
lean_object* lean_array_push(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_presenter;
static lean_object* lp_proofwidgets_presenter___lam__0___closed__3;
static lean_object* lp_proofwidgets_presenter___lam__0___closed__4;
uint64_t lean_string_hash(lean_object*);
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00presenter_spec__0(lean_object*, lean_object*, lean_object*);
static lean_object* lp_proofwidgets_presenter___lam__0___closed__0;
static lean_object* lp_proofwidgets_presenter___lam__0___closed__6;
LEAN_EXPORT lean_object* lp_proofwidgets_presenter___lam__0___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_proofwidgets_presenter___lam__0___closed__5;
static lean_object* lp_proofwidgets_presenter___lam__0___closed__7;
static lean_object* lp_proofwidgets_presenter___closed__0;
static lean_object* lp_proofwidgets_presenter___lam__0___closed__1;
lean_object* l_Lean_PrettyPrinter_Delaborator_delab___boxed(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* lp_proofwidgets_ProofWidgets_instRpcEncodableInteractiveCodeProps_enc_00___x40_ProofWidgets_Component_Basic_1956376046____hygCtx___hyg_1_(lean_object*, lean_object*);
static lean_object* lp_proofwidgets_presenter___lam__0___closed__9;
static lean_object* _init_lp_proofwidgets_presenter___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("With octopodes", 14, 14);
return x_1;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_PrettyPrinter_Delaborator_delab___boxed), 7, 0);
return x_1;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("span", 4, 4);
return x_1;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__2() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(0u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__5() {
_start:
{
lean_object* x_1; 
x_1 = lp_proofwidgets_ProofWidgets_InteractiveCode;
return x_1;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00presenter_spec__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; uint64_t x_7; lean_object* x_8; lean_object* x_9; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = lean_ctor_get(x_1, 1);
x_6 = lean_ctor_get(x_4, 0);
x_7 = lean_string_hash(x_6);
x_8 = lean_alloc_closure((void*)(lp_proofwidgets_ProofWidgets_instRpcEncodableInteractiveCodeProps_enc_00___x40_ProofWidgets_Component_Basic_1956376046____hygCtx___hyg_1_), 2, 1);
lean_closure_set(x_8, 0, x_2);
lean_inc_ref(x_5);
x_9 = lean_alloc_ctor(2, 3, 8);
lean_ctor_set(x_9, 0, x_5);
lean_ctor_set(x_9, 1, x_8);
lean_ctor_set(x_9, 2, x_3);
lean_ctor_set_uint64(x_9, sizeof(void*)*3, x_7);
return x_9;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__6() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked(" 🐙", 5, 2);
return x_1;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__7() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_proofwidgets_presenter___lam__0___closed__6;
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__3() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("🐙 ", 5, 2);
return x_1;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__4() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lp_proofwidgets_presenter___lam__0___closed__3;
x_2 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_2, 0, x_1);
return x_2;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__8() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_unsigned_to_nat(3u);
x_2 = lean_mk_empty_array_with_capacity(x_1);
return x_2;
}
}
static lean_object* _init_lp_proofwidgets_presenter___lam__0___closed__9() {
_start:
{
lean_object* x_1; lean_object* x_2; lean_object* x_3; 
x_1 = lp_proofwidgets_presenter___lam__0___closed__4;
x_2 = lp_proofwidgets_presenter___lam__0___closed__8;
x_3 = lean_array_push(x_2, x_1);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_presenter___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_7; lean_object* x_8; 
x_7 = lp_proofwidgets_presenter___lam__0___closed__0;
x_8 = l_Lean_Widget_ppExprTagged(x_1, x_7, x_2, x_3, x_4, x_5);
if (lean_obj_tag(x_8) == 0)
{
uint8_t x_9; 
x_9 = !lean_is_exclusive(x_8);
if (x_9 == 0)
{
lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; lean_object* x_17; lean_object* x_18; lean_object* x_19; 
x_10 = lean_ctor_get(x_8, 0);
x_11 = lp_proofwidgets_presenter___lam__0___closed__1;
x_12 = lp_proofwidgets_presenter___lam__0___closed__2;
x_13 = lp_proofwidgets_presenter___lam__0___closed__5;
x_14 = lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00presenter_spec__0(x_13, x_10, x_12);
x_15 = lp_proofwidgets_presenter___lam__0___closed__7;
x_16 = lp_proofwidgets_presenter___lam__0___closed__9;
x_17 = lean_array_push(x_16, x_14);
x_18 = lean_array_push(x_17, x_15);
x_19 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_19, 0, x_11);
lean_ctor_set(x_19, 1, x_12);
lean_ctor_set(x_19, 2, x_18);
lean_ctor_set(x_8, 0, x_19);
return x_8;
}
else
{
lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; lean_object* x_29; lean_object* x_30; 
x_20 = lean_ctor_get(x_8, 0);
lean_inc(x_20);
lean_dec(x_8);
x_21 = lp_proofwidgets_presenter___lam__0___closed__1;
x_22 = lp_proofwidgets_presenter___lam__0___closed__2;
x_23 = lp_proofwidgets_presenter___lam__0___closed__5;
x_24 = lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00presenter_spec__0(x_23, x_20, x_22);
x_25 = lp_proofwidgets_presenter___lam__0___closed__7;
x_26 = lp_proofwidgets_presenter___lam__0___closed__9;
x_27 = lean_array_push(x_26, x_24);
x_28 = lean_array_push(x_27, x_25);
x_29 = lean_alloc_ctor(0, 3, 0);
lean_ctor_set(x_29, 0, x_21);
lean_ctor_set(x_29, 1, x_22);
lean_ctor_set(x_29, 2, x_28);
x_30 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_30, 0, x_29);
return x_30;
}
}
else
{
uint8_t x_31; 
x_31 = !lean_is_exclusive(x_8);
if (x_31 == 0)
{
return x_8;
}
else
{
lean_object* x_32; lean_object* x_33; 
x_32 = lean_ctor_get(x_8, 0);
lean_inc(x_32);
lean_dec(x_8);
x_33 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_33, 0, x_32);
return x_33;
}
}
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_presenter___lam__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5, lean_object* x_6) {
_start:
{
lean_object* x_7; 
x_7 = lp_proofwidgets_presenter___lam__0(x_1, x_2, x_3, x_4, x_5);
return x_7;
}
}
static lean_object* _init_lp_proofwidgets_presenter() {
_start:
{
lean_object* x_1; lean_object* x_2; uint8_t x_3; lean_object* x_4; 
x_1 = lean_alloc_closure((void*)(lp_proofwidgets_presenter___lam__0___boxed), 6, 0);
x_2 = lp_proofwidgets_presenter___closed__0;
x_3 = 1;
x_4 = lean_alloc_ctor(0, 2, 1);
lean_ctor_set(x_4, 0, x_2);
lean_ctor_set(x_4, 1, x_1);
lean_ctor_set_uint8(x_4, sizeof(void*)*2, x_3);
return x_4;
}
}
LEAN_EXPORT lean_object* lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00presenter_spec__0___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; 
x_4 = lp_proofwidgets_ProofWidgets_Html_ofComponent___at___00presenter_spec__0(x_1, x_2, x_3);
lean_dec_ref(x_1);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_SelectionPanel(uint8_t builtin);
lean_object* initialize_proofwidgets_ProofWidgets_Component_Panel_GoalTypePanel(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_proofwidgets_ProofWidgets_Demos_ExprPresentation(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Panel_SelectionPanel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_proofwidgets_ProofWidgets_Component_Panel_GoalTypePanel(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_proofwidgets_presenter___closed__0 = _init_lp_proofwidgets_presenter___closed__0();
lean_mark_persistent(lp_proofwidgets_presenter___closed__0);
lp_proofwidgets_presenter___lam__0___closed__0 = _init_lp_proofwidgets_presenter___lam__0___closed__0();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__0);
lp_proofwidgets_presenter___lam__0___closed__1 = _init_lp_proofwidgets_presenter___lam__0___closed__1();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__1);
lp_proofwidgets_presenter___lam__0___closed__2 = _init_lp_proofwidgets_presenter___lam__0___closed__2();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__2);
lp_proofwidgets_presenter___lam__0___closed__5 = _init_lp_proofwidgets_presenter___lam__0___closed__5();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__5);
lp_proofwidgets_presenter___lam__0___closed__6 = _init_lp_proofwidgets_presenter___lam__0___closed__6();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__6);
lp_proofwidgets_presenter___lam__0___closed__7 = _init_lp_proofwidgets_presenter___lam__0___closed__7();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__7);
lp_proofwidgets_presenter___lam__0___closed__3 = _init_lp_proofwidgets_presenter___lam__0___closed__3();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__3);
lp_proofwidgets_presenter___lam__0___closed__4 = _init_lp_proofwidgets_presenter___lam__0___closed__4();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__4);
lp_proofwidgets_presenter___lam__0___closed__8 = _init_lp_proofwidgets_presenter___lam__0___closed__8();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__8);
lp_proofwidgets_presenter___lam__0___closed__9 = _init_lp_proofwidgets_presenter___lam__0___closed__9();
lean_mark_persistent(lp_proofwidgets_presenter___lam__0___closed__9);
lp_proofwidgets_presenter = _init_lp_proofwidgets_presenter();
lean_mark_persistent(lp_proofwidgets_presenter);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
