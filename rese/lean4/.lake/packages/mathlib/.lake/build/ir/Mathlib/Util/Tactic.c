// Lean compiler output
// Module: Mathlib.Util.Tactic
// Imports: public import Init public import Mathlib.Init public meta import Lean.MetavarContext
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
lean_object* l_Lean_PersistentHashMap_insert___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentHashMap_find_x3f___redArg(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___lam__0(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instHashableFVarId_hash___boxed(lean_object*);
lean_object* lean_local_ctx_find(lean_object*, lean_object*);
lean_object* l_Lean_instBEqFVarId_beq___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext(lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_PersistentArray_set___redArg(lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0(lean_object*, lean_object*, lean_object*);
lean_object* l_Lean_instHashableMVarId_hash___boxed(lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1;
static lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0;
lean_object* l_Lean_instBEqMVarId_beq___boxed(lean_object*, lean_object*);
static lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0;
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___redArg___lam__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget(lean_object*, lean_object*, lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(lean_object*, lean_object*, lean_object*);
static lean_object* _init_lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_instBEqMVarId_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_instHashableMVarId_hash___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_6 = lean_ctor_get(x_5, 0);
x_7 = lean_ctor_get(x_5, 1);
x_8 = lean_ctor_get(x_5, 2);
x_9 = lean_ctor_get(x_5, 3);
x_10 = lean_ctor_get(x_5, 4);
x_11 = lean_ctor_get(x_5, 5);
x_12 = lean_ctor_get(x_5, 6);
x_13 = lean_ctor_get(x_5, 7);
x_14 = lean_ctor_get(x_5, 8);
lean_inc(x_3);
lean_inc_ref(x_10);
lean_inc_ref(x_2);
lean_inc_ref(x_1);
x_15 = l_Lean_PersistentHashMap_find_x3f___redArg(x_1, x_2, x_10, x_3);
if (lean_obj_tag(x_15) == 0)
{
lean_dec_ref(x_4);
lean_dec(x_3);
lean_dec_ref(x_2);
lean_dec_ref(x_1);
return x_5;
}
else
{
uint8_t x_16; 
lean_inc_ref(x_14);
lean_inc_ref(x_13);
lean_inc_ref(x_12);
lean_inc_ref(x_11);
lean_inc_ref(x_10);
lean_inc_ref(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
x_16 = !lean_is_exclusive(x_5);
if (x_16 == 0)
{
lean_object* x_17; lean_object* x_18; lean_object* x_19; lean_object* x_20; lean_object* x_21; lean_object* x_22; lean_object* x_23; lean_object* x_24; lean_object* x_25; lean_object* x_26; lean_object* x_27; lean_object* x_28; 
x_17 = lean_ctor_get(x_5, 8);
lean_dec(x_17);
x_18 = lean_ctor_get(x_5, 7);
lean_dec(x_18);
x_19 = lean_ctor_get(x_5, 6);
lean_dec(x_19);
x_20 = lean_ctor_get(x_5, 5);
lean_dec(x_20);
x_21 = lean_ctor_get(x_5, 4);
lean_dec(x_21);
x_22 = lean_ctor_get(x_5, 3);
lean_dec(x_22);
x_23 = lean_ctor_get(x_5, 2);
lean_dec(x_23);
x_24 = lean_ctor_get(x_5, 1);
lean_dec(x_24);
x_25 = lean_ctor_get(x_5, 0);
lean_dec(x_25);
x_26 = lean_ctor_get(x_15, 0);
lean_inc(x_26);
lean_dec_ref(x_15);
x_27 = lean_apply_1(x_4, x_26);
x_28 = l_Lean_PersistentHashMap_insert___redArg(x_1, x_2, x_10, x_3, x_27);
lean_ctor_set(x_5, 4, x_28);
return x_5;
}
else
{
lean_object* x_29; lean_object* x_30; lean_object* x_31; lean_object* x_32; 
lean_dec(x_5);
x_29 = lean_ctor_get(x_15, 0);
lean_inc(x_29);
lean_dec_ref(x_15);
x_30 = lean_apply_1(x_4, x_29);
x_31 = l_Lean_PersistentHashMap_insert___redArg(x_1, x_2, x_10, x_3, x_30);
x_32 = lean_alloc_ctor(0, 9, 0);
lean_ctor_set(x_32, 0, x_6);
lean_ctor_set(x_32, 1, x_7);
lean_ctor_set(x_32, 2, x_8);
lean_ctor_set(x_32, 3, x_9);
lean_ctor_set(x_32, 4, x_31);
lean_ctor_set(x_32, 5, x_11);
lean_ctor_set(x_32, 6, x_12);
lean_ctor_set(x_32, 7, x_13);
lean_ctor_set(x_32, 8, x_14);
return x_32;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; 
x_4 = lean_ctor_get(x_1, 1);
lean_inc(x_4);
lean_dec_ref(x_1);
x_5 = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0;
x_6 = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1;
x_7 = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___lam__0), 5, 4);
lean_closure_set(x_7, 0, x_5);
lean_closure_set(x_7, 1, x_6);
lean_closure_set(x_7, 2, x_2);
lean_closure_set(x_7, 3, x_3);
x_8 = lean_apply_1(x_4, x_7);
return x_8;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyMetavarDecl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_2, 2);
x_5 = lean_apply_1(x_1, x_4);
lean_ctor_set(x_2, 2, x_5);
return x_2;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_2, 2);
x_9 = lean_ctor_get(x_2, 3);
x_10 = lean_ctor_get(x_2, 4);
x_11 = lean_ctor_get_uint8(x_2, sizeof(void*)*7);
x_12 = lean_ctor_get(x_2, 5);
x_13 = lean_ctor_get(x_2, 6);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_2);
x_14 = lean_apply_1(x_1, x_8);
x_15 = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(x_15, 0, x_6);
lean_ctor_set(x_15, 1, x_7);
lean_ctor_set(x_15, 2, x_14);
lean_ctor_set(x_15, 3, x_9);
lean_ctor_set(x_15, 4, x_10);
lean_ctor_set(x_15, 5, x_12);
lean_ctor_set(x_15, 6, x_13);
lean_ctor_set_uint8(x_15, sizeof(void*)*7, x_11);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyTarget___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(x_1, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyTarget(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Mathlib_Tactic_modifyTarget___redArg(x_2, x_3, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg___lam__0(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; 
x_3 = !lean_is_exclusive(x_2);
if (x_3 == 0)
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_ctor_get(x_2, 1);
x_5 = lean_apply_1(x_1, x_4);
lean_ctor_set(x_2, 1, x_5);
return x_2;
}
else
{
lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; lean_object* x_10; uint8_t x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; 
x_6 = lean_ctor_get(x_2, 0);
x_7 = lean_ctor_get(x_2, 1);
x_8 = lean_ctor_get(x_2, 2);
x_9 = lean_ctor_get(x_2, 3);
x_10 = lean_ctor_get(x_2, 4);
x_11 = lean_ctor_get_uint8(x_2, sizeof(void*)*7);
x_12 = lean_ctor_get(x_2, 5);
x_13 = lean_ctor_get(x_2, 6);
lean_inc(x_13);
lean_inc(x_12);
lean_inc(x_10);
lean_inc(x_9);
lean_inc(x_8);
lean_inc(x_7);
lean_inc(x_6);
lean_dec(x_2);
x_14 = lean_apply_1(x_1, x_7);
x_15 = lean_alloc_ctor(0, 7, 1);
lean_ctor_set(x_15, 0, x_6);
lean_ctor_set(x_15, 1, x_14);
lean_ctor_set(x_15, 2, x_8);
lean_ctor_set(x_15, 3, x_9);
lean_ctor_set(x_15, 4, x_10);
lean_ctor_set(x_15, 5, x_12);
lean_ctor_set(x_15, 6, x_13);
lean_ctor_set_uint8(x_15, sizeof(void*)*7, x_11);
return x_15;
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; 
x_4 = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg___lam__0), 2, 1);
lean_closure_set(x_4, 0, x_3);
x_5 = lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg(x_1, x_2, x_4);
return x_5;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalContext(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; 
x_5 = lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg(x_2, x_3, x_4);
return x_5;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_instBEqFVarId_beq___boxed), 2, 0);
return x_1;
}
}
static lean_object* _init_lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1() {
_start:
{
lean_object* x_1; 
x_1 = lean_alloc_closure((void*)(l_Lean_instHashableFVarId_hash___boxed), 1, 0);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0(lean_object* x_1, lean_object* x_2, lean_object* x_3) {
_start:
{
lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_ctor_get(x_3, 1);
x_6 = lean_ctor_get(x_3, 2);
lean_inc_ref(x_3);
x_7 = lean_local_ctx_find(x_3, x_1);
if (lean_obj_tag(x_7) == 0)
{
lean_dec_ref(x_2);
return x_3;
}
else
{
lean_object* x_8; lean_object* x_9; lean_object* x_10; lean_object* x_11; lean_object* x_12; lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_20; lean_object* x_24; 
lean_inc(x_6);
lean_inc_ref(x_5);
lean_inc_ref(x_4);
if (lean_is_exclusive(x_3)) {
 lean_ctor_release(x_3, 0);
 lean_ctor_release(x_3, 1);
 lean_ctor_release(x_3, 2);
 x_8 = x_3;
} else {
 lean_dec_ref(x_3);
 x_8 = lean_box(0);
}
x_9 = lean_ctor_get(x_7, 0);
lean_inc(x_9);
if (lean_is_exclusive(x_7)) {
 lean_ctor_release(x_7, 0);
 x_10 = x_7;
} else {
 lean_dec_ref(x_7);
 x_10 = lean_box(0);
}
x_11 = lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0;
x_12 = lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1;
x_13 = lean_apply_1(x_2, x_9);
x_24 = lean_ctor_get(x_13, 1);
lean_inc(x_24);
x_20 = x_24;
goto block_23;
block_19:
{
lean_object* x_16; lean_object* x_17; lean_object* x_18; 
if (lean_is_scalar(x_10)) {
 x_16 = lean_alloc_ctor(1, 1, 0);
} else {
 x_16 = x_10;
}
lean_ctor_set(x_16, 0, x_13);
x_17 = l_Lean_PersistentArray_set___redArg(x_5, x_15, x_16);
lean_dec(x_15);
if (lean_is_scalar(x_8)) {
 x_18 = lean_alloc_ctor(0, 3, 0);
} else {
 x_18 = x_8;
}
lean_ctor_set(x_18, 0, x_14);
lean_ctor_set(x_18, 1, x_17);
lean_ctor_set(x_18, 2, x_6);
return x_18;
}
block_23:
{
lean_object* x_21; lean_object* x_22; 
lean_inc_ref(x_13);
x_21 = l_Lean_PersistentHashMap_insert___redArg(x_11, x_12, x_4, x_20, x_13);
x_22 = lean_ctor_get(x_13, 0);
lean_inc(x_22);
x_14 = x_21;
x_15 = x_22;
goto block_19;
}
}
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
lean_object* x_5; lean_object* x_6; 
x_5 = lean_alloc_closure((void*)(lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0), 3, 2);
lean_closure_set(x_5, 0, x_3);
lean_closure_set(x_5, 1, x_4);
x_6 = lp_mathlib_Mathlib_Tactic_modifyLocalContext___redArg(x_1, x_2, x_5);
return x_6;
}
}
LEAN_EXPORT lean_object* lp_mathlib_Mathlib_Tactic_modifyLocalDecl(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4, lean_object* x_5) {
_start:
{
lean_object* x_6; 
x_6 = lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg(x_2, x_3, x_4, x_5);
return x_6;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_mathlib_Mathlib_Init(uint8_t builtin);
lean_object* initialize_Lean_MetavarContext(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_mathlib_Mathlib_Util_Tactic(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_mathlib_Mathlib_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_MetavarContext(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0 = _init_lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__0);
lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1 = _init_lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_modifyMetavarDecl___redArg___closed__1);
lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0 = _init_lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__0);
lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1 = _init_lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1();
lean_mark_persistent(lp_mathlib_Mathlib_Tactic_modifyLocalDecl___redArg___lam__0___closed__1);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
