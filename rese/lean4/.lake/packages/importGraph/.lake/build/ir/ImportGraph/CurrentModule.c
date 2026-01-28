// Lean compiler output
// Module: ImportGraph.CurrentModule
// Imports: public import Init public import Lake.Load.Manifest
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
lean_object* l_Lake_Manifest_load_x3f(lean_object*);
uint8_t l_Lean_Name_isPrefixOf(lean_object*, lean_object*);
static lean_object* lp_importGraph_ImportGraph_getCurrentModule___closed__0;
lean_object* l_Lean_Name_capitalize(lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_ImportGraph_getCurrentModule();
LEAN_EXPORT lean_object* lp_importGraph_ImportGraph_isInModule___boxed(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_importGraph_ImportGraph_getCurrentModule___boxed(lean_object*);
LEAN_EXPORT uint8_t lp_importGraph_ImportGraph_isInModule(lean_object*, lean_object*);
static lean_object* _init_lp_importGraph_ImportGraph_getCurrentModule___closed__0() {
_start:
{
lean_object* x_1; 
x_1 = lean_mk_string_unchecked("lake-manifest.json", 18, 18);
return x_1;
}
}
LEAN_EXPORT lean_object* lp_importGraph_ImportGraph_getCurrentModule() {
_start:
{
lean_object* x_2; lean_object* x_3; 
x_2 = lp_importGraph_ImportGraph_getCurrentModule___closed__0;
x_3 = l_Lake_Manifest_load_x3f(x_2);
if (lean_obj_tag(x_3) == 0)
{
uint8_t x_4; 
x_4 = !lean_is_exclusive(x_3);
if (x_4 == 0)
{
lean_object* x_5; 
x_5 = lean_ctor_get(x_3, 0);
if (lean_obj_tag(x_5) == 0)
{
lean_object* x_6; 
x_6 = lean_box(0);
lean_ctor_set(x_3, 0, x_6);
return x_3;
}
else
{
lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_7 = lean_ctor_get(x_5, 0);
lean_inc(x_7);
lean_dec_ref(x_5);
x_8 = lean_ctor_get(x_7, 0);
lean_inc(x_8);
lean_dec(x_7);
x_9 = l_Lean_Name_capitalize(x_8);
lean_ctor_set(x_3, 0, x_9);
return x_3;
}
}
else
{
lean_object* x_10; 
x_10 = lean_ctor_get(x_3, 0);
lean_inc(x_10);
lean_dec(x_3);
if (lean_obj_tag(x_10) == 0)
{
lean_object* x_11; lean_object* x_12; 
x_11 = lean_box(0);
x_12 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_12, 0, x_11);
return x_12;
}
else
{
lean_object* x_13; lean_object* x_14; lean_object* x_15; lean_object* x_16; 
x_13 = lean_ctor_get(x_10, 0);
lean_inc(x_13);
lean_dec_ref(x_10);
x_14 = lean_ctor_get(x_13, 0);
lean_inc(x_14);
lean_dec(x_13);
x_15 = l_Lean_Name_capitalize(x_14);
x_16 = lean_alloc_ctor(0, 1, 0);
lean_ctor_set(x_16, 0, x_15);
return x_16;
}
}
}
else
{
uint8_t x_17; 
x_17 = !lean_is_exclusive(x_3);
if (x_17 == 0)
{
return x_3;
}
else
{
lean_object* x_18; lean_object* x_19; 
x_18 = lean_ctor_get(x_3, 0);
lean_inc(x_18);
lean_dec(x_3);
x_19 = lean_alloc_ctor(1, 1, 0);
lean_ctor_set(x_19, 0, x_18);
return x_19;
}
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_ImportGraph_getCurrentModule___boxed(lean_object* x_1) {
_start:
{
lean_object* x_2; 
x_2 = lp_importGraph_ImportGraph_getCurrentModule();
return x_2;
}
}
LEAN_EXPORT uint8_t lp_importGraph_ImportGraph_isInModule(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_1) == 0)
{
uint8_t x_3; 
x_3 = 0;
return x_3;
}
else
{
lean_object* x_4; uint8_t x_5; 
x_4 = lean_ctor_get(x_1, 0);
x_5 = l_Lean_Name_isPrefixOf(x_4, x_2);
return x_5;
}
}
}
LEAN_EXPORT lean_object* lp_importGraph_ImportGraph_isInModule___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
uint8_t x_3; lean_object* x_4; 
x_3 = lp_importGraph_ImportGraph_isInModule(x_1, x_2);
lean_dec(x_2);
lean_dec(x_1);
x_4 = lean_box(x_3);
return x_4;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lake_Load_Manifest(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_importGraph_ImportGraph_CurrentModule(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lake_Load_Manifest(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_importGraph_ImportGraph_getCurrentModule___closed__0 = _init_lp_importGraph_ImportGraph_getCurrentModule___closed__0();
lean_mark_persistent(lp_importGraph_ImportGraph_getCurrentModule___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
