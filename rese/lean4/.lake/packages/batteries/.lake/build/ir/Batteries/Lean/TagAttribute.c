// Lean compiler output
// Module: Batteries.Lean.TagAttribute
// Imports: public import Init public import Lean.Attributes
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
lean_object* lean_array_push(lean_object*, lean_object*);
static lean_object* lp_batteries_Lean_TagAttribute_getDecls___closed__0;
lean_object* l_Lean_instInhabitedPersistentEnvExtensionState___redArg(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__2(lean_object*, size_t, size_t, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls_core(lean_object*);
lean_object* l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(lean_object*, lean_object*, lean_object*, lean_object*, lean_object*);
lean_object* l_Array_append___redArg(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(lean_object*, lean_object*);
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__2___boxed(lean_object*, lean_object*, lean_object*, lean_object*);
size_t lean_usize_add(size_t, size_t);
lean_object* lean_array_uget(lean_object*, size_t);
size_t lean_array_size(lean_object*);
uint8_t lean_usize_dec_lt(size_t, size_t);
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls___boxed(lean_object*, lean_object*);
lean_object* lean_mk_empty_array_with_capacity(lean_object*);
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
if (lean_obj_tag(x_2) == 0)
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; 
x_3 = lean_ctor_get(x_2, 1);
lean_inc(x_3);
x_4 = lean_ctor_get(x_2, 3);
lean_inc(x_4);
x_5 = lean_ctor_get(x_2, 4);
lean_inc(x_5);
lean_dec_ref(x_2);
x_6 = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(x_1, x_4);
x_7 = lean_array_push(x_6, x_3);
x_1 = x_7;
x_2 = x_5;
goto _start;
}
else
{
return x_1;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(x_1, x_2);
return x_3;
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__2(lean_object* x_1, size_t x_2, size_t x_3, lean_object* x_4) {
_start:
{
uint8_t x_5; 
x_5 = lean_usize_dec_lt(x_3, x_2);
if (x_5 == 0)
{
return x_4;
}
else
{
lean_object* x_6; lean_object* x_7; size_t x_8; size_t x_9; 
x_6 = lean_array_uget(x_1, x_3);
x_7 = l_Array_append___redArg(x_4, x_6);
lean_dec(x_6);
x_8 = 1;
x_9 = lean_usize_add(x_3, x_8);
x_3 = x_9;
x_4 = x_7;
goto _start;
}
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls_core(lean_object* x_1) {
_start:
{
lean_object* x_2; lean_object* x_3; lean_object* x_4; 
x_2 = lean_ctor_get(x_1, 0);
lean_inc_ref(x_2);
x_3 = lean_ctor_get(x_1, 1);
lean_inc(x_3);
lean_dec_ref(x_1);
if (lean_obj_tag(x_3) == 0)
{
lean_object* x_11; 
x_11 = lean_ctor_get(x_3, 0);
lean_inc(x_11);
x_4 = x_11;
goto block_10;
}
else
{
lean_object* x_12; 
x_12 = lean_unsigned_to_nat(0u);
x_4 = x_12;
goto block_10;
}
block_10:
{
lean_object* x_5; lean_object* x_6; size_t x_7; size_t x_8; lean_object* x_9; 
x_5 = lean_mk_empty_array_with_capacity(x_4);
lean_dec(x_4);
x_6 = lp_batteries_Std_DTreeMap_Internal_Impl_foldlM___at___00Std_DTreeMap_Internal_Impl_foldl___at___00Lean_TagAttribute_getDecls_core_spec__0_spec__0(x_5, x_3);
x_7 = lean_array_size(x_2);
x_8 = 0;
x_9 = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__2(x_2, x_7, x_8, x_6);
lean_dec_ref(x_2);
return x_9;
}
}
}
LEAN_EXPORT lean_object* lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__2___boxed(lean_object* x_1, lean_object* x_2, lean_object* x_3, lean_object* x_4) {
_start:
{
size_t x_5; size_t x_6; lean_object* x_7; 
x_5 = lean_unbox_usize(x_2);
lean_dec(x_2);
x_6 = lean_unbox_usize(x_3);
lean_dec(x_3);
x_7 = lp_batteries___private_Init_Data_Array_Basic_0__Array_forIn_x27Unsafe_loop___at___00Lean_TagAttribute_getDecls_core_spec__2(x_1, x_5, x_6, x_4);
lean_dec_ref(x_1);
return x_7;
}
}
static lean_object* _init_lp_batteries_Lean_TagAttribute_getDecls___closed__0() {
_start:
{
lean_object* x_1; lean_object* x_2; 
x_1 = lean_box(1);
x_2 = l_Lean_instInhabitedPersistentEnvExtensionState___redArg(x_1);
return x_2;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; lean_object* x_4; lean_object* x_5; lean_object* x_6; lean_object* x_7; lean_object* x_8; lean_object* x_9; 
x_3 = lean_ctor_get(x_1, 1);
x_4 = lean_ctor_get(x_3, 0);
x_5 = lean_ctor_get(x_4, 2);
x_6 = lp_batteries_Lean_TagAttribute_getDecls___closed__0;
x_7 = lean_box(0);
x_8 = l___private_Lean_Environment_0__Lean_EnvExtension_getStateUnsafe___redArg(x_6, x_4, x_2, x_5, x_7);
x_9 = lp_batteries_Lean_TagAttribute_getDecls_core(x_8);
return x_9;
}
}
LEAN_EXPORT lean_object* lp_batteries_Lean_TagAttribute_getDecls___boxed(lean_object* x_1, lean_object* x_2) {
_start:
{
lean_object* x_3; 
x_3 = lp_batteries_Lean_TagAttribute_getDecls(x_1, x_2);
lean_dec_ref(x_1);
return x_3;
}
}
lean_object* initialize_Init(uint8_t builtin);
lean_object* initialize_Lean_Attributes(uint8_t builtin);
static bool _G_initialized = false;
LEAN_EXPORT lean_object* initialize_batteries_Batteries_Lean_TagAttribute(uint8_t builtin) {
lean_object * res;
if (_G_initialized) return lean_io_result_mk_ok(lean_box(0));
_G_initialized = true;
res = initialize_Init(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
res = initialize_Lean_Attributes(builtin);
if (lean_io_result_is_error(res)) return res;
lean_dec_ref(res);
lp_batteries_Lean_TagAttribute_getDecls___closed__0 = _init_lp_batteries_Lean_TagAttribute_getDecls___closed__0();
lean_mark_persistent(lp_batteries_Lean_TagAttribute_getDecls___closed__0);
return lean_io_result_mk_ok(lean_box(0));
}
#ifdef __cplusplus
}
#endif
