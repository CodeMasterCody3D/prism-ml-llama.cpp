// TAARDIS: ternary k1 KV pair (hand-added; generate_cu_files.py does not know custom types).

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q1_T_g128, GGML_TYPE_Q1_T_g128);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q1_T_g128, GGML_TYPE_Q1_T_g128);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q1_T_g128, GGML_TYPE_Q1_T_g128);
