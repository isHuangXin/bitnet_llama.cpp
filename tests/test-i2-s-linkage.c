#include "ggml-quants.h"

#include <stdint.h>

enum {
    ELEMENT_COUNT = 128,
    QUANTIZED_SIZE = 64,
    TEST_FAILURE = 1,
};
static const float QUANTIZATION_SCALE = 1.0f;

// ggml-base delegates platform initialization to the backend linked by applications.
void ggml_bitnet_init(void);
void ggml_bitnet_init(void) {
}

int main(void) {
    float source[ELEMENT_COUNT] = { -QUANTIZATION_SCALE, 0.0f, QUANTIZATION_SCALE };
    uint8_t quantized[QUANTIZED_SIZE];
    float dequantized[ELEMENT_COUNT];

    const size_t written = quantize_i2_s(source, quantized, 1, ELEMENT_COUNT, NULL);
    dequantize_row_i2_s(quantized, dequantized, ELEMENT_COUNT, QUANTIZATION_SCALE);

    if (written != QUANTIZED_SIZE) {
        return TEST_FAILURE;
    }

    return 0;
}
