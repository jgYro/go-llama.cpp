.PHONY: test clean llama-build

INCLUDE_PATH := $(abspath ./)
LIBRARY_PATH := $(abspath ./)

UNAME_S ?= $(shell uname -s)
UNAME_M ?= $(shell uname -m)

BUILD_TYPE ?=
BUILD_DIR ?= build

CFLAGS   := -I./llama.cpp/include -I./llama.cpp/ggml/include -I. -O3 -DNDEBUG -std=c11 -fPIC
CXXFLAGS := -I./llama.cpp/include -I./llama.cpp/ggml/include -I./llama.cpp/ggml/src -I. -O3 -DNDEBUG -std=c++17 -fPIC
LDFLAGS  :=

CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function

ifneq ($(filter $(UNAME_S),Linux Darwin FreeBSD NetBSD OpenBSD Haiku),)
	CFLAGS   += -pthread
	CXXFLAGS += -pthread
endif

ifeq ($(UNAME_M),$(filter $(UNAME_M),x86_64 i686))
	CFLAGS   += -march=native -mtune=native
	CXXFLAGS += -march=native -mtune=native
endif

ifneq ($(filter aarch64% arm64%,$(UNAME_M)),)
	CFLAGS   += -mcpu=native
	CXXFLAGS += -mcpu=native
endif

CMAKE_ARGS := \
	-DCMAKE_BUILD_TYPE=Release \
	-DBUILD_SHARED_LIBS=OFF \
	-DLLAMA_BUILD_COMMON=OFF \
	-DLLAMA_BUILD_TESTS=OFF \
	-DLLAMA_BUILD_EXAMPLES=OFF \
	-DLLAMA_BUILD_SERVER=OFF \
	-DLLAMA_BUILD_TOOLS=OFF \
	-DLLAMA_BUILD_APP=OFF

ifeq ($(UNAME_S),Darwin)
	CMAKE_ARGS += -DGGML_METAL=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Apple
	LDFLAGS += -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
endif

ifeq ($(BUILD_TYPE),metal)
	CMAKE_ARGS += -DGGML_METAL=ON
endif

ifeq ($(BUILD_TYPE),openblas)
	CMAKE_ARGS += -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
endif

ifeq ($(BUILD_TYPE),cublas)
	CMAKE_ARGS += -DGGML_CUDA=ON
endif

ifeq ($(BUILD_TYPE),hipblas)
	CMAKE_ARGS += -DGGML_HIP=ON
endif

ifeq ($(BUILD_TYPE),clblas)
	CMAKE_ARGS += -DGGML_CLBLAST=ON
endif

ifeq ($(GPU_TESTS),true)
	TEST_LABEL=gpu
else
	TEST_LABEL=!gpu
endif

$(info I llama.cpp build info:)
$(info I UNAME_S:    $(UNAME_S))
$(info I UNAME_M:    $(UNAME_M))
$(info I CFLAGS:     $(CFLAGS))
$(info I CXXFLAGS:   $(CXXFLAGS))
$(info I LDFLAGS:    $(LDFLAGS))
$(info I BUILD_TYPE: $(BUILD_TYPE))
$(info I CMAKE_ARGS: $(CMAKE_ARGS))
$(info )

llama-build:
	cmake -S llama.cpp -B $(BUILD_DIR) $(CMAKE_ARGS)
	cmake --build $(BUILD_DIR) --config Release --target llama

binding.o: binding.cpp binding.h llama.cpp/include/llama.h
	$(CXX) $(CXXFLAGS) binding.cpp -o binding.o -c

libbinding.a: llama-build binding.o
	@libs="$(BUILD_DIR)/src/libllama.a $(BUILD_DIR)/ggml/src/libggml.a $(BUILD_DIR)/ggml/src/libggml-base.a $(BUILD_DIR)/ggml/src/libggml-cpu.a $$(find "$(BUILD_DIR)/ggml/src" -mindepth 2 -name 'libggml*.a' | sort)"; \
	if [ -z "$$libs" ]; then \
		echo "no llama.cpp static libraries found under $(BUILD_DIR)"; \
		exit 1; \
	fi; \
	if [ "$(UNAME_S)" = "Darwin" ]; then \
		libtool -static -o libbinding.a binding.o $$libs; \
	else \
		rm -rf "$(BUILD_DIR)/libbinding-objects"; \
		mkdir -p "$(BUILD_DIR)/libbinding-objects/binding"; \
		cp binding.o "$(BUILD_DIR)/libbinding-objects/binding/binding.o"; \
		root="$$(pwd)"; \
		i=0; \
		for lib in $$libs; do \
			i=$$((i + 1)); \
			mkdir -p "$(BUILD_DIR)/libbinding-objects/lib$$i"; \
			(cd "$(BUILD_DIR)/libbinding-objects/lib$$i" && ar x "$$root/$$lib"); \
		done; \
		ar rcs libbinding.a $$(find "$(BUILD_DIR)/libbinding-objects" -name '*.o'); \
	fi

clean:
	rm -rf *.o *.a $(BUILD_DIR)

ggllm-test-model.bin:
	wget -q https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q2_K.gguf -O ggllm-test-model.bin

test: ggllm-test-model.bin libbinding.a
	C_INCLUDE_PATH=${INCLUDE_PATH} CGO_LDFLAGS="${CGO_LDFLAGS}" LIBRARY_PATH=${LIBRARY_PATH} TEST_MODEL=ggllm-test-model.bin go run github.com/onsi/ginkgo/v2/ginkgo --label-filter="$(TEST_LABEL)" --flake-attempts 5 -v -r ./...
