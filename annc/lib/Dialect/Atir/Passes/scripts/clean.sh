#!/bin/bash
PASSES_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# empty generated files from previous run
rm -f ${PASSES_DIR}/prompts/*
rm -f ${PASSES_DIR}/outputs/compile_outputs/*
rm -f ${PASSES_DIR}/outputs/generated_files/*
rm -f ${PASSES_DIR}/outputs/so/*
rm -f ${PASSES_DIR}/outputs/test_outputs/*
rm -f ${PASSES_DIR}/templates/modules/*

mkdir -p ${PASSES_DIR}/prompts
mkdir -p ${PASSES_DIR}/outputs/compile_outputs
mkdir -p ${PASSES_DIR}/outputs/generated_files
mkdir -p ${PASSES_DIR}/outputs/so
mkdir -p ${PASSES_DIR}/outputs/test_outputs
mkdir -p ${PASSES_DIR}/templates/modules