#!/bin/bash

###############################################################################
# QAT Phi-2 Complete Execution Pipeline
# 
# the full QAT training and visualization pipeline with
# error handling and progress reporting.
###############################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON_CMD=${PYTHON_CMD:-python3}
VENV_DIR="./venv_qat"
USE_VENV=${USE_VENV:-true}

###############################################################################
# Helper Functions
###############################################################################

print_header() {
    echo ""
    echo "=========================================================================="
    echo "$1"
    echo "=========================================================================="
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed or not in PATH"
        exit 1
    fi
}

###############################################################################
# Environment Setup
###############################################################################

setup_environment() {
    print_header "Setting Up Environment"
    
    # Check Python
    check_command $PYTHON_CMD
    print_success "Python found: $($PYTHON_CMD --version)"
    
    # Check CUDA
    if ! $PYTHON_CMD -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        print_error "CUDA not available. QAT requires GPU."
        exit 1
    fi
    print_success "CUDA available"
    
    # Create virtual environment if requested
    if [ "$USE_VENV" = true ]; then
        if [ ! -d "$VENV_DIR" ]; then
            print_warning "Creating virtual environment..."
            $PYTHON_CMD -m venv $VENV_DIR
            source $VENV_DIR/bin/activate
            
            # Upgrade pip
            pip install --upgrade pip
            
            # Install dependencies
            print_warning "Installing dependencies (this may take a while)..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            pip install transformers datasets torchao accelerate
            pip install matplotlib seaborn tqdm
            
            print_success "Dependencies installed"
        else
            print_success "Virtual environment found"
            source $VENV_DIR/bin/activate
        fi
    fi
    
    # Verify imports
    $PYTHON_CMD -c "
import torch
import transformers
from torchao.quantization.qat import Int8DynActInt4WeightQATQuantizer
print('All imports successful')
" || {
        print_error "Dependency check failed. Please install required packages."
        exit 1
    }
    
    print_success "Environment setup complete"
}

###############################################################################
# Cleanup
###############################################################################

cleanup_previous_runs() {
    print_header "Cleaning Up Previous Runs"
    
    # Remove old weight logs
    if [ -d "./qat_outputs/weight_logs" ]; then
        print_warning "Removing old weight logs..."
        rm -rf ./qat_outputs/weight_logs
    fi
    
    # Remove old visualizations
    if [ -d "./qat_outputs/visualizations" ]; then
        print_warning "Removing old visualizations..."
        rm -rf ./qat_outputs/visualizations
    fi
    
    print_success "Cleanup complete"
}

###############################################################################
# Model Testing
###############################################################################

test_model_loading() {
    print_header "Testing Model Loading"
    
    $PYTHON_CMD -c "
from model import get_phi2_qat_model
print('Loading QAT model...')
model, tokenizer = get_phi2_qat_model()
print('Model loaded successfully')
" || {
        print_error "Model loading failed"
        exit 1
    }
    
    print_success "Model test passed"
}

###############################################################################
# Training
###############################################################################

run_training() {
    print_header "Running QAT Training"
    
    $PYTHON_CMD train.py || {
        print_error "Training failed"
        exit 1
    }
    
    print_success "Training completed"
}

###############################################################################
# Visualization
###############################################################################

run_visualization() {
    print_header "Generating Visualizations"
    
    # Check if weight logs exist
    if [ ! -d "./qat_outputs/weight_logs" ] || [ -z "$(ls -A ./qat_outputs/weight_logs)" ]; then
        print_error "No weight logs found. Training may have failed."
        exit 1
    fi
    
    $PYTHON_CMD viz.py || {
        print_error "Visualization failed"
        exit 1
    }
    
    print_success "Visualizations generated"
}

###############################################################################
# Verification
###############################################################################

verify_outputs() {
    print_header "Verifying Outputs"
    
    # Check for expected files
    local all_good=true
    
    # Training outputs
    if [ -f "./qat_outputs/metrics.json" ]; then
        print_success "Found metrics.json"
    else
        print_error "Missing metrics.json"
        all_good=false
    fi
    
    # Weight logs
    if [ -d "./qat_outputs/weight_logs" ] && [ "$(ls -A ./qat_outputs/weight_logs/*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
        local num_logs=$(ls -1 ./qat_outputs/weight_logs/*.pt 2>/dev/null | wc -l)
        print_success "Found $num_logs weight log files"
    else
        print_error "No weight log files found"
        all_good=false
    fi
    
    # Visualizations
    if [ -d "./qat_outputs/visualizations" ] && [ "$(ls -A ./qat_outputs/visualizations/*.png 2>/dev/null | wc -l)" -gt 0 ]; then
        local num_plots=$(ls -1 ./qat_outputs/visualizations/*.png 2>/dev/null | wc -l)
        print_success "Found $num_plots visualization plots"
    else
        print_error "No visualization plots found"
        all_good=false
    fi
    
    # Report
    if [ -f "./qat_outputs/visualizations/clustering_report.txt" ]; then
        print_success "Found clustering report"
    else
        print_error "Missing clustering report"
        all_good=false
    fi
    
    if [ "$all_good" = false ]; then
        print_error "Some expected outputs are missing"
        exit 1
    fi
    
    print_success "All expected outputs verified"
}

###############################################################################
# Summary
###############################################################################

print_summary() {
    print_header "Pipeline Complete!"
    
    echo "Output files:"
    echo "  • Training metrics:     ./qat_outputs/metrics.json"
    echo "  • Weight logs:          ./qat_outputs/weight_logs/"
    echo "  • Visualizations:       ./qat_outputs/visualizations/"
    echo "  • Clustering report:    ./qat_outputs/visualizations/clustering_report.txt"
    echo ""
    echo "Next steps:"
    echo "  1. Review the clustering report"
    echo "  2. Examine weight evolution plots"
    echo "  3. Verify training loss convergence"
    echo "  4. Convert to deployment format using torchao.quantization.convert()"
    echo ""
}

###############################################################################
# Main Execution
###############################################################################

main() {
    local start_time=$(date +%s)
    
    print_header "QAT Phi-2 Complete Pipeline"
    echo "Started at: $(date)"
    
    # Execute pipeline
    setup_environment
    cleanup_previous_runs
    test_model_loading
    run_training
    run_visualization
    verify_outputs
    print_summary
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "Completed at: $(date)"
    echo "Total time: ${duration}s ($(($duration / 60))m $(($duration % 60))s)"
    print_success "Pipeline execution successful!"
}

# Handle Ctrl+C gracefully
trap 'echo ""; print_error "Pipeline interrupted by user"; exit 130' INT

# Run main function
main "$@"