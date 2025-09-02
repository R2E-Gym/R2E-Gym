#!/bin/bash

# Automated version of Docker image cleanup script for watchdog use
# Script to delete all Docker images except those from slimshetty/swebench-verified and slimshetty/swebench-lite repositories
# Uses multi-processing for faster deletion - NO USER CONFIRMATION REQUIRED

set -e

# Configuration
PRESERVE_REPOS=("slimshetty/swebench-verified" "slimshetty/swebench-lite" "swebench")
MAX_PARALLEL_JOBS=10  # Number of parallel deletion jobs
DRY_RUN=false
AUTO_MODE=true  # This flag indicates automated execution

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to log with timestamp
log_with_timestamp() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dry-run     Show what would be deleted without actually deleting"
    echo "  -j, --jobs N      Number of parallel jobs (default: 10)"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "This script deletes all Docker images except those from repositories: ${PRESERVE_REPOS[*]}"
    echo "NOTE: This is the automated version - NO CONFIRMATION REQUIRED"
}

# Function to get all Docker images except the preserved ones
get_images_to_delete() {
    # Get all images in format "REPOSITORY:TAG IMAGE_ID"
    local exclude_pattern=""
    for repo in "${PRESERVE_REPOS[@]}"; do
        if [ -z "$exclude_pattern" ]; then
            exclude_pattern="^${repo}:"
        else
            exclude_pattern="${exclude_pattern}|^${repo}:"
        fi
    done
    
    docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}" | \
    grep -v "REPOSITORY:TAG" | \
    grep -vE "$exclude_pattern" | \
    grep -v "^<none>:" || true
}

# Function to delete a single image
delete_image() {
    local image_info="$1"
    local image_id=$(echo "$image_info" | awk '{print $2}')
    local image_name=$(echo "$image_info" | awk '{print $1}')
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would delete: $image_name ($image_id)"
        return 0
    fi
    
    log_with_timestamp "Deleting image: $image_name ($image_id)"
    
    if docker rmi -f "$image_id" 2>/dev/null; then
        log_with_timestamp "SUCCESS: Deleted $image_name ($image_id)"
        return 0
    else
        log_with_timestamp "ERROR: Failed to delete $image_name ($image_id)"
        return 1
    fi
}

# Export function for parallel execution
export -f delete_image
export -f log_with_timestamp
export DRY_RUN
export RED GREEN YELLOW BLUE NC

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -j|--jobs)
            MAX_PARALLEL_JOBS="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate parallel jobs parameter
if ! [[ "$MAX_PARALLEL_JOBS" =~ ^[0-9]+$ ]] || [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
    print_error "Invalid number of parallel jobs: $MAX_PARALLEL_JOBS"
    exit 1
fi

# Main execution
main() {
    log_with_timestamp "=== AUTOMATED Docker Image Cleanup Started ==="
    print_info "Docker Image Cleanup Script (AUTOMATED MODE)"
    print_info "Preserving images from repositories: ${PRESERVE_REPOS[*]}"
    print_info "Using $MAX_PARALLEL_JOBS parallel jobs"
    
    if [ "$DRY_RUN" = true ]; then
        print_warning "DRY RUN MODE - No images will be actually deleted"
    fi
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        print_error "Docker is not running or not accessible"
        log_with_timestamp "ERROR: Docker not accessible"
        exit 1
    fi
    
    # Get list of images to delete
    print_info "Scanning for Docker images..."
    images_to_delete=$(get_images_to_delete)
    
    if [ -z "$images_to_delete" ]; then
        print_success "No images found to delete!"
        log_with_timestamp "No images found to delete - cleanup complete"
        exit 0
    fi
    
    # Count images
    image_count=$(echo "$images_to_delete" | wc -l)
    print_info "Found $image_count images to delete"
    log_with_timestamp "Found $image_count images to delete"
    
    # Show preview of images to be deleted (first 5 only for automated mode)
    echo ""
    print_info "Sample images to be deleted:"
    echo "$images_to_delete" | head -5
    if [ "$image_count" -gt 5 ]; then
        print_info "... and $(($image_count - 5)) more images"
    fi
    echo ""
    
    # Show preserved images
    for repo in "${PRESERVE_REPOS[@]}"; do
        preserved_images=$(docker images --format "table {{.Repository}}:{{.Tag}}\t{{.ID}}" | grep "^${repo}:" || true)
        if [ -n "$preserved_images" ]; then
            preserved_count=$(echo "$preserved_images" | wc -l)
            print_success "Found $preserved_count images from $repo that will be preserved"
            log_with_timestamp "Preserving $preserved_count images from $repo"
        else
            print_warning "No images found from $repo repository"
        fi
    done
    
    # NO CONFIRMATION PROMPT - Automated execution
    if [ "$DRY_RUN" = false ]; then
        print_warning "AUTOMATED MODE: Proceeding with deletion of $image_count images without confirmation"
        log_with_timestamp "Starting automated deletion of $image_count images"
    fi
    
    # Delete images in parallel
    print_info "Starting deletion with $MAX_PARALLEL_JOBS parallel jobs..."
    start_time=$(date +%s)
    
    # Use GNU parallel if available, otherwise use xargs
    if command -v parallel >/dev/null 2>&1; then
        echo "$images_to_delete" | parallel -j "$MAX_PARALLEL_JOBS" delete_image
    else
        echo "$images_to_delete" | xargs -I {} -P "$MAX_PARALLEL_JOBS" bash -c 'delete_image "$@"' _ {}
    fi
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    if [ "$DRY_RUN" = false ]; then
        print_success "Deletion completed in ${duration} seconds"
        log_with_timestamp "Deletion completed in ${duration} seconds"
        
        # Show final statistics
        remaining_images=$(docker images -q | wc -l)
        
        print_info "Final statistics:"
        print_info "  - Total remaining images: $remaining_images"
        log_with_timestamp "Final stats: $remaining_images images remaining"
        
        for repo in "${PRESERVE_REPOS[@]}"; do
            preserved_final=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^${repo}:" | wc -l || echo "0")
            print_info "  - Preserved $repo images: $preserved_final"
            log_with_timestamp "Preserved $repo images: $preserved_final"
        done
    else
        print_success "Dry run completed in ${duration} seconds"
        log_with_timestamp "Dry run completed in ${duration} seconds"
    fi
    
    log_with_timestamp "=== AUTOMATED Docker Image Cleanup Completed ==="
}

# Trap to handle Ctrl+C
trap 'print_warning "Operation interrupted by user"; log_with_timestamp "Operation interrupted"; exit 1' INT

# Run main function
main