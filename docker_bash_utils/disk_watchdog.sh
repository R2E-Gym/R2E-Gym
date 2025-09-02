#!/bin/bash

# Disk Space Watchdog Script
# Monitors disk usage and triggers Docker cleanup when available space drops below threshold
# Designed to run as a background service or via cron

set -e

# Configuration
THRESHOLD_GB=500  # Trigger cleanup when available space drops below this (in GB)
CHECK_INTERVAL=300  # Check every 5 minutes (300 seconds)
MOUNT_POINT="/"  # Mount point to monitor
CLEANUP_SCRIPT_PATH="$(dirname "$0")/remove_images_except_swebench_auto.sh"
LOG_FILE="$(dirname "$0")/disk_watchdog.log"
LOCK_FILE="/tmp/disk_watchdog.lock"
MAX_LOG_SIZE=10485760  # 10MB in bytes

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
log_message() {
    local level="$1"
    local message="$2"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [$level] $message" | tee -a "$LOG_FILE"
}

# Function to rotate log file if it gets too large
rotate_log() {
    if [ -f "$LOG_FILE" ] && [ $(stat -c%s "$LOG_FILE") -gt $MAX_LOG_SIZE ]; then
        mv "$LOG_FILE" "${LOG_FILE}.old"
        log_message "INFO" "Log file rotated due to size limit"
    fi
}

# Function to get available disk space in GB
get_available_space_gb() {
    df -BG "$MOUNT_POINT" | awk 'NR==2 {print $4}' | sed 's/G$//'
}

# Function to get disk usage statistics
get_disk_stats() {
    df -h "$MOUNT_POINT" | awk 'NR==2 {printf "Used: %s/%s (%s), Available: %s", $3, $2, $5, $4}'
}

# Function to check if cleanup script exists and is executable
validate_cleanup_script() {
    if [ ! -f "$CLEANUP_SCRIPT_PATH" ]; then
        log_message "ERROR" "Cleanup script not found: $CLEANUP_SCRIPT_PATH"
        return 1
    fi
    
    if [ ! -x "$CLEANUP_SCRIPT_PATH" ]; then
        log_message "WARNING" "Making cleanup script executable: $CLEANUP_SCRIPT_PATH"
        chmod +x "$CLEANUP_SCRIPT_PATH"
    fi
    
    return 0
}

# Function to run cleanup
run_cleanup() {
    log_message "WARNING" "Disk space critically low - triggering Docker cleanup"
    log_message "INFO" "Running: $CLEANUP_SCRIPT_PATH"
    
    # Run cleanup script and capture output
    if "$CLEANUP_SCRIPT_PATH" >> "$LOG_FILE" 2>&1; then
        log_message "SUCCESS" "Docker cleanup completed successfully"
        
        # Check new available space
        local new_space=$(get_available_space_gb)
        local new_stats=$(get_disk_stats)
        log_message "INFO" "Post-cleanup disk status: $new_stats (Available: ${new_space}GB)"
        
        if [ "$new_space" -gt "$THRESHOLD_GB" ]; then
            log_message "SUCCESS" "Disk space restored above threshold (${new_space}GB > ${THRESHOLD_GB}GB)"
        else
            log_message "WARNING" "Disk space still below threshold after cleanup (${new_space}GB <= ${THRESHOLD_GB}GB)"
        fi
    else
        log_message "ERROR" "Docker cleanup failed - check log for details"
    fi
}

# Function to check if another instance is running
check_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local lock_pid=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$lock_pid" ] && kill -0 "$lock_pid" 2>/dev/null; then
            log_message "WARNING" "Another instance is already running (PID: $lock_pid)"
            return 1
        else
            log_message "INFO" "Stale lock file found, removing"
            rm -f "$LOCK_FILE"
        fi
    fi
    return 0
}

# Function to create lock file
create_lock() {
    echo $$ > "$LOCK_FILE"
}

# Function to remove lock file
remove_lock() {
    rm -f "$LOCK_FILE"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --threshold N    Set disk space threshold in GB (default: $THRESHOLD_GB)"
    echo "  -i, --interval N     Set check interval in seconds (default: $CHECK_INTERVAL)"
    echo "  -o, --once           Run once and exit (don't loop)"
    echo "  -d, --daemon         Run as daemon (background process)"
    echo "  -s, --status         Show current disk status and exit"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "This script monitors disk usage on $MOUNT_POINT and triggers Docker cleanup"
    echo "when available space drops below the threshold."
}

# Function to show disk status
show_status() {
    local available_space=$(get_available_space_gb)
    local disk_stats=$(get_disk_stats)
    
    print_info "Disk Watchdog Status"
    print_info "Mount point: $MOUNT_POINT"
    print_info "Current status: $disk_stats"
    print_info "Available space: ${available_space}GB"
    print_info "Threshold: ${THRESHOLD_GB}GB"
    
    if [ "$available_space" -le "$THRESHOLD_GB" ]; then
        print_warning "Status: CRITICAL - Below threshold!"
    else
        print_success "Status: OK - Above threshold"
    fi
}

# Function to run as daemon
run_daemon() {
    log_message "INFO" "Starting disk watchdog daemon (PID: $$)"
    log_message "INFO" "Configuration: Threshold=${THRESHOLD_GB}GB, Interval=${CHECK_INTERVAL}s, Mount=${MOUNT_POINT}"
    
    while true; do
        rotate_log
        
        local available_space=$(get_available_space_gb)
        local disk_stats=$(get_disk_stats)
        
        log_message "DEBUG" "Disk check: $disk_stats (Available: ${available_space}GB)"
        
        if [ "$available_space" -le "$THRESHOLD_GB" ]; then
            log_message "WARNING" "Disk space below threshold: ${available_space}GB <= ${THRESHOLD_GB}GB"
            run_cleanup
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# Main function
main() {
    local run_once=false
    local run_as_daemon=false
    local show_status_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--threshold)
                THRESHOLD_GB="$2"
                shift 2
                ;;
            -i|--interval)
                CHECK_INTERVAL="$2"
                shift 2
                ;;
            -o|--once)
                run_once=true
                shift
                ;;
            -d|--daemon)
                run_as_daemon=true
                shift
                ;;
            -s|--status)
                show_status_only=true
                shift
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
    
    # Validate parameters
    if ! [[ "$THRESHOLD_GB" =~ ^[0-9]+$ ]] || [ "$THRESHOLD_GB" -lt 1 ]; then
        print_error "Invalid threshold: $THRESHOLD_GB (must be positive integer)"
        exit 1
    fi
    
    if ! [[ "$CHECK_INTERVAL" =~ ^[0-9]+$ ]] || [ "$CHECK_INTERVAL" -lt 10 ]; then
        print_error "Invalid interval: $CHECK_INTERVAL (must be >= 10 seconds)"
        exit 1
    fi
    
    # Show status only
    if [ "$show_status_only" = true ]; then
        show_status
        exit 0
    fi
    
    # Validate cleanup script
    if ! validate_cleanup_script; then
        exit 1
    fi
    
    # Check for another instance (skip for one-time runs)
    if [ "$run_once" = false ] && ! check_lock; then
        exit 1
    fi
    
    # Create lock file for continuous runs
    if [ "$run_once" = false ]; then
        create_lock
        trap 'remove_lock; exit' INT TERM EXIT
    fi
    
    if [ "$run_once" = true ]; then
        # Single check
        print_info "Running single disk space check..."
        local available_space=$(get_available_space_gb)
        local disk_stats=$(get_disk_stats)
        
        print_info "Current disk status: $disk_stats"
        print_info "Available space: ${available_space}GB, Threshold: ${THRESHOLD_GB}GB"
        
        if [ "$available_space" -le "$THRESHOLD_GB" ]; then
            print_warning "Disk space below threshold - running cleanup"
            run_cleanup
        else
            print_success "Disk space above threshold - no cleanup needed"
        fi
    elif [ "$run_as_daemon" = true ]; then
        # Run as background daemon
        print_info "Starting as background daemon..."
        run_daemon &
        local daemon_pid=$!
        print_success "Daemon started with PID: $daemon_pid"
        print_info "Log file: $LOG_FILE"
        print_info "To stop: kill $daemon_pid"
    else
        # Run in foreground
        print_info "Starting disk watchdog in foreground mode..."
        print_info "Press Ctrl+C to stop"
        run_daemon
    fi
}

# Run main function
main "$@"