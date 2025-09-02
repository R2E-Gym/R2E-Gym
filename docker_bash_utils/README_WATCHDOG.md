# Docker Disk Space Watchdog

A monitoring system that automatically cleans up Docker images when disk space runs low.

## Overview

This system consists of two main components:
1. **`remove_images_except_swebench_auto.sh`** - Automated Docker cleanup script (no user confirmation required)
2. **`disk_watchdog.sh`** - Disk space monitoring script that triggers cleanup when space drops below threshold

## Quick Start

### Check Current Disk Status
```bash
./disk_watchdog.sh --status
```

### Run One-Time Check and Cleanup (if needed)
```bash
./disk_watchdog.sh --once
```

### Start Watchdog in Foreground (for testing)
```bash
./disk_watchdog.sh --threshold 500 --interval 300
```

## Installation Options

### Option 1: Systemd Service (Recommended)

1. **Copy the service file:**
   ```bash
   sudo cp docker_bash_utils/disk-watchdog.service /etc/systemd/system/
   ```

2. **Edit the service file paths if needed:**
   ```bash
   sudo nano /etc/systemd/system/disk-watchdog.service
   ```
   Update paths to match your installation directory.

3. **Enable and start the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable disk-watchdog.service  # This ensures auto-start on boot
   sudo systemctl start disk-watchdog.service
   ```
   
   **Note**: The `enable` command ensures the service will automatically start when you restart your server.

4. **Check service status:**
   ```bash
   sudo systemctl status disk-watchdog.service
   ```

5. **View logs:**
   ```bash
   sudo journalctl -u disk-watchdog.service -f
   ```

## Configuration Changes

### Changing the Disk Space Threshold (e.g., from 500GB to 1000GB)

**Method 1: Edit the systemd service file**
1. Edit the service file:
   ```bash
   sudo nano /etc/systemd/system/disk-watchdog.service
   ```
2. Change the `--threshold 1000` parameter in the `ExecStart` line
3. Reload and restart the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart disk-watchdog.service
   ```

**Method 2: Edit the script defaults**
1. Edit the watchdog script:
   ```bash
   nano docker_bash_utils/disk_watchdog.sh
   ```
2. Change `THRESHOLD_GB=500` to your desired value
3. Restart the service:
   ```bash
   sudo systemctl restart disk-watchdog.service
   ```

### Changing Check Interval
To change from 5 minutes (300 seconds) to another interval:
1. Edit the service file and change `--interval 300` to your desired seconds
2. Reload and restart: `sudo systemctl daemon-reload && sudo systemctl restart disk-watchdog.service`

### Verify Configuration Changes
```bash
# Check current service configuration
sudo systemctl cat disk-watchdog.service

# Check if service will auto-start on boot
sudo systemctl is-enabled disk-watchdog.service

# Test with current settings
./docker_bash_utils/disk_watchdog.sh --status
```

### Option 2: Cron Job

Add to your crontab to run every 5 minutes:
```bash
crontab -e
```

Add this line:
```bash
*/5 * * * * /home/jiahaoyu/projects/R2E-Gym/docker_bash_utils/disk_watchdog.sh --once >/dev/null 2>&1
```

### Option 3: Manual Background Process

Start as a background daemon:
```bash
./disk_watchdog.sh --daemon
```

## Configuration

### Environment Variables
You can set these at the top of `disk_watchdog.sh`:

- `THRESHOLD_GB=500` - Trigger cleanup when available space drops below this (GB)
- `CHECK_INTERVAL=300` - Check interval in seconds (300 = 5 minutes)
- `MOUNT_POINT="/"` - Mount point to monitor
- `MAX_PARALLEL_JOBS=10` - Parallel cleanup jobs

### Command Line Options

**disk_watchdog.sh options:**
- `-t, --threshold N` - Set disk space threshold in GB (default: 500)
- `-i, --interval N` - Set check interval in seconds (default: 300)
- `-o, --once` - Run once and exit (good for cron jobs)
- `-d, --daemon` - Run as daemon (background process)
- `-s, --status` - Show current disk status and exit
- `-h, --help` - Show help message

**remove_images_except_swebench_auto.sh options:**
- `-d, --dry-run` - Show what would be deleted without actually deleting
- `-j, --jobs N` - Number of parallel jobs (default: 10)
- `-h, --help` - Show help message

## What Gets Cleaned Up

The cleanup script will remove all Docker images **EXCEPT**:
- Images from `slimshetty/swebench-verified` repository
- Images tagged as `<none>` (these are usually intermediate images)

## Monitoring and Logs

### Service Logs (if using systemd)
```bash
# View recent logs
sudo journalctl -u disk-watchdog.service --since "1 hour ago"

# Follow logs in real-time
sudo journalctl -u disk-watchdog.service -f

# View all logs
sudo journalctl -u disk-watchdog.service
```

### Script Logs
The watchdog script creates its own log file:
```bash
tail -f docker_bash_utils/disk_watchdog.log
```

## Troubleshooting

### Check if Docker is running
```bash
docker info
```

### Test cleanup script manually
```bash
./remove_images_except_swebench_auto.sh --dry-run
```

### Check disk space manually
```bash
df -h /
```

### Stop the watchdog service
```bash
sudo systemctl stop disk-watchdog.service
```

### Disable the watchdog service
```bash
sudo systemctl disable disk-watchdog.service
```

## Safety Features

1. **Lock file protection** - Prevents multiple instances from running simultaneously
2. **Preserved repositories** - Critical images are never deleted
3. **Detailed logging** - All actions are logged with timestamps
4. **Dry run mode** - Test what would be deleted without actually deleting
5. **Error handling** - Robust error handling and reporting

## Example Usage Scenarios

### Scenario 1: Set up permanent monitoring
```bash
# Install as systemd service
sudo cp docker_bash_utils/disk-watchdog.service /etc/systemd/system/
sudo systemctl enable disk-watchdog.service
sudo systemctl start disk-watchdog.service
```

### Scenario 2: One-time cleanup when space is low
```bash
# Check status first
./disk_watchdog.sh --status

# Run cleanup if needed
./disk_watchdog.sh --once
```

### Scenario 3: Custom threshold for testing
```bash
# Use 1TB threshold instead of 500GB
./disk_watchdog.sh --threshold 1000 --interval 60
```

### Scenario 4: Test what would be cleaned
```bash
# Dry run to see what would be deleted
./remove_images_except_swebench_auto.sh --dry-run
```

## Performance Impact

- **CPU Usage**: Minimal (< 1% during checks)
- **Memory Usage**: < 50MB typically
- **Disk I/O**: Low during monitoring, high during cleanup
- **Network**: None during monitoring

## Security Considerations

- Scripts run with user permissions (not root)
- Lock files prevent conflicts
- Systemd service includes security restrictions
- Logs are rotated to prevent disk fill-up

---

## Quick Reference Commands

```bash
# Status check
./disk_watchdog.sh --status

# One-time run
./disk_watchdog.sh --once

# Start as daemon
./disk_watchdog.sh --daemon

# Test cleanup (dry run)
./remove_images_except_swebench_auto.sh --dry-run

# View service status
sudo systemctl status disk-watchdog.service

# View logs
tail -f docker_bash_utils/disk_watchdog.log