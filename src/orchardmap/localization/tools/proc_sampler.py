#!/usr/bin/env python3
"""Sample CPU time and RAM of the processes whose command line matches a regex.

    python3 proc_sampler.py <regex> <out_dir> [interval_s=2]

Runs until SIGINT/SIGTERM, then writes <out_dir>/resources.json:
  cpu_seconds        user+sys CPU of all matched processes (sum of each pid's last sample)
  peak_rss_mb        max over samples of the summed RSS of matched processes
  min_mem_available_mb  lowest MemAvailable of the whole VM during the run (OOM margin)
and a per-sample <out_dir>/resources.csv (wall, n_procs, cpu_s, rss_mb, mem_available_mb).
"""
import json
import os
import re
import signal
import sys
import time

TICK = os.sysconf('SC_CLK_TCK')
PAGE = os.sysconf('SC_PAGE_SIZE')


def sample(rx, me):
    cpu, rss = {}, 0
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='ignore')
            if not cmd or not rx.search(cmd):
                continue
            st = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()
            cpu[int(pid)] = (int(st[11]) + int(st[12])) / TICK  # utime + stime
            rss += int(st[21]) * PAGE
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
            continue
    return cpu, rss / 2**20


def mem_available_mb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1]) / 1024
    return float('nan')


def main():
    rx = re.compile(sys.argv[1])
    out = sys.argv[2]
    dt = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
    stop = []
    signal.signal(signal.SIGINT, lambda *a: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *a: stop.append(1))
    last_cpu, peak, min_avail = {}, 0.0, float('inf')
    with open(os.path.join(out, 'resources.csv'), 'w') as f:
        f.write('wall,n_procs,cpu_s,rss_mb,mem_available_mb\n')
        while not stop:
            cpu, rss = sample(rx, os.getpid())
            last_cpu.update(cpu)
            peak = max(peak, rss)
            avail = mem_available_mb()
            min_avail = min(min_avail, avail)
            f.write(f'{time.time():.1f},{len(cpu)},{sum(last_cpu.values()):.1f},{rss:.1f},{avail:.0f}\n')
            f.flush()
            time.sleep(dt)
    res = dict(pattern=sys.argv[1], cpu_seconds=round(sum(last_cpu.values()), 1), peak_rss_mb=round(peak, 1),
               min_mem_available_mb=round(min_avail, 0), n_pids_seen=len(last_cpu))
    with open(os.path.join(out, 'resources.json'), 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    main()
