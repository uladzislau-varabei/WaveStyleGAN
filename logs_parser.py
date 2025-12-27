from copy import deepcopy

import numpy as np

from shared_utils import format_time, read_txt


def extract_gpu_stats(logs):
    def get_empty_stats():
        return {"peak": [], "reserved": [], "chip": [], "mem": [], "clock": [], "temp": [], "power": []}

    stats = {}
    for line in logs:
        if "[rank=" in line and "GPU" in line:
            gpu_key = line.split("[rank=")[1].split(":")[0].split(" ")[1]
            if gpu_key not in stats:
                stats[gpu_key] = get_empty_stats()
            # 1. Peak memory
            peak_memory = float(line.split("peak")[1].split("GB")[0].strip())
            stats[gpu_key]["peak"].append(peak_memory)
            # 2. Reserved memory
            res_memory = float(line.split("reserved")[1].split("GB")[0].strip())
            stats[gpu_key]["reserved"].append(res_memory)
            # 3. Chip usage
            chip_usage = float(line.split("CHIP usage")[1].split("%")[0].strip())
            stats[gpu_key]["chip"].append(chip_usage)
            # 4. Memory usage
            mem_usage = float(line.split("MEM usage")[1].split("%")[0].strip())
            stats[gpu_key]["mem"].append(mem_usage)
            # 5. Clock
            clock = int(line.split("MEM usage")[1].split(",")[1].split("MHz")[0].strip())
            stats[gpu_key]["clock"].append(clock)
            # 6. Temperature
            temp = int(line.split("MHz")[1].split(",")[1].split("C")[0].strip())
            stats[gpu_key]["temp"].append(temp)
            # 7. Power
            power = int(line.split("MHz")[1].split(",")[2].split("W")[0].strip())
            stats[gpu_key]["power"].append(power)
    upd_stats = {}
    for gpu_key, values in stats.items():
        upd_stats[gpu_key] = {}
        for stat, value in values.items():
            array = np.sort(np.array(value))
            n_items = len(array)
            cutoff_items = int(0.1 * n_items)
            array = array[cutoff_items: -cutoff_items]
            upd_stats[gpu_key][stat] = round(float(array.mean()), 3)
    # Format stats
    upd_stats_str = ""
    for k in sorted(upd_stats.keys()):
        v = upd_stats[k]
        upd_stats_str += f"{k}\n{v}\n"
    return upd_stats, upd_stats_str.strip()


def extract_time_stats(logs):
    stats = {
        "sec/tick": [], "sec/kimg": [], "img/sec": []
    }
    all_tick_times = []
    # Accumulation
    for line in logs:
        if "tick " in line:
            kimg_time = float(line.split("sec/kimg")[1].split(",")[0].strip())
            tick_time = float(line.split("sec/tick")[1].split(",")[0].strip())
            speed = float(line.split("img/sec")[1].split(",")[0].strip())
            stats["sec/kimg"].append(kimg_time)
            stats["sec/tick"].append(tick_time)
            stats["img/sec"].append(speed)
            tick_finish_time = line.split("time")[1].split(",")[0].strip()
            all_tick_times.append(tick_finish_time)
    # Stats
    upd_stats = {}
    for k, v in stats.items():
        array = np.array(v)
        array = array[array > 0.0]
        log_cutoff = False
        if log_cutoff:
            n_items = len(v)
            cutoff_items = int(0.03 * n_items)
            print(f"{k:>10}: src_size={len(v)}, upd_size={len(array)}, cutoff_items={cutoff_items}")
        upd_stats[k] = round(float(array.mean()), 3)
    # Total training time. There can be breaks in training
    runs_times = []
    runs_times_str = []
    prev_line = None
    prev_seconds = -1
    for tick_time in all_tick_times:
        time_splits = tick_time.split(" ")
        tick_seconds = 0
        for s in time_splits:
            number = int(s[:-1])
            letter = s[-1]
            mult = {
                "d": 3600 * 24,
                "h": 3600,
                "m": 60,
                "s": 1
            }[letter]
            tick_seconds += mult * number
        if tick_seconds < prev_seconds:
            runs_times.append(prev_seconds)
            runs_times_str.append(prev_line)
        prev_seconds = tick_seconds
        prev_line = deepcopy(tick_time)
    upd_stats["total_time"] = format_time(sum(runs_times))
    return upd_stats


if __name__ == '__main__':
    logs_path = './results/FFHQ_v1_1024x1024_best/logs.txt'
    logs = read_txt(logs_path)

    print(f'Logs path: {logs_path}\n')
    gpu_stats, gpu_stats_str = extract_gpu_stats(logs)
    print(f'GPU stats:\n{gpu_stats_str}\n')
    time_stats = extract_time_stats(logs)
    print(f'Time stats:\n{time_stats}\n')
    print('--- Done parsing logs ---')
