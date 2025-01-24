import logging
import statistics
import time

logger = logging.getLogger(__name__)

item_splits = {}
item_durations_seconds = []
current_start_timestamp = None
current_item_splits = []


def reset():
    global item_splits
    global item_durations_seconds
    global current_start_timestamp
    global current_item_splits

    item_splits = {}
    item_durations_seconds = []
    current_start_timestamp = None
    current_item_splits = []


def notify_start_item():
    global current_start_timestamp
    global current_item_splits
    current_item_splits = []
    current_start_timestamp = time.time()


def notify_end_item():
    global current_start_timestamp
    global current_item_splits
    global item_splits
    global item_durations_seconds

    now_time_seconds = time.time()

    assert current_start_timestamp is not None, "notify_start_item() must be called before notify_end_item()"
    item_duration = now_time_seconds - current_start_timestamp
    item_durations_seconds.append(item_duration)

    for current_item_split in current_item_splits:
        if current_item_split["split_name"] not in item_splits:
            item_splits[current_item_split["split_name"]] = []
        item_splits[current_item_split["split_name"]].append(current_item_split["duration_to_previous_seconds"])


def add_timing(split_name: str):
    global current_start_timestamp
    global current_item_splits

    now_time_seconds = time.time()

    #logger.info(f"Adding timing for split_name '{split_name}'")

    assert current_start_timestamp is not None, "notify_start_item() must be called before add_timing()"

    for timing in current_item_splits:
        if timing["split_name"] == split_name:
            raise Exception(f"Type '{split_name}' already added")

    previous_timing = current_item_splits[-1] if len(current_item_splits) > 0 else None

    current_item_splits.append(
        {
            "split_name": split_name,
            "duration_to_previous_seconds": now_time_seconds - previous_timing["timestamp_seconds"] if previous_timing is not None else now_time_seconds - current_start_timestamp,
            "duration_to_start_seconds": now_time_seconds - current_start_timestamp,
            "timestamp_seconds": now_time_seconds,
        }
    )


def log_final_statistics():
    global item_splits
    global item_durations_seconds

    logger.info("====== Runtime statistics ======")

    logger.info("### Overall item durations:")
    logger.info(f"  - Mean: {statistics.mean(item_durations_seconds)} sec")
    logger.info(f"  - Median: {statistics.median(item_durations_seconds)} sec")
    logger.info(f"  - Min: {min(item_durations_seconds)} sec")
    logger.info(f"  - Max: {max(item_durations_seconds)} sec")
    logger.info(f"  - Count: {len(item_durations_seconds)}")
    logger.info("")

    logger.info("### Single splits:")
    for split_name in item_splits.keys():
        split_durations_seconds = item_splits[split_name]
        logger.info("")
        logger.info(f"Runtime timings for split_name '{split_name}':")
        logger.info(f"  - Mean: {statistics.mean(split_durations_seconds)} sec")
        logger.info(f"  - Median: {statistics.median(split_durations_seconds)} sec")
        logger.info(f"  - Min: {min(split_durations_seconds)} sec")
        logger.info(f"  - Max: {max(split_durations_seconds)} sec")
        logger.info(f"  - Count: {len(split_durations_seconds)}")
        logger.info("")
