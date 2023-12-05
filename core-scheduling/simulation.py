from typing import Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from copy import deepcopy
import asyncio

from icecream import ic  # type: ignore
from tqdm import tqdm  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from usim import time, run, Lock, Scope, instant  # type: ignore


# LAMBDA = 30
LAMBDA = 1 / 3
EXECUTION_TIME = 1 / 1
# N_BACKLOG_TASKS = 100
N_BACKLOG_TASKS = 20
# N_BACKLOG_TASKS = 10
# N_BACKLOG_TASKS = 0
PROCESSORS = {
    0: 0.6,
    1: 0.05,
    2: 0.05,
    3: 0.1,
    4: 0.033,
    5: 0.1,
    6: 0.033,
    7: 0.034,
}

EPSTIME = 0.01
# EPSTIME = 0.000001
# EPSTIME = 0
# EPSTIME = 1e-9

N_SIMULATIONS = 1000


async def simulate_once_work_sharing(
    time_until_serviced: list[float],
    time_until_done: list[float],
    processor_queues_over_time: list[dict[int, int]],
    *,
    rng: np.random.Generator,
):
    queue: list[int] = []
    queue_lock = Lock()
    processor_queues: dict[int, list[tuple[float, float]]] = {
        i: [] for i in PROCESSORS.keys()
    }

    async def arrival_process():
        i = 0
        while True:
            await (time + EPSTIME)
            if i >= N_BACKLOG_TASKS:
                interarrival_time = rng.exponential(1 / LAMBDA)
                await (time + interarrival_time)
            async with queue_lock:
                queue.append((i, time.now))
            i += 1

    async def distributor():
        started = False
        while True:
            # ic({k: len(v) for k, v in processor_queues.items()})

            await (time + EPSTIME)
            async with queue_lock:
                if len(queue) == 0:
                    if started and all(len(l) == 0 for l in processor_queues.values()):
                        break
                    else:
                        continue
                else:
                    new_task = queue[0]
                    del queue[0]

            processor = rng.choice(list(PROCESSORS.keys()), p=list(PROCESSORS.values()))
            processor_queues[processor].append(new_task)
            started = True

    async def observer():
        while True:
            await (time + EPSTIME)
            async with queue_lock:
                processor_queues_over_time.append(
                    {k: len(v) for k, v in processor_queues.items()}
                )

    async def processor(i: int):
        # print(f"Spawnnig processor: {i}")

        while True:
            await (time + EPSTIME)
            async with queue_lock:
                this_queue = processor_queues[i]
                if len(this_queue) == 0:
                    continue
                else:
                    new_task = this_queue[0]
                    del this_queue[0]

            # print(f"[{i}] running")
            time_until_serviced.append(time.now - new_task[1])
            execution_time = np.random.exponential(1 / EXECUTION_TIME)
            # ic(i, execution_time, time.now)
            await (time + execution_time)

    async with Scope() as scope:
        start = time.now
        arrival_task = scope.do(arrival_process())
        processor_tasks = {i: scope.do(processor(i)) for i in PROCESSORS.keys()}
        observer_task = scope.do(observer())
        await distributor()
        end = time.now
        arrival_task.cancel()
        observer_task.cancel()
        for processor_task in processor_tasks.values():
            processor_task.cancel()

        time_until_done.append(end - start)


async def simulate_once_work_stealing(
    time_until_serviced: list[float],
    time_until_done: list[float],
    processor_queues_over_time: list[dict[int, int]],
    switches: list[bool],
    *,
    rng: np.random.Generator,
):
    queue: list[int] = []
    queue_lock = Lock()
    processor_queues: dict[int, list[tuple[int, float]]] = {
        i: [] for i in PROCESSORS.keys()
    }
    original_assingments: dict[int, int] = {}

    async def arrival_process():
        i = 0
        while True:
            interarrival_time = rng.exponential(1 / LAMBDA)
            if i >= N_BACKLOG_TASKS:
                await (time + interarrival_time)
            async with queue_lock:
                queue.append((i, time.now))
            i += 1

    async def distributor():
        started = False
        while True:
            # ic({k: len(v) for k, v in processor_queues.items()})

            await (time + EPSTIME)
            async with queue_lock:
                if len(queue) == 0:
                    if started and all(len(l) == 0 for l in processor_queues.values()):
                        break
                    else:
                        continue
                else:
                    new_task = queue[0]
                    del queue[0]

            processor = rng.choice(list(PROCESSORS.keys()), p=list(PROCESSORS.values()))
            processor_queues[processor].append(new_task)
            original_assingments[new_task[0]] = processor
            started = True

    async def observer():
        while True:
            await (time + EPSTIME)
            async with queue_lock:
                processor_queues_over_time.append(
                    {k: len(v) for k, v in processor_queues.items()}
                )

    async def processor(i: int):
        while True:
            await (time + EPSTIME)
            async with queue_lock:
                this_queue = processor_queues[i]
                if len(this_queue) == 0:
                    w = [len(l) for j, l in processor_queues.items() if j != i]
                    if np.sum(w) == 0:
                        p = np.ones(len(w)) / len(w)
                    else:
                        p = w / np.sum(w)
                    processor = rng.choice(
                        [j for j in processor_queues.keys() if j != i],
                        p=p,
                    )
                    # processor = max(processor_queues.items(), key=lambda x: len(x[1]))[
                    #     0
                    # ]
                    this_queue = processor_queues[processor]
                    if len(this_queue) > 0:
                        new_task = this_queue.pop()
                    else:
                        continue
                else:
                    new_task = this_queue[0]
                    del this_queue[0]

            # print(f"[{i}] running")
            time_until_serviced.append(time.now - new_task[1])
            switches.append(i == original_assingments[new_task[0]])
            execution_time = rng.exponential(1 / EXECUTION_TIME)
            await (time + execution_time)

    async with Scope() as scope:
        start = time.now
        arrival_task = scope.do(arrival_process())
        processor_tasks = {i: scope.do(processor(i)) for i in PROCESSORS.keys()}
        observer_task = scope.do(observer())
        await distributor()
        end = time.now
        arrival_task.cancel()
        observer_task.cancel()
        for processor_task in processor_tasks.values():
            processor_task.cancel()

        time_until_done.append(end - start)


def print_statistics(label: str, xs):
    print(f"=> {label}")
    ic(np.mean(xs))
    ic(np.std(xs))
    ic((np.quantile(xs, 0.05), np.quantile(xs, 0.95)))


rng = np.random.default_rng(2)

print()
print(" === Work sharing === ")
print()

time_until_serviced_work_sharing: list[float] = []
time_until_done_work_sharing: list[float] = []
processor_queues_over_time_work_sharing: list[list[dict[int, int]]] = []
for i in tqdm(range(N_SIMULATIONS)):
    this_processor_queues_over_time_work_sharing: list[dict[int, int]] = []
    run(
        simulate_once_work_sharing(
            time_until_serviced_work_sharing,
            time_until_done_work_sharing,
            this_processor_queues_over_time_work_sharing,
            rng=rng,
        )
    )
    processor_queues_over_time_work_sharing.append(
        this_processor_queues_over_time_work_sharing
    )
print_statistics("time until done", time_until_done_work_sharing)
print_statistics("time until serviced", time_until_serviced_work_sharing)

print()
print(" === Work stealing === ")
print()

time_until_serviced_work_stealing: list[float] = []
time_until_done_work_stealing: list[float] = []
switches_work_stealing: list[bool] = []
processor_queues_over_time_work_stealing: list[list[dict[int, int]]] = []
for i in tqdm(range(N_SIMULATIONS)):
    this_processor_queues_over_time_work_stealing: list[dict[int, int]] = []
    run(
        simulate_once_work_stealing(
            time_until_serviced_work_stealing,
            time_until_done_work_stealing,
            this_processor_queues_over_time_work_stealing,
            switches_work_stealing,
            rng=rng,
        )
    )
    processor_queues_over_time_work_stealing.append(
        this_processor_queues_over_time_work_stealing
    )
print_statistics("time until done", time_until_done_work_stealing)
print_statistics("time until serviced", time_until_serviced_work_stealing)
print("probability of switching:", np.mean(switches_work_stealing))

# ---


def plot_histograms(data0, data1, *, title, output):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.4, 2.8))
    for i, (ax, this_data) in enumerate([(ax0, data0), (ax1, data1)]):
        ax.hist(data0, density=False, alpha=0.4, label="work sharing")
        ax.hist(data1, density=False, alpha=0.4, label="work stealing")
        if i == 1:
            ax.legend(loc="best")
        ax.set_xlim(min(this_data), max(this_data))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    fig.savefig(output)
    # fig.close()


def plot_histograms_together(data0, data1, *, title, output):
    fig, ax = plt.subplots(1, 1)
    ax.hist(data0, density=False, alpha=0.4, label="work sharing")
    ax.hist(data1, density=False, alpha=0.4, label="work stealing")
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle(title)
    fig.savefig(output)


def transpose(xs: list[list[Any]]) -> list[list[Any]]:
    return list(zip(*xs))


def plot_processor_queues_over_time(
    processor_queues_over_time0: list[list[dict[int, int]]],
    processor_queues_over_time1: list[list[dict[int, int]]],
    *,
    output: str,
    whether_is_idle: bool,
):
    processor_queues_over_time0 = transpose(processor_queues_over_time0)
    processor_queues_over_time1 = transpose(processor_queues_over_time1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.4, 2.8))

    if whether_is_idle:
        f = lambda x: x > 0
    else:
        f = lambda x: x

    for ax, title, processor_queues_over_time in [
        (ax0, "Work sharing", processor_queues_over_time0),
        (ax1, "Work stealing", processor_queues_over_time1),
    ]:
        img = np.empty((len(PROCESSORS), len(processor_queues_over_time)))
        for i, sizess in enumerate(processor_queues_over_time):
            img[:, i] = np.mean(
                np.array([[f(x) for x in sizes.values()] for sizes in sizess]), axis=0
            )

        im = ax.imshow(img, interpolation="nearest", aspect="auto")
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
        ax.set_xlabel("time")
        ax.set_ylabel("core")
        ax.title.set_text(title)

    fig.subplots_adjust(right=0.8, bottom=0.2)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(
        im,
        cax=cbar_ax,
        label="probability of being idle"
        if whether_is_idle
        else "average number of queued tasks",
    )

    fig.savefig(output)


plot_histograms_together(
    time_until_done_work_sharing,
    time_until_done_work_stealing,
    title="Time until done with task queue",
    output="out-time_until_done-together.png",
)
plot_histograms_together(
    time_until_serviced_work_sharing,
    time_until_serviced_work_stealing,
    title="Time until a task is processed",
    output="out-time_until_processed-together.png",
)
plot_histograms(
    time_until_done_work_sharing,
    time_until_done_work_stealing,
    title="Time until done with task queue",
    output="out-time_until_done.png",
)
plot_histograms(
    time_until_serviced_work_sharing,
    time_until_serviced_work_stealing,
    title="Time until a task is processed",
    output="out-time_until_processed.png",
)
plot_processor_queues_over_time(
    processor_queues_over_time_work_sharing,
    processor_queues_over_time_work_stealing,
    whether_is_idle=False,
    output="out-cores_queue_count.png",
)
plot_processor_queues_over_time(
    processor_queues_over_time_work_sharing,
    processor_queues_over_time_work_stealing,
    whether_is_idle=True,
    output="out-cores_idle.png",
)

# plt.figure()
# plt.hist(time_until_done_work_sharing, density=True, alpha=0.4, label="work sharing")
# plt.hist(time_until_done_work_stealing, density=True, alpha=0.4, label="work stealing")
# plt.title("Time until done with task queue")
# plt.legend(loc="best")
# plt.savefig("out.png")
# plt.close()
#
# plt.figure()
# plt.hist(
#     time_until_serviced_work_sharing, density=True, alpha=0.4, label="work sharing"
# )
# plt.hist(
#     time_until_serviced_work_stealing, density=True, alpha=0.4, label="work stealing"
# )
# plt.title("Time until a task is processed")
# plt.legend(loc="best")
# plt.savefig("out.png")
# plt.close()
