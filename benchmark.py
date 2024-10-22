# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from datetime import timedelta

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchbenchmark.util.experiment.instantiator import (
    load_model,
    TorchBenchModelConfig,
)
from torchbenchmark.util.experiment.metrics import get_model_flops
from torchbenchmark.util.input import input_cast

from torchtitan import utils
from torchtitan.checkpoint import TrainState
from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.logging import init_logger, logger
from torchtitan.metrics import build_gpu_memory_monitor
from torchtitan.parallelisms import ParallelDims
from torchtitan.parallelisms.parallelize_llama import (
    apply_ac_vision,
    torch_spmd_parallelize,
)
from torchtitan.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    # used for colorful printing
    color = utils.Color if job_config.metrics.enable_color_printing else utils.NoColor

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp=job_config.training.data_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
        dp_type=job_config.training.data_parallel_type,
    )
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)
    utils.init_distributed(job_config)
    # initialize GPU memory monitor and get peak flops for MFU calculation
    gpu_memory_monitor = build_gpu_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(gpu_memory_monitor.device_name)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    model_name = job_config.model.name

    # initiate model from torchbench
    config = TorchBenchModelConfig(
        name=model_name,
        test="train",
        device="cuda",
        batch_size=job_config.training.batch_size,
        extra_args=[],
    )
    benchmark_model = load_model(config)
    model, _ = benchmark_model.get_module()

    # TODO: there seems to be a bug with dtype conversion (e.g. use resnet50)
    # cast input dtype if needed
    param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    input_cond = lambda x: x.dtype == torch.float32
    input_action = lambda x: x.to(param_dtype)
    if hasattr(benchmark_model, "example_inputs"):
        benchmark_model.example_inputs = input_cast(
            input_cond, input_action, benchmark_model.example_inputs
        )
    else:
        logger.warning(
            f"{model_name} example inputs haven't been cast to {action} yet!"
        )

    # log model size
    model_param_count = utils.get_num_params(model)
    logger.info(
        f"{color.blue}Model {model_name} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
    if job_config.experimental.torch_spmd:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        model = torch_spmd_parallelize(model, world_mesh, parallel_dims, job_config)
    else:
        from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

        if job_config.activation_checkpoint.mode != "none":
            apply_ac_vision(model, job_config.activation_checkpoint)
        param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
        reduce_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce]
        mp_policy = MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=reduce_dtype
        )
        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

        if job_config.training.compile:
            for idx, (name, block) in enumerate(
                model.vision_model.transformer.layers.named_children()
            ):
                block = torch.compile(block, fullgraph=True)
                model.vision_model.transformer.layers.register_module(name, block)
            for idx, (name, block) in enumerate(
                model.vision_model.global_transformer.layers.named_children()
            ):
                block = torch.compile(block, fullgraph=True)
                model.vision_model.global_transformer.layers.register_module(
                    name, block
                )
            for idx, (name, block) in enumerate(
                model.language_model.model.layers.named_children()
            ):
                block = torch.compile(block, fullgraph=True)
                model.language_model.model.layers.register_module(name, block)

        for idx, (name, block) in enumerate(
            model.vision_model.transformer.layers.named_children()
        ):
            reshard_after_forward = False
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        for idx, (name, block) in enumerate(
            model.vision_model.global_transformer.layers.named_children()
        ):
            reshard_after_forward = True
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

        total_module_num = len(list(model.language_model.model.layers.children()))
        for idx, (name, block) in enumerate(
            model.language_model.model.layers.named_children()
        ):
            if idx < total_module_num - 1:
                reshard_after_forward = True
            else:
                reshard_after_forward = False
            fully_shard(
                block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        fully_shard(model, **fsdp_config, reshard_after_forward=True)

    benchmark_model.set_module(model)
    optimizer = benchmark_model.get_optimizer()
    if optimizer is not None:
        optimizer.add_param_group({"params": model.parameters()})

    model.train()

    gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
    logger.info(
        f"GPU memory usage for model: "
        f"{gpu_mem_stats.max_reserved_gib:.2f}GiB"
        f"({gpu_mem_stats.max_reserved_pct:.2f}%)"
    )

    train_state = TrainState()

    # variables used to keep info for metrics logging
    losses_since_last_log = []
    gpu_memory_monitor.reset_peak_stats()

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"total steps {job_config.training.steps}"
    )
    ntokens_since_last_log = 0
    time_last_log = time.perf_counter()

    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            ntokens_since_last_log += benchmark_model.example_inputs["labels"].numel()
            # Collect time_ns() instead of time() which does not provide better precision than 1
            # second according to https://docs.python.org/3/library/time.html#time.time.
            t0 = time.time_ns()
            start_event.record()

            is_staged = (
                hasattr(benchmark_model, "forward")
                and hasattr(benchmark_model, "backward")
                and hasattr(benchmark_model, "optimizer_step")
            )
            if is_staged and (getattr(benchmark_model, "train", None) is None):
                if optimizer is not None:
                    optimizer.zero_grad()
                loss = benchmark_model.forward()
                benchmark_model.backward(loss)
                if optimizer is not None:
                    benchmark_model.optimizer_step()
            else:
                loss = benchmark_model.train()

            end_event.record()
            torch.cuda.synchronize()
            t1 = time.time_ns()
            time_delta = start_event.elapsed_time(end_event), (t1 - t0) / 1_000_000

            # log metrics
            losses_since_last_log.append(loss)
            if (
                train_state.step == 1
                or train_state.step % job_config.metrics.log_freq == 0
            ):
                losses = [
                    loss.item() if isinstance(loss, torch.Tensor) else loss
                    for loss in losses_since_last_log
                ]
                avg_loss, max_loss = sum(losses) / len(losses), max(losses)
                if parallel_dims.dp_enabled:
                    global_avg_loss, global_max_loss = (
                        utils.dist_mean(avg_loss, dp_mesh),
                        utils.dist_max(max_loss, dp_mesh),
                    )
                else:
                    global_avg_loss, global_max_loss = avg_loss, max_loss

                gpu_mem_stats = gpu_memory_monitor.get_peak_stats()
                subtime = time.perf_counter() - time_last_log
                wps = ntokens_since_last_log / (
                    subtime * parallel_dims.model_parallel_size
                )

                logger.info(
                    f"{color.cyan}step: {train_state.step:2}  "
                    f"{color.green}loss: {global_avg_loss:7.4f}  "
                    f"{color.yellow}memory: {gpu_mem_stats.max_reserved_gib:5.2f}GiB"
                    f"({gpu_mem_stats.max_reserved_pct:.2f}%)  "
                    f"{color.blue}GPU time: {time_delta[0]:.3f}ms  "
                    f"CPU wall time: {time_delta[1]:.3f}ms  "
                    f"wps: {round(wps)}{color.reset} "
                )
                ntokens_since_last_log = 0
                time_last_log = time.perf_counter()
                losses_since_last_log.clear()
                gpu_memory_monitor.reset_peak_stats()

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
