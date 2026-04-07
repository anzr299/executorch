#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch
from executorch.examples.models.llama.eval_llama_lib import GraphModuleEvalWrapper
from lm_eval.evaluator import simple_evaluate
from pytorch_tokenizers import get_tokenizer


def evaluate_exported_graph(
    exported_program_path: str,
    tokenizer_path: str,
    tasks: list[str],
    limit: int | None,
    seq_length: int,
    calibration_data: str,
    use_kv_cache: bool,
    generate_full_logits: bool,
    enable_dynamic_shape: bool,
    use_cuda: bool,
    split_model_across_gpus: bool
) -> None:
    tokenizer = get_tokenizer(tokenizer_path)

    exported_program = torch.export.load(exported_program_path)
    graph_module = exported_program.module(check_guards=False)

    if use_cuda:
        from accelerate import Accelerator, dispatch_model, infer_auto_device_map

        if split_model_across_gpus:
            device = torch.device("cuda")
            # max_memory = {
            #     0: "20GiB",
            #     1: "20GiB",
            # }
            # device_map = infer_auto_device_map(
            #     graph_module,
            #     max_memory=max_memory,
            # )
            device_map = {}
            device_map["tok_embeddings"] = 0
            device_map["rope"] = 0

            for i in range(32):
                if i <= 19:
                    device_map[f"layers.{i}"] = 0
                else:
                    device_map[f"layers.{i}"] = 1

            device_map["norm"] = 1
            device_map["output"] = 1

            print(f"Device map: {device_map}")

            # Dispatch model across devices
            graph_module = dispatch_model(
                graph_module, 
                device_map=device_map
            )
        else:
            accelerator = Accelerator()
            device = accelerator.device
            for module in graph_module.modules():
                for name, param in module._parameters.items():
                    if param is not None:
                        module._parameters[name] = torch.nn.Parameter(
                            param.data.to(device), requires_grad=param.requires_grad
                        )
                for name, buf in module._buffers.items():
                    if buf is not None:
                        module._buffers[name] = buf.to(device)
        ARANGE_OPS = {
            torch.ops.aten.arange.default,
            torch.ops.aten.arange.start,
            torch.ops.aten.arange.start_step,
        }
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target in ARANGE_OPS:
                new_kwargs = dict(node.kwargs)
                new_kwargs["device"] = device
                node.kwargs = new_kwargs

        for node in list(graph_module.graph.nodes):
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten._assert_tensor_metadata.default
            ):
                node.replace_all_uses_with(node.args[0])
                graph_module.graph.erase_node(node)
        graph_module.graph.lint()
        graph_module.recompile()

    if not use_cuda:
        # torch._dynamo.config.disable = True
        device = torch.device("cpu")
        graph_module = torch.compile(
            graph_module
        )
        inps = (torch.tensor([[2, 3, 4]]), {"input_pos": torch.tensor([0])})
        graph_module(*inps)

    logging.info("Running calibration warmup on loaded graph module...")

    eval_wrapper = GraphModuleEvalWrapper(
        model=graph_module,
        tokenizer=tokenizer,
        max_seq_length=seq_length,
        use_kv_cache=use_kv_cache,
        generate_full_logits=generate_full_logits,
        enable_dynamic_shape=enable_dynamic_shape,
        # device=device,
    )

    if use_cuda:
        torch.cuda.empty_cache()
        print(f"After empty_cache: Allocated={torch.cuda.memory_allocated()/1e9:.2f} GB, Reserved={torch.cuda.memory_reserved()/1e9:.2f} GB")


    logging.info("Running simple_evaluate...")
    with torch.no_grad():
        eval_results = simple_evaluate(
            model=eval_wrapper,
            tasks=tasks,
            limit=limit,
            device=device,
        )

    if eval_results is not None:
        for task, res in eval_results["results"].items():
            print(f"{task}: {res}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a compressed/exported llama GraphModule (.pt2) with lm-eval simple_evaluate.",
    )
    parser.add_argument(
        "--exported_program",
        type=str,
        required=True,
        help="Path to compressed exported program (.pt2).",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Tokenizer path used by the model.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="lm-eval tasks.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional evaluation sample limit.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=1024,
        help="Sequence length for warmup/eval wrapper.",
    )
    parser.add_argument(
        "--calibration_data",
        type=str,
        default="Once upon a time",
        help="Prompt text used for warmup before simple_evaluate.",
    )
    parser.add_argument(
        "--use_kv_cache",
        action="store_true",
        help="Set if the exported graph uses kv cache inputs.",
    )
    parser.add_argument(
        "--generate_full_logits",
        action="store_true",
        help="Set if model generates full logits.",
    )
    parser.add_argument(
        "--enable_dynamic_shape",
        dest="enable_dynamic_shape",
        action="store_true",
        help="Enable dynamic shape path in eval wrapper.",
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Run model on CUDA. If not set, runs on CPU and compiles with OpenVINO backend.",
    )
    parser.add_argument(
        "--split_model_across_gpus",
        action="store_true",
        help="Split 1 model across multiple GPUs. If false, then runs a copy of the model on both GPUs",
    )
    parser.set_defaults(enable_dynamic_shape=False)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args()
    evaluate_exported_graph(
        exported_program_path=args.exported_program,
        tokenizer_path=args.tokenizer_path,
        tasks=args.tasks,
        limit=args.limit,
        seq_length=args.seq_length,
        calibration_data=args.calibration_data,
        use_kv_cache=args.use_kv_cache,
        generate_full_logits=args.generate_full_logits,
        enable_dynamic_shape=args.enable_dynamic_shape,
        use_cuda=args.cuda,
        split_model_across_gpus=args.split_model_across_gpus,
    )


if __name__ == "__main__":
    main()
