"""Measured behavior of the existing MRIROIEncoder; no new encoder architecture."""
import copy
import torch
from torch import nn


def bn_snapshot(encoder):
    return {name: dict(running_mean=m.running_mean.detach().clone(),
                       running_var=m.running_var.detach().clone(),
                       num_batches_tracked=m.num_batches_tracked.detach().clone())
            for name, m in encoder.named_modules() if isinstance(m, nn.BatchNorm2d)}


def buffer_difference(a, b):
    return {name: dict(mean_max_abs=float((a[name]["running_mean"]-b[name]["running_mean"]).abs().max()),
                       var_max_abs=float((a[name]["running_var"]-b[name]["running_var"]).abs().max()),
                       tracked_difference=int(a[name]["num_batches_tracked"]-b[name]["num_batches_tracked"]))
            for name in a}


def run_batchnorm_audit(encoder):
    """Restore all input encoder states/modes in finally; use synthetic images only."""
    original = copy.deepcopy(encoder.state_dict())
    modes = {name: m.training for name, m in encoder.named_modules()}
    flags = [p.requires_grad for p in encoder.parameters()]
    inventory = [dict(name=name, channels=m.num_features, eps=m.eps, momentum=m.momentum,
                      affine=m.affine, track_running_stats=m.track_running_stats,
                      weight_trainable=m.weight.requires_grad, bias_trainable=m.bias.requires_grad)
                 for name, m in encoder.named_modules() if isinstance(m, nn.BatchNorm2d)]
    initial_bn = bn_snapshot(encoder)
    g = torch.Generator().manual_seed(630)
    images = torch.randn(2, 1, 64, 64, generator=g)
    images[1] = images[1]*1.7 + 0.8
    probe = torch.randn(1, 1, 64, 64, generator=g)

    def execute(inputs, mode, *, sequential=False, reverse=False):
        encoder.load_state_dict(original)
        encoder.train(mode != "eval")
        if mode == "fixed_stats_candidate":
            for m in encoder.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        shapes = {}
        first_moments = {}
        handles = []
        def hook(name):
            def observe(module, args):
                value = args[0]
                shapes[name] = list(value.shape)
                if name == "backbone.bn1":
                    first_moments["mean"] = value.mean((0, 2, 3))
                    first_moments["variance"] = value.var((0, 2, 3), unbiased=True) if value.numel()//value.shape[1] > 1 else None
            return observe
        for name, m in encoder.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                handles.append(m.register_forward_pre_hook(hook(name)))
        try:
            with torch.no_grad():
                if sequential:
                    order = list(reversed(range(len(inputs)))) if reverse else list(range(len(inputs)))
                    output = torch.cat([encoder(inputs[i:i+1]) for i in order])
                    if reverse:
                        output = output.flip(0)
                else:
                    output = encoder(inputs)
            snapshot = bn_snapshot(encoder)
            formula = None
            if mode == "train" and not sequential:
                bn = encoder.backbone.bn1
                expected_mean = initial_bn["backbone.bn1"]["running_mean"]*(1-bn.momentum) + first_moments["mean"]*bn.momentum
                expected_var = initial_bn["backbone.bn1"]["running_var"]*(1-bn.momentum) + first_moments["variance"]*bn.momentum
                formula = dict(mean_error=float((snapshot["backbone.bn1"]["running_mean"]-expected_mean).abs().max()),
                               variance_error=float((snapshot["backbone.bn1"]["running_var"]-expected_var).abs().max()),
                               reduction_axes=[0, 2, 3], effective_values_per_channel=int(shapes["backbone.bn1"][0]*shapes["backbone.bn1"][2]*shapes["backbone.bn1"][3]))
            return output, snapshot, shapes, formula
        finally:
            for h in handles:
                h.remove()

    try:
        comparisons = {}
        for mode in ("train", "eval", "fixed_stats_candidate"):
            stacked, stacked_bn, shapes, formula = execute(images, mode)
            sequential, sequential_bn, _, _ = execute(images, mode, sequential=True)
            _, reversed_bn, _, _ = execute(images, mode, sequential=True, reverse=True)
            # Probe the consequence of different accumulated running statistics.
            with torch.no_grad():
                for name, module in encoder.named_modules():
                    if name in sequential_bn:
                        for key, value in sequential_bn[name].items():
                            getattr(module, key).copy_(value)
                encoder.eval()
                normal_probe = encoder(probe)
                for name, module in encoder.named_modules():
                    if name in reversed_bn:
                        for key, value in reversed_bn[name].items():
                            getattr(module, key).copy_(value)
                reverse_probe = encoder(probe)
            comparisons[mode] = dict(
                stacked_vs_sequential_max_abs=float((stacked-sequential).abs().max()),
                sequential_order_effect_on_subsequent_eval=float((normal_probe-reverse_probe).abs().max()),
                stacked_buffer_changes=buffer_difference(stacked_bn, initial_bn),
                sequential_buffer_changes=buffer_difference(sequential_bn, initial_bn),
                sequential_reverse_buffer_difference=buffer_difference(sequential_bn, reversed_bn),
                stacked_bn_input_shapes=shapes, first_bn_running_update_formula=formula)

        _, small_bn, small_shapes, _ = execute(torch.ones(2, 1, 32, 32), "train")
        _, large_bn, large_shapes, _ = execute(torch.ones(2, 1, 64, 64), "train")
        spatial_audit = dict(input_intensity="constant 1; independently constructed synthetic native arrays",
                            small_first_bn_shape=small_shapes["backbone.bn1"],
                            large_first_bn_shape=large_shapes["backbone.bn1"],
                            first_bn_buffer_difference=buffer_difference(small_bn, large_bn)["backbone.bn1"])
        sizes = [(1,1),(1,3),(16,16),(26,18),(32,27),(32,32),(33,33),(32,33),(1,33)]
        size_results = []
        for h, w in sizes:
            for batch_size, mode in [(1,"train"), (2,"train"), (1,"eval"), (1,"fixed_stats_candidate")]:
                inputs = torch.randn(batch_size, 1, h, w, generator=g)
                try:
                    output, state, shapes, _ = execute(inputs, mode)
                    size_results.append(dict(height=h, width=w, batch_size=batch_size, mode=mode,
                        success=True, output_shape=list(output.shape), finite=bool(torch.isfinite(output).all()),
                        final_bn_shape=shapes["backbone.layer4.1.bn2"]))
                except ValueError as exc:
                    size_results.append(dict(height=h, width=w, batch_size=batch_size, mode=mode,
                                             success=False, error=str(exc)))
        return dict(inventory=inventory, bn_count=len(inventory), comparisons=comparisons,
                    size_results=size_results, spatial_audit=spatial_audit, input_kind="synthetic only; supplied pretrained encoder state",
                    optimization=False)
    finally:
        encoder.load_state_dict(original)
        for name, m in encoder.named_modules():
            m.training = modes[name]
        assert flags == [p.requires_grad for p in encoder.parameters()]
