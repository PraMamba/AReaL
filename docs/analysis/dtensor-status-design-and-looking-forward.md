# DTensor - Status, Design and Looking Forward

This post outlines the current status, design principles, and future work items for PyTorch DTensor.

# Overall Design Objective and Principles

DTensor is the PyTorch native tensor sharding primitive. For the motivations and use cases, please refer the original RFC and the design doc from this issue[RFC] PyTorch DistributedTensor · Issue \#88838 · pytorch/pytorch · GitHub

DTensor hold a couple of overall design objective and principles:

- Simple SPMD sharding primitive : Our goal is building a simple SPMD sharding primitive that could accelerate the research innovations for distributed training/inference using PyTorch, and offering a simple programming model for distributed algorithms written in PyTorch.
- Fully PyTorch Native : DTensor should in-theory work seamlessly with any PyTorch subsystems, including the Autograd Engine, torch.compile, nested subclasses, low-precision dtypes, custom operators, etc.
- Single Device Semantic : Author distributed algorithm as if it’s a single-device program with the same convergence property. Single Device Semantic is crucially important for DTensor as it is the ONLY reliable way to ensure convergence/numerics for SPMD sharding algorithms, we have ensured all APIs to follow single device semantic.
- Less is more : DTensor focuses on building the right primitives, not the high level distributed algorithm APIs. We prefer to expose the minimum amount of APIs that are simple and necessary, and give flexibility to users (i.e. system developers) to build on top of it.

# User Experiences

Note that all public APIs are currently documented in torch.distributed.tensor — PyTorch main documentation and we are committing backward compatibility for the non-experimental APIs.

## DeviceMesh for multi-dimensional sharding

DeviceMesh serves as a “Device” Manager or Communicator in the SPMD programming model. In the world that needs multi-dimensional parallelisms and parallelisms becomes more and more complicated, even communication setups become challenging and hard to understand. Being able to describe device layout is one fundamental step to implement multi-dimensional sharding.

As part of the DTensor development, we built DeviceMesh to manage the cluster device layouts and initialize communicators (ProcessGroups) across devices in the cluster. We have released beta for DeviceMesh as a core distributed abstraction since 2.2 release (recipe: Getting Started with DeviceMesh — PyTorch Tutorials 2.9.0+cu128 documentation).

## Sharding APIs

The sharding APIs we have developed are pretty stable since the initial introduction. Notably we offer APIs to construct DTensor, transform DTensor Layout, interact with local torch.Tensor, debugging tools, and some experimental APIs like custom operator registration.

To construct DTensor, we offer three ways:

1. `distribute_tensor` : This API intended to be used for leaf full tensors (i.e. nn.Parameters or Buffers), and it is primarily used for parameter sharding initialization. To correctly preserve single device semantics, we by default `broadcast/scatter` from `group_rank=0` on all mesh dimensions. We recently introduced `src_data_rank` kwarg that controls the source data rank, and if passing `None` , we skip the communication.
2. `DTensor.from_local` : static method that allows initialization from local tensors, this API is autograd aware and mainly used in the middle of tensor computation.
3. Native DTensor constructors (i.e. `torch.distributed.tensor.zeros/ones` , etc) with `device_mesh` and `placements` to specify sharding. Compared to 1, the native Tensor constructors does not need to perform sharding (i.e. `scatter/broadcast` ) on full tensors, and can directly initialize the sharded data via proper RNG support (covered in the random operators section)

There are three types of placement we support:

- `Shard(dim)` : describes sharding on the tensor dimension dim over the current mesh dimension. The sharding current uses the `torch.chunk` semantic for unevenness
- `Replicate()` : describes replication over the current mesh dimension
- `Partial()` : describes the partial tensor state that is pending reduction. This placement usually comes from the intermediate computations (i.e. output of `aten.mm.default` ), we choose to make this public because PyTorch is eager-first so every state can be examined by the end user.

Note that for each operator, if one input tensor is `DTensor` , we require ALL tensor inputs of that operator to be `DTensors` (put it in another way: for operators, `DTensor` only works with `DTensor` ). This is because for each operator, one would need to know the sharding layout of all the input operands in order to derive the correct sharding layout for the output.

Although DTensor primarily offers tensor level sharding primitive APIs, we have also added a module level API `distribute_module` to help developers write sharding algorithms that apply on `nn.Module` s. It takes in three different functions as args:

- `partition_fn` : a callable that defines how to partition/shard the model parameters or buffers within the `nn.Module` , we use a callable to give the flexibility to the developers to define the detailed sharding behaviors (i.e. one can call `distribute_tensor` or `DTensor.from_local` for different parameters)
- `input_fn` : a callable that defines how to deal with the input tensors. One could use this callable to turn plain torch.Tensors to `DTensor` s with the desired sharding annotation (i.e. use `DTensor.from_local` ), redistribute the input `DTensor` to another sharding layout, etc.
- `output_fn` : a callable that defines how to deal with the output Tensors. One could call `redistribute` to change the output to a different sharding layout, call `to_local` on the output DTensors so that the outputs could exit the `DTensor` compute region, etc.

## Transform DTensor Layout (“Collectives”)

In the SPMD programming model, without higher level abstractions like DTensor, users have to manually perform tensor sharding, write different differentiable collectives to preserve single device semantics, and bookkeeping the tensor sharding state everywhere. This is not only tricky/tedious, but also becomes very complicated to write/maintain, especially when dealing with multi-dimensional shardings.

The redistribute API abstracts out those complexity by modeling different collective calls as DTensor Layout transformations. In this way, users could directly deal with Tensor sharding layout transformation, without the headache of writing the lower level implementation on how to perform different collectives to reach the desired sharding layout, which could result in potential collective hangs or deadlocks.

Some common transformations we have supported with the redistribute API:

1. `Shard(dim) -> Replicate()` : `all_gather`
2. `Shard(src_dim) -> Shard(dst_dim)` : `all_to_all`
3. `Replicate() -> Shard(dim)` : local `torch.chunk`
4. `Partial() -> Replicate()` : `all_reduce`
5. `Partial() -> Shard(dim)` : `reduce_scatter`

The above transformations describe the sharding layout change on a single mesh dimension. Another important feature that `redistribute` API offers is to figure out the correct transformation steps on multi-dimensional DeviceMesh. This becomes way more complicated for even advanced users to figure out how to do it with manual collectives, and the right sharding abstraction like DTensor helps. An example transformation we want to achieve on a 2-D DeviceMesh:

```
[Shard(0), Shard(1)] -> [Shard(1), Shard(0)]
```

`redistribute` will break this down into several steps to make sure the sharding layout get properly transformed with the right data on the right rank:

```
[Shard(0), Shard(1)] -> [Shard(0), Replicate()]
[Shard(0), Replicate()] -> [Shard(1), Replicate()]
[Shard(1), Replicate()] -> [Shard(1), Shard(0)]
```

We can see that even for 2D DeviceMesh there are many steps to take for one case, and even more for 3D DeviceMesh or above, the redistribute API simplifies the transformation and allows users to focus on building the sharding algorithm. Please refer to this PR about the detailed multi-dim mesh redistribute algorithm changes https://github.com/pytorch/pytorch/pull/131210

There’re two things to note here:

- `redistribute` is just an API that performs standardized DTensor Layout transformations, users could still write manual collective implementations with DTensor inputs and construct the output DTensor with the proper DTensor Layout.
- We currently support transforming DTensor Layouts within the same DeviceMesh, not cross DeviceMesh yet. However depending on different needs, we can implement layout transformation across different submeshes (i.e. for the case of MPMD or checkpoint resharding purposes)

## Random Operators

Random operators in PyTorch are special compared to other operators, this is because it involves how the random number generation works. Specifically for abstractions like `DTensor` , we need to care about how the random numbers should be generated for different type shardings, so that the output tensor sharding still makes sense.

For a PyTorch operator (i.e. `aten.dropout.default` ). The important properties we want to preserve w.r.t. sharding:

- For replicated tensor input, it should produce replicated tensor output, which means the SAME data should be produced on the ranks of a mesh dimension.
- For sharded tensor input, it should produce sharded tensor output, which means DIFFERENT data should be produced on ranks of a mesh dimension. In an ideal situation, the data produced from different shards should be as if it’s the data generated for the “global/full” tensor on a single device.

Not only for the runtime random operators, random operators with sharding are also for tensor creation ops (i.e. `torch.randn, torch.rand` , etc.)

Given the above properties, we leverage the PhiloxRNG and use both seed/offset when sampling values from the random distribution. Specifically, we implemented an `OffsetRNGTracker` where we lazily sync/broadcast a seed from `rank=0` , and use the broadcast seed as the “global seed”, then for every random operators, DTensor uses the `RNGTracker` to compute the offset w.r.t that global seed, and move/keep track of the offset according to the input DTensor sharding.

For example, suppose we start from `(global_seed, offset)` state, and we run sharded dropout computation on a ShardedDTensor input. During the execution, shard/rank `i` would start with `(global_seed, offset + (i-1) * (numel_local_shard))` After the execution, the offset on each rank would be moved after `(offset + numel_global_tensor)` . In this way we ensure the sampling behavior mimics the single device semantic.

## Custom Operators

We have been focusing on enabling many native PyTorch operators in DTensor. But extensibility is crucial for PyTorch. We should make sure the PyTorch extension system works well with DTensor, specifically if a user defines a custom operator, it should also work with DTensor!

How could an arbitrary operator work with DTensor? Since we don’t know the math semantics of the custom operator (i.e. whether it’s a pointwise op or a reduction op), the custom op author is required to provide a “sharding formula” to tell us how to deal with sharding. This is similar to the shape formula needed for the new custom op API.

We have implemented an experimental API: `register_sharding` . This API allows the user to define the sharding strategy of a custom op, so that DTensor knows how to produce the output sharding given the input shardings. We should solicit feedback about this API and make it stable. We can also consider expose some common sharding strategy (i.e. pointwise, follow strategy)

## Local_map: Interact with local Tensor(non-DTensor) program

DTensor allows users to write a distributed program as if it’s a single device program with the same convergence property. However there might be many cases where developers still want to be able to go one level deeper, and write the manual collective code as before. This is especially true for many optimization explorations. For example, we could have some cases like below:

- Case 1: Developers might want to leverage DTensor for most of the model computation, but write some custom optimization code for 1-2 layers, where it would be beneficial to early call a collective to achieve better overlap.
- Case 2: Developers write a custom triton code, they might want to directly call it in Python without going through custom op registration, and they know the expected input/output sharding be look like, so they just want this triton kernel to run with local tensors.

We introduced `local_map` experimental API, this is a function decorator where it extracts DTensor local shard (by calling to_local) and calls the function with local tensor directly, and with user provided out_placements, reconstructs the outputs to the corresponding DTensors. In this way we give advanced users flexibility to go deeper and do whatever they want.

One notable example usage of local_map is the custom RMSNorm triton kernel in torchtitan: with local_map, we are able to run SequenceParallel with a custom RMSNorm layer easily, and preserve all other DTensor shardings.

## Debuggability

Note that ANY abstraction or API would hide certain details of the underlying implementation. Advanced users (i.e. distributed system developers) would want to dive into what happened under the hood, this applies the same to DTensor.

DTensor manages sharding and DTensor Layouts under the hood when executing the PyTorch operators, it is natural for advanced users or DTensor developers to understand what actually happened. For better debuggability, we should provide two levels of debugging tools well:

- Sharding visualization: this is useful for developers to understand how sharding (especially multi dimensional sharding) is performed, and the relationship between shards ↔ devices. For this we have the `visualize_sharding` API, but it’s a preliminary tool and not polished well yet.
- Communication and Computation Tracking: track the communication and computation happening when executing the DTensor region, not only the number of times each operator executes, but also the execution order for complete debug analysis. For this we have the `CommDebugMode` , we should also polish it to make it more user friendly.

Note that `CommDebugMode` itself could be useful and be released as a separate feature if we polish it well, this tool would be useful for users who are using/developing complicated parallelism solutions, and want to ensure the communication happens in the right order. As an example, developers use this tool very often when building out parallelism solutions like FSDP/TP, because when building higher level parallelism solutions, it’s ALWAYS critical to ensure the right communications happen in the right place.

## XLA Backend

torchxla recently developed their SPMD sharding API, and under the hood it utilizes the GSPMD partitioner from the XLA compiler. DTensor integrates with the torchxla SPMD sharding API as we share similar concepts about sharding. Reference issue:[RFC] XLA Lazy Backend Support In DistributedTensor API · Issue \#92909 · pytorch/pytorch · GitHub

# Parallelism Authoring

DTensor is a sharding primitive to help parallelism authoring be easier. Being able to build parallelism solutions on top testifies the effectiveness of DTensor. So far we have developed a bunch of parallelism solutions on top.

## FSDP2

FSDP2 uses DTensor as the data abstraction layer, for:

1. Simpler checkpointing save/load (DTensor + DCP is our preferred solution)
2. Easy manipulation of individual parameters
3. Better support for model parallelism (2D/3D), where it natively integrates with PyTorch TP, and support sharded state_dict for 2D/3D parallelism
4. Better tensor level feature support like meta device initialization, grad norm clipping, quantizations.

There’s also SimpleFSDP exploration we had gives us an alternative way to implement FSDP with pure compile optimizations, and further simplify the FSDP implementation. SimpleFSDP is still in the process of exploration.

## Tensor/Sequence Parallel

Tensor/Sequence Parallel is another parallelism that uses DTensor. TP APIs use DTensor in a more straightforward way, it requires the user to annotate the sharding for model parameters and inputs. Then it simply just constructs DTensors with those user provided shardings, and runs the sharded computation using DTensor. This means:

- All input/model parameters are DTensors
- All computation happened through DTensor’s `__torch_dispatch__`
- Checkpoint save/load directly interacts with DTensor (which DCP supports)

We have built/released the TP APIs to beta. There’re a couple of future features we should work on:

- Sharding Convolution layers using TP (i.e. support aten.conv ops to support conv layer sharding in models like ViT)
- More quantization (mixed dtype) support with torchao
- Support more customizations with the help from custom operator registration or local_map
- Fused QKV sharding (support can be done within TP API). This is rarely used in training (except GPT2), but could be useful for inference

## Context Parallel

The current CP implementation leverages DTensor’s dispatch approach to allow non-model intrusive changes. It currently only works with FlashAttention, specifically the `F.scaled_dot_production_attention` API and its FlashAttention kernel path. There’re a couple improvements we should do:

- `FlexAttention` : Given that there’re many innovations for the attention part, we should try to support context parallel in a more fine-grained way. With Flex Attention, there are many attention variants, and we should well equip DTensor to support context parallelism. It could be more natural to use DTensor for the FlexAttention work.
- The existing solution should support the `F.sdpa` path well, including flash attention and memory efficient attention. We should make sure proper knobs are exposed to users (i.e. whether to do allgather/ring communication), and improve the API based on user feedback.

## Expert Parallel

The EP work is currently on-going and the requirements to the DTensor infra is not fully done yet, but Tianyu Liu have a great doc outlining the progress. PT-D MoE & EP exploration

# Performance

## Tensor Subclass Overhead

DTensor is an `__torch_dispatch__` based Tensor subclass, this means it would have all the pros and cons from the Tensor subclass. For CPU sensitive workloads (i.e. model with large number of parameters but very low computation), there might be noticeable overhead on CPU.

On one hand, there aren’t too many improvements we can do due to the Python Tensor Subclass overhead, which triggers python → C++ → python → C++ kernel roundtrips. For example, if we run torch.add on a tensor subclass input, in PyTorch we need to go through:

```
torch.add -> torch.ops.aten.add.default -> aten:add -> subclass’s __torch_dispatch__ (through torch_dispatch_key) -> aten.add.default -> aten:add -> cuda:add
```

It takes 4-5 steps in the PyTorch dispatcher to lower to the actual kernel, compared to normal torch.Tensor, Tensor subclass introduced at least 2 additional trips: 1 trip from C++ to python, and 1 trip from python to C++ again.

So in theory, tensor subclass overhead can only be fully gotten rid of with tools that could remove CPU overhead, notably two options: 1. Torch.compile 2. Cuda Graph. They are not mutually exclusive (i.e. torch.compile have a cudagraph backend), the first option is more on enabling the compiler for DTensor, and the second option is more on getting eager cudagraph API to work with DTensor.

## CPU overhead Improvement in TorchDispatch

On the other hand, if we want to improve the CPU overhead without the help from torch.compile or CUDAGraph, we can try to improve the dispatching logic. In fact we have already improved quite a bit on DTensor’s dispatch logic, including proper caching at various levels, flattening only when needed to try to minimize the CPU overhead.

There’re still a couple of things we can do to further optimize the CPU overhead. Today primary overhead comes from the caching layer, i.e. `__hash__` and `__eq__` of the OpSchema to check whether to reuse the sharding propagation result. So some exploration we can do:

- See how we could further improve the python performance on the dispatch logic, i.e. whether we could do more caching, and improve `__hash__` and `__eq__` performance.
- Experiment with moving some implementation to C++: This might be much more involved, i.e. we can experiment moving the basic components, like DeviceMesh, Placement, and the corresponding `DTensorSpec` to C++, and see if that would improve the hashing performance
- Look into the tensor subclass re-dispatch logic in PyTorch core, and see where we could improve the redispatching performance

## Torch.compile DTensor

Enabling `torch.compile` to work with DTensor could help us completely remove the CPU overhead. We have enabled torch.compile + DTensor already, and implemented several optimizations on top:

- Completely remove any subclass related CPU overhead.
- Model applied DTensor can get out of box compute fusion from TorchInductor.
- DTensor authored sharding algorithm could be further optimized inside TorchInductor, i.e. Async TP and Simple FSDP both implemented optimizations for communication overlap/reordering
- We can keep sharding logic be reasonably simple and leverage compile to perform more fine grained optimizations

We should harden the torch.compile + DTensor path, including:

- Get all our unit tests to work with torch.compile, make sure the compiler covers every new/existing feature that DTensor has.
- End to end compile integration tests with 2D/3D parallelism to guard: 1. Compile time regression 2. Performance regression

# Future Works

Since DTensor has been released as public APIs, and many users have started trying it out, there would be more and more issues raised going forward. Working directly on user reported issues is important to address the community needs and make it stable , i.e. a list of issues here. Here I listed the features/improvements we should do to bring DTensor to stable.

## Collective Operators

As of today, when users want to call a collective on a DTensor, it basically does not work (i.e. throw error on op not supported) and our response is that we have a redistribute API that users could specify the sharding and it would figure out the right collective. This is good enough for many users who don’t care about the underlying collective implementation and just want to deal with the sharding as it’s simpler.

However, many system engineers are already familiar with the collective concept, and they want to use DTensor, together with explicit collective APIs. For example, they want to leverage DTensor to handle the sharding complexity, but when optimizing for some certain workloads, write manual collective is the most familiar interface. `local_map` could do the work, but directly make collective operator work on DTensor require no new API education.

What we should do is that we can support certain collective operator (start with functional collective) directly with DTensor inputs, DTensor just need to overwrite collective operators and do some sanity checks:

1. For the `process_group` arg, make sure it matches with the DTensor’s device_mesh (i.e. make sure they are the same communicator)
2. For input DTensors to the collective op, we could write proper sharding strategy:
3. `all_gather_into_tensor` : `Shard(dim)` input and produce `Replicate()` output
4. `all_reduce` : `Partial()` input and produce `Replicate()` output
5. `reduce_scatter` : `Partial()` input and produce `Shard(dim)` output
6. `shard_dim_all_to_all` : `Shard(src_dim)` input and produce `Shard(dst_dim)` output

## Batched collectives

Batching collectives is a very important optimization for per-parameter sharding, i.e. FSDP2 needs an efficient batched all_gather in order to prefetch the next bucket. However this is currently FSDP2 only and not many advanced users could leverage it. To make optimizations more approachable for advanced use cases, we should add batched collective support, i.e.:

- Add `allgather_copy_in` support, which takes in a list of Shard DTensor
- Add `allgather_copy_out` support, which produce a list of Replicate DTensor
- Add the `reduce_scatter` equivalent batched collective support
- Add the `all_reduce` equivalent batched collective support

## Uneven sharding + padding

One issue of the current implementation we have is the un-even padding behavior, where for a parameter dimension that is not divisible by the mesh dimension size, by default it does not pad the local shard, and only pad when we perform collectives like `allgather` (or `redistribute` ). This approach is easier to implement and that was the main reason why we chose this approach to prototype. However this approach suffers from the fact that it incurs additional copy, and when we want to group many parameters together (i.e. FSDP algorithm for communication overlap), we need to pad several parameters and the copies become expensive.

For this reason, we should change our sharding to be padded by default. This means we need to:

1. `distribute_tensor` should by default first pad the tensor to the multiples of the mesh dimension size, then we shard the tensors and scatter them.
2. The `_local_tensor` of DTensor should always stay padded, so that the communication operations can directly use the local tensor without incurring copy.
3. `DTensorSpec` could record/compute whether the `_local_tensor` of its current rank is padded or not, and how to unpad the `_local_tensor` , and this can be a method/property of `DTensorSpec`
4. Distributed operators now need to reason the padded data, this part can be quite complicated as it is operator specific (i.e. different reduction operators would need different padding values to make the math right). Framework like GSPMD enumerate all the operators and make the padding behavior be correct in math. Since PyTorch have a huge operator base, this approach might not be feasible
5. What we can do instead is that we can only keep the `_local_tensor` (or data) be padded by default, but we dynamically `unpad` (similar to `redistribute_local_tensor` ). This means when we perform DTensor’s sharding propagation, we record whether the inputs need to unpad or not, and first `unpad` the `_local_tensor` before doing the actual computation.

In this way, DTensor would stay padded by default. This could be useful for explorations like SimpleFSDP, to not worry about the additional padding impacting the performance.

## Support Customizations

A good programming model could let users utilize the higher level abstraction, and it does not prevent users from getting into things deeper and doing whatever things they want to do. We would need to focus on supporting the customizations and the PyTorch extension subsystem, because:

1. The extension system is popular in many cases, and be customizable is important for advanced customers
2. Many innovations happen in this space, i.e. users write optimization operators (i.e, `FusedRMSNorm` ) and then register as a PyTorch operator or simply call them in their model. If DTensor be blind to those features, it would be hard for users to leverage DTensor.
3. Supporting better customization allows us to develop in pace and still unblock users. With the right extension point, we don’t need to implement every feature for users.

Three important customizations point we are introducing:

1. `register_sharding` : harden this API to be stable, make the user experience better.
2. `local_map` : allow users to write manual collectives or kernels without registering as an op. We should collect feedback from users whether this API is flexible enough.
3. `Collectives/Custom collectives` : we discussed in previous sections, system developers who want to call manual collectives should still be able to and expect the behavior to be consistent!

## Torch Operator Coverage

There’re around 2000+ operators in PyTorch, and currently DTensor supports many popular ones (i.e. 300+ ops), but a bit far from covering the full PyTorch operator set.

Writing sharding strategies for every operator is not an easy task. Although arguably we can parallelize the work, it would still be better to figure out a novel approach to help accelerate the work, for the reasons:

1. Some PyTorch operators have complicated math semantics (i.e. conv, spda), and to figure out the correct sharding strategies requires non-trival amount of work
2. Writing all sharding strategies for all existing operators requires maintaining those sharding strategies and fixing bugs 1 by 1, resulting in significant engineering effort.

Therefore it would be good to use some clever method to drive the operator coverage. Tianyu Liu and I worked on a prototype that could leverage the decomposition functions developed in PT2, to perform a sharding propagation on the decomposed function directly. This could significantly help with sharding strategy authoring, as we have most basic math operators already enabled, and for complicated math operators, we can simply leverage the decomposed sharding propagation to produce the sharding strategy.

## Shard Placement Order

We have implemented a private `_StridedShard` placement as part of our effort to make FSDP2 + TP state_dict behave correctly. `_StridedShard` differs from the normal Shard placement in that it has a split_factor where it records the sharding happened in a strided way. `_StridedShard` placement unblocks the 2D sharding for FSDP2 + TP, but it is hard for users to reason about what’s a strided sharding.

The real underlying motivation we want to introduce a “strided” sharding, it is because for the n-D mesh sharding, sharding on the same tensor dimension on different device mesh dimensions could have different order!

We should strive for simplicity! Instead of introducing a separate placement and let users specify a split_factor, which is hard to determine. We could do it in a simpler and easy to understand way. The new approach could be:

- Adds an optional `shard_order` list to DTensorSpec, to describe the “placement” to “mesh dimension” order. For normal sharding it would be None (or [1, 2, 3…])
- For the strided sharding case, we would simply know it from this shard_order list that the same tensor dimension got sharded on multiple mesh dimensions, and the shard order is strided
- All the relevant _StridedShard communication logic could be reused for this shard_order approach, and it does not require a separate Placement.
