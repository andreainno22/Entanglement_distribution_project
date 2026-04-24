# ================================================================
# Assignment 4 - Part 1: Entanglement Distribution
# Quantum Network Simulation
# ================================================================
#
# This script implements the topology using QuantumSavory's
# with ProtocolZoo components:
# - EntanglerProt for link-level Bell-pair generation
# - SwapperProt for repeater-side entanglement swapping
# - EntanglementTracker to propagate swap updates/corrections
#
# Then, an application process (Poisson requests) consumes end-to-end pairs
# on A-C and B-D and records the resulting fidelity.
# ================================================================

#todo: fare test sia con Purging delle coppie sia senza, vedere come cambia throughput, fidelities, numero di coppie totali generate. 

using ConcurrentSim
using ResumableFunctions
using QuantumSavory
using QuantumSavory.ProtocolZoo
using Distributions
using Statistics
using Random
using Graphs
using Logging

# ================================================================
# SECTION 1 - MODEL PARAMETERS
# ================================================================

# Average successful generation rate target per physical link [pairs/s].
# In EntanglerProt, this is translated into attempt_time so the expected
# success process has approximately this average rate.
const BELL_RATE = 100.0

# Depolarization parameter for each freshly generated Bell pair.
# We use a Werner state model:
#   rho = (1 - p) * |Phi+><Phi+| + p * I/4
const P_DEPOL = 0.02

# Memory dephasing coherence time [s] used by T2Dephasing background.
const T2 = 0.5

# Memory slots (qubits) per node.
const N_MEM = 5

# Application-layer request arrival rate [requests/s] for each flow.
# We instantiate one process for A-C and one for B-D.
const APP_RATE = 0.47 * BELL_RATE

# Maximum time a request waits for a target end-to-end pair.
# Inf means blocking wait until a pair appears.
const APP_MAX_WAIT = Inf

# Polling step used while waiting for a target pair.
const APP_WAIT_POLL_DT = 1e-3

# Total simulation time [s].
const SIM_DURATION = 10.0

# Node identifiers in the dumbbell graph.
const NODE_A  = 1
const NODE_B  = 2
const NODE_R1 = 3
const NODE_R2 = 4
const NODE_C  = 5
const NODE_D  = 6
const N_NODES = 6

# Physical links in canonical (min, max) node order.
const PHYSICAL_LINKS = Set([(NODE_A, NODE_R1),
                            (NODE_B, NODE_R1),
                            (NODE_R1, NODE_R2),
                            (NODE_R2, NODE_C),
                            (NODE_R2, NODE_D)])

# Ideal Bell state in computational basis.
# Z1 == |0>, Z2 == |1> in QuantumSavory symbolic notation.
const Φ⁺ = (Z₁ ⊗ Z₁ + Z₂ ⊗ Z₂) / sqrt(2)
const Φ⁺_DM = SProjector(Φ⁺)

function validated_probability(p::Real, name::AbstractString)
    if !isfinite(p)
        throw(ArgumentError("$name must be finite, got $p"))
    end
    if p < 0.0 || p > 1.0
        throw(ArgumentError("$name must be in [0, 1], got $p"))
    end
    return Float64(p)
end

const P_DEPOL_VALID = validated_probability(P_DEPOL, "P_DEPOL")


# Link-level imperfect pair model used by EntanglerProt(pairstate=...).
const WERNER_PAIR = (1.0 - P_DEPOL_VALID) * Φ⁺_DM + P_DEPOL_VALID * (I ⊗ I) / 4

canonical_nodes(u::Int, v::Int) = u < v ? (u, v) : (v, u)

function _counterpart_pair_key(local_node::Int, local_slot::Int, remote_node::Int, remote_slot::Int)
    a = (local_node, local_slot)
    b = (remote_node, remote_slot)
    return a < b ? (a[1], a[2], b[1], b[2]) : (b[1], b[2], a[1], a[2])
end

function _snapshot_counterpart_pairs(net::RegisterNet)
    pairs = Set{NTuple{4, Int}}()
    for node in 1:N_NODES
        for q in queryall(net[node], EntanglementCounterpart, ❓, ❓; locked=false, assigned=true)
            push!(pairs, _counterpart_pair_key(node, q.slot.idx, q.tag[2], q.tag[3]))
        end
    end
    return pairs
end

"""
    monitor_pair_counters(sim, net, counters; poll_dt=1e-3)

Runtime monitor that tracks newly appeared entangled pairs.
- `generated_est`: new pairs on physical links
- `swaps_est`: new pairs on non-physical links

These are event estimates useful for debugging bottlenecks.
"""
@resumable function monitor_pair_counters(sim::Simulation,
                                          net::RegisterNet,
                                          counters::Dict{Symbol, Int};
                                          poll_dt::Float64=1e-3)
    seen_pairs = _snapshot_counterpart_pairs(net)
    while true
        @yield timeout(sim, poll_dt)
        current_pairs = _snapshot_counterpart_pairs(net)
        for pair in current_pairs
            if pair in seen_pairs
                continue
            end
            u, _, v, _ = pair
            if canonical_nodes(u, v) in PHYSICAL_LINKS
                counters[:generated_est] += 1
            else
                counters[:swaps_est] += 1
            end
        end
        seen_pairs = current_pairs
    end
end

# ================================================================
# SECTION 2 - NETWORK CONSTRUCTION
# ================================================================

"""
    build_dumbbell_net(; n_mem=N_MEM) -> RegisterNet

Create the fixed 6-node dumbbell topology.
Each node is a `Register(n_mem, T2Dephasing(T2))`, i.e. all local memory slots
share the same T2 dephasing background.
"""
function build_dumbbell_net(; n_mem::Int=N_MEM)
    g = SimpleGraph(N_NODES)

    # Physical links.
    add_edge!(g, NODE_A, NODE_R1)
    add_edge!(g, NODE_B, NODE_R1)
    add_edge!(g, NODE_R1, NODE_R2)
    add_edge!(g, NODE_R2, NODE_C)
    add_edge!(g, NODE_R2, NODE_D)

    # One register per node, each with n_mem memory slots.
    regs = [Register(n_mem, T2Dephasing(T2)) for _ in 1:N_NODES]

    return RegisterNet(g, regs)
end

# ================================================================
# SECTION 3 - APPLICATION LAYER (POISSON CONSUMER)
# ================================================================

"""
    app_process(sim, net, node_query, node_target, label, fids, rng)

Poisson request process for one end-to-end pair between `node_query` (A or B) and `node_target` (C or D).

Behavior per request:
1. Query for one available tagged pair (EntanglementCounterpart) on the query side.
2. Verify reciprocal tag on the target side.
3. Lock both slots, re-check and remove tags atomically.
4. Measure fidelity F = <Phi+|rho|Phi+>.
5. Store F in `fids` and consume pair via traceout!.

If no pair is ready, the request is considered not served.
"""
@resumable function app_process(sim::Simulation,
                                net::RegisterNet,
                                label::String,
                                node_query::Int,
                                node_target::Int,
                                target_fids::Vector{Float64},
                                rng_poisson::AbstractRNG,
                                counters::Union{Nothing,Dict{Symbol, Int}}=nothing,
                                app_logs::Bool=false)
    # Exponential inter-arrivals -> Poisson request stream.
    dist = Exponential(1.0 / (APP_RATE/2)) # MOD: divide by 2 because we have two independent processes generating requests at the same rate

    while true
        # Wait for next request arrival.
        @yield timeout(sim, rand(rng_poisson, dist))
        req_t0 = now(sim)
        t = req_t0

        if label == "A-C"
            counter_suffix = :AC
        else
            counter_suffix = :BD
        end
        
        
        isnothing(counters) || (counters[Symbol("requests_$counter_suffix")] += 1)

        app_logs && @info("[t=$(round(t, digits=3))s] APP REQUEST [$label]")

        # Wait (possibly indefinitely) until a consistent target pair is available.
        q1 = nothing
        q2 = nothing
        saw_candidate_without_reciprocal = false
        while true
            # Query one local slot currently tagged as entangled with nodeB.
            q1 = query(net[node_query], EntanglementCounterpart, node_target, ❓; locked=false, assigned=true)
            if !isnothing(q1)
                # Query the reciprocal metadata on nodeB.
                q2 = query(net[node_target], EntanglementCounterpart, node_query, q1.slot.idx; locked=false, assigned=true)
                if !isnothing(q2)
                    break
                end
                saw_candidate_without_reciprocal = true
            end

            waited = now(sim) - req_t0
            if waited >= APP_MAX_WAIT
                if saw_candidate_without_reciprocal
                    isnothing(counters) || (counters[Symbol("miss_unsynced_$counter_suffix")] += 1)
                    app_logs && @info("  -> [$label] Waited $(round(waited, digits=4))s; counterpart metadata still unsynchronized.")
                else
                    isnothing(counters) || (counters[Symbol("miss_no_pair_$counter_suffix")] += 1)
                    app_logs && @info("  -> [$label] Waited $(round(waited, digits=4))s; no target pair became available.")
                end
                q1 = nothing
                q2 = nothing
                break
            end

            # Event-driven wait: wake up when tags change on either endpoint.
            tag_change_evt = QuantumSavory.onchange_tag(net[node_query]) | QuantumSavory.onchange_tag(net[node_target])
            if isinf(APP_MAX_WAIT)
                @yield tag_change_evt
            else
                @yield tag_change_evt | timeout(sim, APP_MAX_WAIT - waited)
            end
        end

        if isnothing(q1) || isnothing(q2)
            continue
        end

        isnothing(counters) || (counters[Symbol("ready_$counter_suffix")] += 1)

        # Lock both endpoints before consuming. Lock ensure the slot is not modified by concurrent processes (e.g. swappers) while we check and consume it.
        @yield lock(q1.slot) & lock(q2.slot)

        # Remove reciprocal tags under lock to guarantee consistency.
        t1 = querydelete!(q1.slot, EntanglementCounterpart, node_target, q2.slot.idx)
        t2 = querydelete!(q2.slot, EntanglementCounterpart, node_query, q1.slot.idx)
        if isnothing(t1) || isnothing(t2)
            unlock(q1.slot)
            unlock(q2.slot)
            isnothing(counters) || (counters[Symbol("miss_changed_$counter_suffix")] += 1)
            app_logs && @info("  -> [$label] Pair changed while locking - retry later.")
            continue
        end

        # Compute fidelity at current simulation time; using an earlier request timestamp
        # can force a negative Δt in background updates when the request waited in queue.
        t_obs = now(sim)
        F = real(observable((q1.slot, q2.slot), Φ⁺_DM; something=0.0, time=t_obs))
        F = clamp(F, 0.0, 1.0)
        push!(target_fids, F)
        isnothing(counters) || (counters[Symbol("delivered_$counter_suffix")] += 1)

        app_logs && @info("  -> [$label] DELIVERED F = $(round(F, digits=4))")

        # Consume pair and release both memories.
        traceout!(q1.slot, q2.slot)
        unlock(q1.slot)
        unlock(q2.slot)
    end
end

# ================================================================
# SECTION 4 - MAIN SIMULATION DRIVER
# ================================================================

"""
    run_simulation(; duration=SIM_DURATION, seed=42, verbose=true, n_mem=N_MEM)

Build and execute the complete ProtocolZoo stack on the dumbbell topology:
- EntanglerProt on each physical link
- SwapperProt at repeaters R1 and R2
- EntanglementTracker on all nodes
- Poisson application consumers for A-C and B-D

Returns named tuple:
  (net=..., fidelities_AC=..., fidelities_BD=...)
"""
function run_simulation(; duration::Float64=SIM_DURATION,
                          seed::Int=42,
                          verbose::Bool=true,
                          n_mem::Int=N_MEM,
                          app_logs::Bool=false)
    rng_poisson = MersenneTwister(seed)
    net = build_dumbbell_net(n_mem=n_mem)

    # RegisterNet owns the underlying ConcurrentSim time tracker.
    sim = get_time_tracker(net)

    if verbose
        println("""
+----------------------------------------------------------+
| Quantum Network Simulation                               |
+----------------------------------------------------------+
| BELL_RATE = $BELL_RATE pairs/s per link                  
| P_DEPOL   = $P_DEPOL                                     
| T2        = $T2 s                                        
| N_MEM     = $n_mem                                       
| APP_RATE  = $APP_RATE req/s per flow                     
| APP_WAIT  = $APP_MAX_WAIT s                          
| SIM_TIME  = $duration s                                  
+----------------------------------------------------------+
""")
    end

    # ------------------------------------------------------------
    # Link-layer entanglement generation
    # ------------------------------------------------------------
    # We run one EntanglerProt per physical edge. Setting rate=BELL_RATE
    # maps to the expected mean generation rate defined in the assignment.
    for (u, v) in [(NODE_A, NODE_R1),
                   (NODE_B, NODE_R1),
                   (NODE_R1, NODE_R2),
                   (NODE_R2, NODE_C),
                   (NODE_R2, NODE_D)]
        
        # todo: capire per bene perchè con margin 3 funziona e con 2 no
        # one entangler process per physical link
        ent = EntanglerProt(sim, net, u, v;
                            rate=BELL_RATE,
                            pairstate=WERNER_PAIR,
                            success_prob=1.0,    # MOD: forza il successo del singolo attempt
                            randomize=true,      # MOD: riduce contese deterministiche sugli stessi slot
                            margin=3,            # MOD: evita monopolio completo della memoria sul link
                            hardmargin=1,        # MOD: lascia sempre almeno 1 slot libero (anti-starvation)
                            retry_lock_time=1e-3 # MOD: backoff breve quando non si riesce a lockare
                            )
        @process ent()
    end

    # ------------------------------------------------------------
    # Network-layer swapping at repeaters
    # ------------------------------------------------------------
    # R1: combine links from {A,B} toward R2.
    # todo: consider adding NODE_C and NODE_D as possible nodeH for R1, to allow direct swapping from A/B to C/D at R1 when R2 is busy.
    # * Lock contention warning: The entangler's aggressive retry interval starves the swapper, 
    # * preventing it from acquiring the two simultaneous locks needed for an entanglement swap.
    swap_r1 = SwapperProt(sim, net, NODE_R1;
                          nodeL=(n -> n == NODE_A || n == NODE_B),
                          nodeH=(n -> n == NODE_R2), # MOD: allow R1 to swap directly toward C/D when R2 is busy
                          retry_lock_time=1e-4)

    # R2: combine incoming entanglement from {R1,A,B} toward {C,D}.
    swap_r2_AC = SwapperProt(sim, net, NODE_R2;
                          nodeL=(n -> n == NODE_A), # R2 can be entangled both with R1 and directly with A/B due to swapping at R1
                          nodeH=(n -> n == NODE_C),
                          retry_lock_time=1e-4)
    
    swap_r2_BD = SwapperProt(sim, net, NODE_R2;
                          nodeL=(n -> n == NODE_B), # R2 can be entangled both with R1 and directly with A/B due to swapping at R1
                          nodeH=(n -> n == NODE_D),
                          retry_lock_time=1e-4)

    @process swap_r1()
    @process swap_r2_AC()
    @process swap_r2_BD()

    # ------------------------------------------------------------
    # Metadata/correction tracking on all nodes
    # ------------------------------------------------------------
    # EntanglementTracker listens to swap update messages and keeps
    # counterpart tags and Pauli-frame corrections coherent network-wide.
    for node in 1:N_NODES
        tracker = EntanglementTracker(sim, net, node)
        @process tracker()
    end

    # ------------------------------------------------------------
    # Application layer processes
    # ------------------------------------------------------------
    fids_ac = Float64[]
    fids_bd = Float64[]
    counters = Dict{Symbol, Int}(
        :generated_est => 0,
        :swaps_est => 0,
        :requests_AC => 0,
        :requests_BD => 0,
        :ready_AC => 0,
        :ready_BD => 0,
        :miss_no_pair_AC => 0,
        :miss_no_pair_BD => 0,
        :miss_unsynced_AC => 0,
        :miss_unsynced_BD => 0,
        :miss_changed_AC => 0,
        :miss_changed_BD => 0,
        :delivered_AC => 0,
        :delivered_BD => 0,
    )

    # Monitor to estimate where throughput is bottlenecked (generation vs swapping).
    @process monitor_pair_counters(sim, net, counters)

    # Process for each target pair. They run independently and concurrently, generating Poisson requests at the same APP_RATE/2.
    @process app_process(sim, net, "A-C", NODE_A, NODE_C, fids_ac, rng_poisson, counters, app_logs)
    
    @process app_process(sim, net, "B-D", NODE_B, NODE_D, fids_bd, rng_poisson, counters, app_logs)

    # Run event-driven simulation.
    run(sim, duration)

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    if verbose
        println("\n" * "="^60)
        println("SIMULATION RESULTS (T = $duration s)")
        println("="^60)
        for (label, fids) in [("A-C", fids_ac), ("B-D", fids_bd)]
            if isempty(fids)
                println("$label : no pairs delivered.")
            else
                println("$label : $(length(fids)) delivered pairs")
                println("  mean F = $(round(mean(fids), digits=4))")
                println("  std  F = $(round(std(fids), digits=4))")
                println("  min  F = $(round(minimum(fids), digits=4))")
                println("  max  F = $(round(maximum(fids), digits=4))")
            end
        end
        println("--- DEBUG COUNTERS ---")
        println("generated_est (physical-link pairs created) = $(counters[:generated_est])")
        println("swaps_est (non-physical pairs created)      = $(counters[:swaps_est])")
        println("requests_AC                                  = $(counters[:requests_AC])")
        println("requests_BD                                  = $(counters[:requests_BD])")
        println("ready_AC (target pair present at request)    = $(counters[:ready_AC])")
        println("ready_BD (target pair present at request)    = $(counters[:ready_BD])")
        println("miss_no_pair_AC                              = $(counters[:miss_no_pair_AC])")
        println("miss_no_pair_BD                              = $(counters[:miss_no_pair_BD])")
        println("miss_unsynced_AC                             = $(counters[:miss_unsynced_AC])")
        println("miss_unsynced_BD                             = $(counters[:miss_unsynced_BD])")
        println("miss_changed_AC                              = $(counters[:miss_changed_AC])")
        println("miss_changed_BD                              = $(counters[:miss_changed_BD])")
        println("delivered_AC                                = $(counters[:delivered_AC])")
        println("delivered_BD                                = $(counters[:delivered_BD])")
        println("="^60)
    end

    return (net=net,
            fidelities_AC=fids_ac,
            fidelities_BD=fids_bd,
            counters=counters)
end

# ================================================================
# SECTION 5B - STRUCTURED SWEEP (MEMORY SLOTS)
# ================================================================

"""
    sweep_memory_slots(mem_values=[5, 10, 15, 20]; repetitions=30,
                       duration=SIM_DURATION, base_seed=42)

Run a structured benchmark over memory size per node.
For each value in `mem_values`, run `repetitions` independent simulations and
report aggregate throughput and fidelity statistics for A-C and B-D.
"""
function sweep_memory_slots(mem_values::Vector{Int}=[5, 10, 15, 20];
                            repetitions::Int=30,
                            duration::Float64=SIM_DURATION,
                            base_seed::Int=42)
    println("\nStructured memory sweep (duration=$(duration)s, repetitions=$(repetitions))")
    println("-"^98)
    println("mem | mean delivered A-C | mean delivered B-D | mean F(A-C) | mean F(B-D) | total pairs")
    println("-"^98)

    rows = NamedTuple[]

    for n_mem in mem_values
        delivered_ac = Int[]
        delivered_bd = Int[]
        all_fids_ac = Float64[]
        all_fids_bd = Float64[]

        for rep in 1:repetitions
            seed = base_seed + 10_000 * n_mem + rep
            out = run_simulation(duration=duration, seed=seed, verbose=false, n_mem=n_mem)

            push!(delivered_ac, length(out.fidelities_AC))
            push!(delivered_bd, length(out.fidelities_BD))
            append!(all_fids_ac, out.fidelities_AC)
            append!(all_fids_bd, out.fidelities_BD)
        end

        mean_delivered_ac = mean(delivered_ac)
        mean_delivered_bd = mean(delivered_bd)
        mean_f_ac = isempty(all_fids_ac) ? NaN : mean(all_fids_ac)
        mean_f_bd = isempty(all_fids_bd) ? NaN : mean(all_fids_bd)
        total_pairs = length(all_fids_ac) + length(all_fids_bd)

        println("$(lpad(n_mem, 3)) | $(lpad(round(mean_delivered_ac, digits=2), 19)) | $(lpad(round(mean_delivered_bd, digits=2), 19)) | $(lpad(round(mean_f_ac, digits=4), 11)) | $(lpad(round(mean_f_bd, digits=4), 11)) | $(lpad(total_pairs, 10))")

        push!(rows, (
            n_mem=n_mem,
            repetitions=repetitions,
            duration=duration,
            mean_delivered_AC=mean_delivered_ac,
            mean_delivered_BD=mean_delivered_bd,
            mean_fidelity_AC=mean_f_ac,
            mean_fidelity_BD=mean_f_bd,
            total_pairs=total_pairs,
        ))
    end

    println("-"^98)
    return rows
end

# ================================================================
# SECTION 5 - PARAMETER SWEEP (DEPOLARIZATION)
# ================================================================

"""
    sweep_depol(p_values=0.0:0.05:0.30; duration=SIM_DURATION, seed=42)

Run multiple simulations by varying depolarization parameter p_u and print:
- mean fidelity for A-C
- mean fidelity for B-D

For each p_u, link-level generated pairs use:
  rho = (1 - p_u)|Phi+><Phi+| + p_u I/4
"""
function sweep_depol(p_values=0.0:0.05:0.30; duration::Float64=SIM_DURATION, seed::Int=42)
    println("\nDepolarizing sweep: p_u | mean F(A-C) | mean F(B-D)")
    println("-"^48)

    rows = NamedTuple[]

    for p in p_values
        p_valid = validated_probability(p, "sweep p_u")
        local_state = (1.0 - p_valid) * Φ⁺_DM + p_valid * (I ⊗ I) / 4

        rng = MersenneTwister(seed)
        net = build_dumbbell_net()
        sim = get_time_tracker(net)

        # Entanglers with overridden pairstate for this sweep point.
        for (u, v) in [(NODE_A, NODE_R1),
                       (NODE_B, NODE_R1),
                       (NODE_R1, NODE_R2),
                       (NODE_R2, NODE_C),
                       (NODE_R2, NODE_D)]
            
            # start of the EntanglerProt
            @process EntanglerProt(sim, net, u, v;
                                   rate=BELL_RATE,
                                   pairstate=local_state,
                                   success_prob=1.0,    # MOD: forza il successo del singolo attempt
                                   randomize=true,      # MOD: riduce contese deterministiche sugli stessi slot
                                   margin=1,            # MOD: evita monopolio completo della memoria sul link
                                   hardmargin=1,        # MOD: lascia sempre almeno 1 slot libero (anti-starvation)
                                   retry_lock_time=1e-4 # MOD: backoff breve quando non si riesce a lockare
                                   )()
        end

        # Same swapping configuration as base run.
        @process SwapperProt(sim, net, NODE_R1;
                             nodeL=(n -> n == NODE_A || n == NODE_B),
                             nodeH=(n -> n == NODE_R2),
                             retry_lock_time=1e-3)()

        @process SwapperProt(sim, net, NODE_R2;
                             nodeL=(n -> n == NODE_R1 || n == NODE_A || n == NODE_B),
                             nodeH=(n -> n == NODE_C || n == NODE_D),
                             retry_lock_time=1e-3)()

        # Trackers are required for correct update propagation.
        for node in 1:N_NODES
            @process EntanglementTracker(sim, net, node)()
        end

        # Application processes for both target pairs.
        fids_ac = Float64[]
        fids_bd = Float64[]
        @process app_process(sim, net, NODE_A, NODE_C, "A-C", fids_ac, rng)
        @process app_process(sim, net, NODE_B, NODE_D, "B-D", fids_bd, rng)

        run(sim, duration)

        mAC = isempty(fids_ac) ? NaN : round(mean(fids_ac), digits=4)
        mBD = isempty(fids_bd) ? NaN : round(mean(fids_bd), digits=4)

        println("$(rpad(round(p, digits=2), 5)) | $mAC       | $mBD")
        push!(rows, (p_u=p, F_AC=mAC, F_BD=mBD))
    end

    return rows
end

# ================================================================
# SECTION 6 - ENTRY POINT
# ================================================================

# Run one baseline simulation.
# invokelatest avoids world-age issues in debugger/Revise workflows.
result = Base.invokelatest(run_simulation;
                           duration=SIM_DURATION,
                           seed=42,
                           verbose=true,
                           n_mem=N_MEM,
                           app_logs=true)

# Uncomment to run depolarization sweep:
# sweep_results = sweep_depol(0.0:0.02:0.20)
