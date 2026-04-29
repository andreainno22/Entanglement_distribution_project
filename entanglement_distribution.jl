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


using ConcurrentSim
using ResumableFunctions
using QuantumSavory
using QuantumSavory.ProtocolZoo
using Distributions
using HypothesisTests
using Statistics
using StatsAPI
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

# Cutoff time for memory pairs [s].
const CUT_OFF_TIME = 0.1

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

# Node identifiers in the graph.
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
        # can force a negative delta t in background updates when the request waited in queue.
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
end#


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
                          app_logs::Bool=false,
                          cutoff_time::Float64=CUT_OFF_TIME,
                          p_depol::Float64=P_DEPOL)
    rng_poisson = MersenneTwister(seed)
    net = build_dumbbell_net(n_mem=n_mem)
    p_depol_valid = validated_probability(p_depol, "p_depol")
    werner_pair = (1.0 - p_depol_valid) * Φ⁺_DM + p_depol_valid * (I ⊗ I) / 4

    # RegisterNet owns the underlying ConcurrentSim time tracker.
    sim = get_time_tracker(net)

    # Ensure the swapper ignores pairs that will be deleted soon
    agelimit = isfinite(cutoff_time) ? 0.2 * cutoff_time : nothing

    if verbose
        println("""
+----------------------------------------------------------+
| Quantum Network Simulation                               |
+----------------------------------------------------------+
| BELL_RATE = $BELL_RATE pairs/s per link                  
| P_DEPOL   = $p_depol_valid                               
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
        # margin = 3 ensure that in a reapeater at most 2 slots can be occupied 
        # by pairs toward one side, allowing having 3 different possible types of pairs 
        # (toward A, toward B, toward R2 for R1; toward R1, toward C, toward D for R2) 
        ent = EntanglerProt(sim, net, u, v;
                            rate=BELL_RATE,
                            pairstate=werner_pair,
                            randomize=true,      # Reduces contention on the same slot
                            margin=3,            # Avoids the monopoly of entanglement slots by a single neighbor in the repeater, preventing deadlock
                            hardmargin=1,        # Similar to margin (anti-deadlock/starvation)
                            retry_lock_time=1e-3 
                            )
        @process ent()
    end

    # ------------------------------------------------------------
    # Network-layer swapping at repeaters
    # ------------------------------------------------------------
    # R1: combine links from {A,B} toward R2.
    # * Lock contention warning: The entangler's aggressive retry interval starves the swapper, 
    # * preventing it from acquiring the two simultaneous locks needed for an entanglement swap.
    # * To mitigate this, we set a shorter retry_lock_time for the swappers.
    swap_r1 = SwapperProt(sim, net, NODE_R1;
                          nodeL=(n -> n == NODE_A || n == NODE_B),
                          nodeH=(n -> n == NODE_R2), # MOD: allow R1 to swap directly toward C/D when R2 is busy
                          retry_lock_time=1e-4,
                          agelimit=agelimit)

    # R2_AC: combine links from R1 toward C.
    swap_r2_AC = SwapperProt(sim, net, NODE_R2;
                          nodeL=(n -> n == NODE_A), # R2 can be entangled both with R1 and directly with A/B due to swapping at R1
                          nodeH=(n -> n == NODE_C),
                          retry_lock_time=1e-4,
                          agelimit=agelimit)
    
    # R2_BD: combine links from R1 toward D.
    swap_r2_BD = SwapperProt(sim, net, NODE_R2;
                          nodeL=(n -> n == NODE_B), # R2 can be entangled both with R1 and directly with A/B due to swapping at R1
                          nodeH=(n -> n == NODE_D),
                          retry_lock_time=1e-4,
                          agelimit=agelimit)
    

    @process swap_r1()
    @process swap_r2_AC()
    @process swap_r2_BD()

    # Cutoff protocol: purge qubits older than cutoff_time.
    if cutoff_time < Inf
        @info "Activation of CutoffProt (t_cut = $(cutoff_time)s)"
        for node in 1:N_NODES
            cutoff = CutoffProt(sim, net, node;
                                period=cutoff_time / 10,
                                retention_time=cutoff_time,
                                announce=true)
            @process cutoff()
        end
    end

    # ------------------------------------------------------------
    # Metadata/correction tracking on all nodes
    # ------------------------------------------------------------
    # EntanglementTracker listens to swap update messages and keeps
    # counterpart tags and Pauli corrections coherent network-wide.
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

function _safe_mean(v::Vector{Float64})
    return isempty(v) ? NaN : mean(v)
end

function _safe_mean_finite(v::Vector{Float64})
    finite_vals = [x for x in v if isfinite(x)]
    return isempty(finite_vals) ? NaN : mean(finite_vals)
end

function _safe_std_finite(v::Vector{Float64})
    finite_vals = [x for x in v if isfinite(x)]
    return length(finite_vals) <= 1 ? NaN : std(finite_vals)
end

function _finite_vals(v::Vector{Float64})
    return [x for x in v if isfinite(x)]
end

function _mean_or_nan(vals::Vector{Float64})
    return isempty(vals) ? NaN : mean(vals)
end


function _paired_ttest(x::Vector{Float64}, y::Vector{Float64})
    """
    __paired_ttest(x, y)

    Perform a paired t-test on two vectors of measurements `x` and `y`.
    The test is performed on the differences `d = x - y`, and the null hypothesis is that the mean of `d` is zero.
    """
    n = min(length(x), length(y))
    if n < 2
        return (t=NaN, df=NaN, p=NaN, n=n, ci_low=NaN, ci_high=NaN)
    end
    d = [x[i] - y[i] for i in 1:n]
    d = _finite_vals(d)
    n = length(d)
    if n < 2
        return (t=NaN, df=NaN, p=NaN, n=n, ci_low=NaN, ci_high=NaN)
    end
    test = OneSampleTTest(d, 0.0)
    ci = StatsAPI.confint(test)
    return (t=test.t, df=test.df, p=StatsAPI.pvalue(test), n=n, ci_low=ci[1], ci_high=ci[2])
end

function _welch_ttest(x::Vector{Float64}, y::Vector{Float64})
    """
    __welch_ttest(x, y)

    Perform Welch's t-test on two vectors of measurements `x` and `y`.
    The test is performed on the null hypothesis that the means of `x` and `y` are equal, without assuming equal variances.
    """
    x = _finite_vals(x)
    y = _finite_vals(y)
    n1 = length(x)
    n2 = length(y)
    if n1 < 2 || n2 < 2
        return (t=NaN, df=NaN, p=NaN, n1=n1, n2=n2, ci_low=NaN, ci_high=NaN)
    end
    test = UnequalVarianceTTest(x, y)
    ci = StatsAPI.confint(test)
    return (t=test.t, df=test.df, p=StatsAPI.pvalue(test), n1=n1, n2=n2, ci_low=ci[1], ci_high=ci[2])
end

function _csv_escape(value)
    s = string(value)
    if occursin('"', s)
        s = replace(s, '"' => "\"\"")
    end
    if occursin(',', s) || occursin('\n', s) || occursin('\r', s)
        return "\"$s\""
    end
    return s
end

function _write_csv(path::AbstractString, header::Vector{String}, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(_csv_escape.(row), ","))
        end
    end
end

"""
    run_structured_campaign(; duration=SIM_DURATION, cutoff_value=CUT_OFF_TIME, base_seed=1, n_repeats=30)

Runs 8 simulations:
1) Grid p_depol in [0.02, 0.05, 0.10] x cutoff_purger in [off, on] -> 6 runs
2) Additional tests with p_depol=0.02, cutoff off, N_MEM in [10, 20] -> 2 runs

Each repetition uses one seed shared across all 8 combinations, so comparisons
between strategies are paired within repetition.
"""
function run_structured_campaign(; duration::Float64=SIM_DURATION,
                                   cutoff_value::Float64=CUT_OFF_TIME,
                                   base_seed::Int=1,
                                   n_repeats::Int=30,
                                   verbose_runs::Bool=false,
                                   output_dir::AbstractString=@__DIR__,
                                   max_retries::Int=5)
    if n_repeats <= 0
        throw(ArgumentError("n_repeats must be > 0, got $n_repeats"))
    end

    configs = NamedTuple[]

    for p in (0.02, 0.05, 0.10)
        push!(configs, (p_depol=p, cutoff_active=false, cutoff_time=Inf, n_mem=N_MEM))
        push!(configs, (p_depol=p, cutoff_active=true, cutoff_time=cutoff_value, n_mem=N_MEM))
    end

    for nm in (10, 20)
        push!(configs, (p_depol=0.02, cutoff_active=false, cutoff_time=Inf, n_mem=nm))
    end

    n_combos = length(configs)
    total_runs = n_combos * n_repeats
    results = NamedTuple[]
    run_idx = 0
    total_failures = 0
    total_retries = 0

    println("Running structured campaign: $n_combos combinations x $n_repeats repetitions = $total_runs runs")

    for rep in 1:n_repeats
        seed = base_seed + rep - 1
        println("\n=== Repetition $rep/$n_repeats (seed=$seed) ===")

        for (combo_idx, cfg) in enumerate(configs)
            run_idx += 1
            cutoff_label = cfg.cutoff_active ? "on" : "off"
            println("[$run_idx/$total_runs] combo $combo_idx/$n_combos | p_depol=$(cfg.p_depol), cutoff=$cutoff_label, n_mem=$(cfg.n_mem)")

            sim_result = nothing
            attempts = 0
            while true
                attempts += 1
                try
                    sim_result = Base.invokelatest(run_simulation;
                                                   duration=duration,
                                                   seed=seed,
                                                   verbose=verbose_runs,
                                                   n_mem=cfg.n_mem,
                                                   app_logs=false,
                                                   cutoff_time=cfg.cutoff_time,
                                                   p_depol=cfg.p_depol)
                    if attempts > 1
                        total_retries += (attempts - 1)
                    end
                    break
                catch e
                    msg = sprint(showerror, e)
                    is_cutoff_err = occursin("EntanglementTracker", msg) ||
                                    occursin("CutoffProt", msg) ||
                                    occursin("EntanglementDelete", msg)
                    if !is_cutoff_err || attempts >= max_retries
                        rethrow(e)
                    end
                    total_failures += 1
                end
            end

            fids_ac = sim_result.fidelities_AC
            fids_bd = sim_result.fidelities_BD
            delivered_total = sim_result.counters[:delivered_AC] + sim_result.counters[:delivered_BD]
            meanF_total = _mean_or_nan(_finite_vals([_safe_mean(fids_ac), _safe_mean(fids_bd)]))

            push!(results, (
                run=run_idx,
                repetition=rep,
                combo_idx=combo_idx,
                seed=seed,
                p_depol=cfg.p_depol,
                cutoff_active=cfg.cutoff_active,
                n_mem=cfg.n_mem,
                delivered_AC=sim_result.counters[:delivered_AC],
                delivered_BD=sim_result.counters[:delivered_BD],
                delivered_total=delivered_total,
                meanF_AC=_safe_mean(fids_ac),
                meanF_BD=_safe_mean(fids_bd),
                meanF_total=meanF_total,
                attempts=attempts
            ))
        end
    end

    println("\nStructured campaign summary by combination (aggregated over repetitions):")
    println("combo | p_depol | cutoff | n_mem | mean_delivered | std_delivered | meanF_AC | meanF_BD")
    summary_rows = Any[]
    for (combo_idx, cfg) in enumerate(configs)
        rows = [r for r in results if r.combo_idx == combo_idx]
        delivered_vals = Float64[r.delivered_total for r in rows]
        meanF_AC_vals = Float64[r.meanF_AC for r in rows]
        meanF_BD_vals = Float64[r.meanF_BD for r in rows]
        cutoff_label = cfg.cutoff_active ? "on" : "off"

        mean_delivered = round(_safe_mean(delivered_vals), digits=3)
        std_delivered = round(_safe_std_finite(delivered_vals), digits=3)
        meanF_AC = round(_safe_mean_finite(meanF_AC_vals), digits=4)
        meanF_BD = round(_safe_mean_finite(meanF_BD_vals), digits=4)

        println("$combo_idx | $(cfg.p_depol) | $cutoff_label | $(cfg.n_mem) | $mean_delivered | $std_delivered | $meanF_AC | $meanF_BD")
        push!(summary_rows, [combo_idx, cfg.p_depol, cutoff_label, cfg.n_mem, mean_delivered, std_delivered, meanF_AC, meanF_BD])
    end

    runs_path = joinpath(output_dir, "campaign_runs.csv")
    summary_path = joinpath(output_dir, "campaign_summary.csv")

    _write_csv(runs_path,
                             ["run", "repetition", "combo_idx", "seed", "p_depol", "cutoff_active", "n_mem",
                                "delivered_AC", "delivered_BD", "delivered_total", "meanF_AC", "meanF_BD", "meanF_total", "attempts"],
                             ([r.run, r.repetition, r.combo_idx, r.seed, r.p_depol, r.cutoff_active, r.n_mem,
                                 r.delivered_AC, r.delivered_BD, r.delivered_total, r.meanF_AC, r.meanF_BD, r.meanF_total, r.attempts] for r in results))

    _write_csv(summary_path,
               ["combo", "p_depol", "cutoff", "n_mem", "mean_delivered", "std_delivered", "meanF_AC", "meanF_BD"],
               summary_rows)

    println("\nSaved CSV results to:")
    println("- $runs_path")
    println("- $summary_path")

    println("\nCutoff-related failures: $total_failures")
    println("Total retries: $total_retries")

    # Statistical tests
    tests_rows = Any[]
    alpha = 0.05

    # Paired cutoff off vs on for each p_depol (N_MEM=5)
    for p in (0.02, 0.05, 0.10)
        off = [r for r in results if r.p_depol == p && !r.cutoff_active && r.n_mem == N_MEM]
        on = [r for r in results if r.p_depol == p && r.cutoff_active && r.n_mem == N_MEM]
        sort!(off, by=r -> r.repetition)
        sort!(on, by=r -> r.repetition)
        x = [r.meanF_total for r in off]
        y = [r.meanF_total for r in on]
        t = _paired_ttest(x, y)
        mean_a = _mean_or_nan(_finite_vals(x))
        mean_b = _mean_or_nan(_finite_vals(y))
        mean_diff = isfinite(mean_a) && isfinite(mean_b) ? mean_a - mean_b : NaN
        reject = isfinite(t.p) ? (t.p < alpha) : false
        push!(tests_rows, ["paired_cutoff", p, "N_MEM=5", "off", "on", t.n, t.t, t.df, t.p, alpha, reject, mean_a, mean_b, mean_diff, t.ci_low, t.ci_high])
    end

    # Welch tests among N_MEM for p_depol=0.02 and cutoff off
    mem_groups = Dict(5 => Float64[], 10 => Float64[], 20 => Float64[])
    for r in results
        if r.p_depol == 0.02 && !r.cutoff_active && haskey(mem_groups, r.n_mem)
            push!(mem_groups[r.n_mem], r.meanF_total)
        end
    end
    mem_pairs = [(5, 10), (5, 20), (10, 20)]
    for (a, b) in mem_pairs
        t = _welch_ttest(mem_groups[a], mem_groups[b])
        mean_a = _mean_or_nan(_finite_vals(mem_groups[a]))
        mean_b = _mean_or_nan(_finite_vals(mem_groups[b]))
        mean_diff = isfinite(mean_a) && isfinite(mean_b) ? mean_a - mean_b : NaN
        reject = isfinite(t.p) ? (t.p < alpha) : false
        push!(tests_rows, ["welch_mem", 0.02, "cutoff_off", string(a), string(b), min(t.n1, t.n2), t.t, t.df, t.p, alpha, reject, mean_a, mean_b, mean_diff, t.ci_low, t.ci_high])
    end

    tests_path = joinpath(output_dir, "campaign_tests.csv")
    _write_csv(tests_path,
               ["test", "p_depol", "context", "group_a", "group_b", "n", "t_stat", "df", "p_value", "alpha", "reject_h0", "mean_a", "mean_b", "mean_diff", "ci_low", "ci_high"],
               tests_rows)
    println("- $tests_path")

    return (configs=configs, runs=results, failures=total_failures, retries=total_retries)
end


# ================================================================
# SECTION 6 - SIMULATION EXECUTION
# ================================================================

# Run one baseline simulation.
#=result = Base.invokelatest(run_simulation;
                           duration=SIM_DURATION,
                           seed=42,
                           verbose=true,
                           n_mem=N_MEM,
                           app_logs=true)=#
# Structured 8-combination campaign:
campaign_results = Base.invokelatest(run_structured_campaign;
                                      duration=SIM_DURATION,
                                      cutoff_value=CUT_OFF_TIME,
                                      base_seed=1,
                                      n_repeats=30,
                                      verbose_runs=false)

