using Random
X = randn(677, 3)

xx1 = getpart(X)


mcd = CovMcd()
xx2 = _partition_data(mcd, X)

xx3 = getpart_views(X)

function _partition_data(model::CovMcd, X::Matrix{Float64})::Vector{Matrix{Float64}}
    n_partitions = isnothing(model.n_partitions) ? (size(X, 1) > 600 ? 5 : 1) : model.n_partitions
    return [X[i:min(i+div(size(X,1), n_partitions)-1, size(X,1)), :] for i in 1:div(size(X,1), n_partitions):size(X,1)]
end

function getpart(X::AbstractMatrix; nmini=300, kmini=5)
    n = size(X, 1)
    ngroup = fld(n, nmini)   # integer division n %/% nmini

    if ngroup < kmini
        # Case 1: fewer groups than kmini → base on ngroup
        # Split n into ngroup parts, equal or +1 remainder
        mm = fld(n, ngroup)              # base group size
        r = n - ngroup * mm              # remainder
        jj = ngroup - r                  # first jj groups have size mm

        mini = vcat(fill(mm, jj), fill(mm+1, ngroup-jj))
        minigr = ngroup * mm + r
    else
        # Case 2: limit to kmini groups of size nmini
        ngroup = kmini
        mini = fill(nmini, kmini)
        minigr = kmini * nmini
    end

    # These variables exist in the R code but are zero and unused
    nhalf = 0
    nrep = 0

    # Random selection of rows
    id = randperm(n)[1:minigr]

    parts = Vector{Matrix{Float64}}(undef, ngroup)
    kk = 1
    for i in 1:ngroup
        inds = id[kk:(kk + mini[i] - 1)]
        parts[i] = X[inds, :]   # copy (like in R)
        kk += mini[i]

        println("\n\nGROUP = $i")
        println(inds)
    end

    print("\nPartitioning n = $n into at most kmini groups: ngroup = $ngroup  minigr = $minigr")
    print("  nhalf = $nhalf  nrep = $nrep\nGroup sizes: (")
    for m in mini
        print("$m ")
    end
    println(")\n")

    return parts
end

function getpart_views(X::AbstractMatrix; nmini=300, kmini=5)
    n = size(X, 1)
    ngroup = fld(n, nmini)

    if ngroup < kmini
        mm = fld(n, ngroup)
        r = n - ngroup * mm
        jj = ngroup - r
        mini = vcat(fill(mm, jj), fill(mm+1, ngroup-jj))
        minigr = ngroup * mm + r
    else
        ngroup = kmini
        mini = fill(nmini, kmini)
        minigr = kmini * nmini
    end

    id = randperm(n)[1:minigr]

    # Step 1: Create zero-copy views
    parts_views = Vector{SubArray}(undef, ngroup)
    kk = 1
    for i in 1:ngroup
        inds = id[kk:(kk + mini[i] - 1)]
        parts_views[i] = @view X[inds, :]
        kk += mini[i]
    end

    # Step 2: Materialize copies → now type is Vector{Matrix}
    return [Matrix(v) for v in parts_views]
end