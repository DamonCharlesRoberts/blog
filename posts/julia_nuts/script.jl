# Script for Turing.jl benchmarks
using BenchmarkTools;
using DataFrames;
using DuckDB;
using Turing;

#  Define the number of times to evaluate the functions.
const times = 100;

# Get the data.
function get_data()
    # Connect to the DB.
    con = DBInterface.connect(DuckDB.DB, "~/Desktop/mlb_pred/data/twenty_five.db")
    # Pull in the data as a  dataframe.
    df = DataFrame(
        DBInterface.execute(
            con 
            , """
            with a as (
                select
                    scores.game_id
                    , schedule.home_team
                    , scores.home_runs
                    , schedule.away_team
                    , scores.away_runs
                from scores
                    left join schedule
                    on scores.game_id=schedule.game_id
                where schedule.season_id like '2025'
            )
            select
                game_id
                , teams.team_abbr
                , dense_rank() over(order by home_team) as home_team
                , dense_rank() over(order by away_team) as away_team
                , (case
                    when home_runs > away_runs then 1
                    else 0
                end) as home_win
            from a
                left join teams
                on a.home_team=teams.team_id
            where teams.season_id like '2025'
            """
        )
    )
    # Close the connection.
    DBInterface.close(con)
    return df
end;

# Benchmark the data retreival.
get_data();
run(@benchmarkable get_data() samples=times evals=1)

# Get the dataframe object.
df = get_data();

# Get a vector of key-value pairs  to map team abbreviations to team ids.
function get_ids(df)
    ids = Dict(Pair.(df.team_abbr, df.home_team))
end

# Benchmark the creation of the vector.
get_ids(df);
run(@benchmarkable get_ids(df) samples=times evals=1)

# Get the ids vector.
ids = get_ids(df);

# The model.
# Define the functions.
@model function turing(x, y, d)
    # Prior for the ability parameter
    # for each team (length of d).
    # HalfNormal.
    α ~ filldist(truncated(Normal(0., 1.), 0., Inf), d)
    # Likelihood.
    for i in 1:length(y)
        θ = log(α[x[i,1]]) - log(α[x[i,2]])
        y[i] ~ BernoulliLogit(θ)
    end
end
# Define the function to rank the teams.
"""
    rank

Ranking the ability scores, α, for each object in each posterior draw.

Args:
    x (Chains): The result of sampling the turing.jl model.
    d (Int8): The number of objects.

Returns:

"""
function rank(x, d)
    # Get the dims of the posterior.
    iters = size(x,1)
    chains = size(x,3)
    # Create a matrix of samples for the α parameter.
    samples = MCMCChains.group(x, :α).value
    # Initialize an array of rankings.
    rank_arr = Array{Integer, 3}(undef, iters, length(d), chains)
    # Rank the α for each sample.
    # This should produce an array with the ranking
    # for each team -- in order.
    for c in 1:chains
        for i in 1:iters
            # Get the current sample for iteration i and chain j
            current_sample = samples[i, :, c]
            # Rank the options by sorting the α values
            # in descending order (higher α means higher rank).
            ranked_indices = sortperm(current_sample, rev=true)
            # Assign ranks to each option based on sorted order
            for rank_idx in 1:length(d)
                rank_arr[i, ranked_indices[rank_idx], c] = rank_idx
            end
        end
    end
    # Initialize a DataFrame.
    df = DataFrame()
    # Place the rankings into a DataFrame.
    for c in 1:chains
        for (key, value) in d
            temp_df = DataFrame(
                iter = repeat(iters:-1:1, outer=1)
                , Rank = rank_arr[:, value, c]
                , Team = key
                , chain = c
            )
            # Append the temporary DataFrame
            # to the main DataFrame
            append!(df, temp_df)
        end
    end    
    # Return the result.
    return df
end

function fit_model(df, ids)
    # Fit the model.
    mod = turing(
        Matrix(select(df, [:home_team, :away_team]))
        , df.home_win
        , length(ids)
    );
    fit = Turing.sample(mod, NUTS(), MCMCThreads(), 4_000, 4);
    turing_ranks = rank(fit, ids);
    return turing_ranks
end

# Benchmark it.
fit_model(df, ids);
run(@benchmarkable fit_model(df, ids) samples=times evals=1)
