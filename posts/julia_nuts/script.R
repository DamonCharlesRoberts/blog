# Script for cmdstanr benchmarks.
library(DBI)
library(dplyr)
library(duckdb)
library(cmdstanr)
library(microbenchmark)

# Define the number of times to evaluate the functions.
evals <- 100

# Get the data.
get_data <- function(){
    # Connect to the DB.
    con <- dbConnect(duckdb(), "~/Desktop/mlb_pred/data/twenty_five.db")
    # Get the dataframe.
    df <- dbGetQuery(
        con
        , "
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
        where teams.season_id like '2025';
        "
    )
    # Close the connection.
    dbDisconnect(con, shutdown=TRUE)
    return(df)
}

# Benchmark the data retreival.
microbenchmark(get_data(), times=evals)

# Get the dataframe object.
df <- get_data()

# Get a vector of key-value pairs to map team abbreviations to team ids.
get_ids <- function(df){
    ids <- vector()
    for (i in seq_len(nrow(df))) {
        key <- df[i, "team_abbr"]
        value <- df[i, "home_team"]
        ids[as.character(key)] <- as.integer(value)
    }
    return(ids)
}

# Benchmark the creation of the vector.
microbenchmark(get_ids(df), times=evals)

# Get the ids vector.
ids <- get_ids(df)

# The model.

fit_model <- function(df, ids){
    # Fit the stan model
    mod <- cmdstan_model("./posts/julia_nuts/bt.stan")
    dat <- list(
        N = nrow(df)
        , J = length(ids)
        , T = as.matrix(df[, c("home_team", "away_team")])
        , y = df$home_win
    )
    fit <- mod$sample(
        data = dat
        , seed = 123
        , chains = 4
        , iter_warmup = 1000
        , iter_sampling = 3000
        , show_messages = FALSE
    )
    return(fit)
}

# Benchmark it.
microbenchmark(fit_model(df, ids), times=evals)
