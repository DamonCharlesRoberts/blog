data {
  int<lower=1> N; // Number of games.
  int<lower=1> J; // Number of teams.
  array[N, 2] int T; // Matrix of team ids.
  array[N] int<lower=0, upper=1> y; // Did the home team win?
}
transformed data {
  // Create a vector indicating each team.
  array[N] int home = to_array_1d(T[,1]);
  array[N] int away = to_array_1d(T[,2]);
}
parameters {
  vector<lower=0>[J] alpha; // The ability for each team.
}
model {
  // Prior on a logged-odds scale of the ability for each team.
  alpha ~ normal(0, 1);
  // Compute the ability for each team given who won.
  // The ability for the away team is dependent on whether they
  // beat the home team.
  // If the away team won, then the logged odd would be
  // alpha_away * 1 - alpha_home * 0
  y ~ bernoulli_logit(log(alpha[home]) - log(alpha[away]));
}
generated quantities {
  // Compute the ranking of each team based on who won.
  array[J] int rank; // Ranking of the teams.
  {
    // Get the ranking of each team in descending order
    // of the alpha.
    array[J] int rank_index = sort_indices_desc(alpha);
    // For each team, apply the rank
    // is the rank for team i.
    for (i in 1:J) {
      rank[rank_index[i]] = i;
    }
  }
}
