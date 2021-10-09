#
# Generate Ising model phase transition examples
#
# USAGE: Rscript --vanilla ising-example.R size chains [nwarm nstats]
#
# EXAMPLE:        Rscript --vanilla ising-example.R 48 3 1e6 2e6
# Replot example: Rscript --vanilla ising-example.R 48 3
#

# I want the chains to go both forward and backward like in Mackay's book example
# hysteresis -> not well converged

betas <- 2^seq(from = -8, to = 5, len = 100)
betas <- c(betas, rev(betas))

args <- commandArgs(trailingOnly = TRUE)

stopifnot(length(args) == 2 || length(args) == 4)

size <- as.integer(args[1])
chains <- as.integer(args[2])  # num. parallel chains (OpenMP threads)

stopifnot(size >= 4 && chains >= 1)

if (length(args) == 4) {
  run_ising <- TRUE
  numwarm <- as.integer(args[3])
  numstat <- as.integer(args[4])
} else {
  run_ising <- FALSE
}

# Ising coupling parameter J, external field H
J <- +1
H <- 0

# Critical temp. if H = 0
Tc <- 2 * J / log(1 + sqrt(2))

# Metropolis or Gibbs?
sampler_str <- 'Metropolis'

if (run_ising) {
  cmdstr <- sprintf('ising %f %f %i %s %i %i %i', J, H, size, sampler_str, numwarm, numstat, chains)
  for (ii in 1:length(betas)) {
    cmdstr <- sprintf('%s %e', cmdstr, betas[ii])
  }
  print(cmdstr)
  system(cmdstr)
}

library(tidyverse)

D <- list()
for (ii in 1:chains) {
  tallyName <- sprintf('ising_tally_%04i_%04i.csv', size, ii - 1)
  print(tallyName)
  Q <- read_csv(tallyName)
  Q <- mutate(Q, logT = -1.0 * log(beta), 
                 EovrN = energy / (size * size),
                 chain = as.factor(ii),
                 VarE = energy_sq - energy * energy)
  D[[ii]] <- Q
}

DA <- D[[1]]
for (ii in 2:chains) {
  DA <- rbind(DA, D[[ii]])
}

dev.new()
gg <- ggplot(data = DA, mapping = aes(x = logT,  y = EovrN, colour = chain)) + 
        geom_point(size = 1, alpha = 0.5) + 
        geom_path()
if (H == 0) gg <- gg + geom_vline(xintercept = log(Tc))
gg <- gg + ggtitle(sprintf('2D Ising model (J = %.2f, H = %.2f)', J, H), 
                   subtitle = sprintf('grid size N = %i x %i, sampler = %s', size, size, sampler_str))
gg <- gg + xlab('log(T)') + ylab('<energy> / N')
plot(gg)

dev.new()
gg <- ggplot(data = DA, mapping = aes(x = logT,  y = magn_sq, colour = chain)) + 
        geom_point(size = 1, alpha = 0.5) + 
        geom_path()
if (H == 0) gg <- gg + geom_vline(xintercept = log(Tc))
gg <- gg + ggtitle(sprintf('2D Ising model (J = %.2f, H = %.2f)', J, H), 
                   subtitle = sprintf('grid size N = %i x %i, sampler = %s', size, size, sampler_str))
gg <- gg + xlab('log(T)') + ylab('<magnetization^2>')
plot(gg)

dev.new()
gg <- ggplot(data = DA, mapping = aes(x = logT,  y = VarE / (size * size), colour = chain)) + 
        geom_point(size = 1, alpha = 0.5) + 
        geom_path()
if (H == 0) gg <- gg + geom_vline(xintercept = log(Tc))
gg <- gg + ggtitle(sprintf('2D Ising model (J = %.2f, H = %.2f)', J, H), 
                   subtitle = sprintf('grid size N = %i x %i, sampler = %s', size, size, sampler_str))
gg <- gg + xlab('log(T)') + ylab('Var[energy] / N')
plot(gg)

print(Tc)
