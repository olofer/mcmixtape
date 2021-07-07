#
# Visual demonstration & test of basic CTMC program
#
# USAGE: Rscript --vanilla ctmc-example.R nsamples
#
# EXAMPLE: 
#   Rscript --vanilla ctmc-example.R 1e6
#

args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 1)
numSamples <- as.integer(args[1])
stopifnot(numSamples >= 10)

library(Matrix)  # sparseMatrix etc..
library(tidyverse) # data file input & plotting

getCollatzStepsAndMax <- function(j) {
  stopifnot(j >= 1)
  max.j <- j
  steps.j <- 0
  while (j != 1) {
    if ((j %% 2) == 0) {
      j <- j / 2
    } else {
      j <- 3 * j + 1
    }
    steps.j <- steps.j + 1
    if (j > max.j) {
      max.j <- j
    }
  }
  return(c(steps.j, max.j))
}

getCollatzTransitionMatrix <- function(N, K, qinit, qabsorb, qeven, qodd) {
  stopifnot(qinit > 0 && qabsorb > 0 && qeven > 0 && qodd > 0)
  r <- array(0, dim = c(N + 1))
  c <- array(0, dim = c(N + 1))
  x <- array(0, dim = c(N + 1))

  for (i in 1:N) {
    r[i] <- 1
    c[i] <- 1 + i
    x[i] <- qinit / N
  }

  r[N + 1] <- 1
  c[N + 1] <- 1
  x[N + 1] <- -1.0 * qinit

  M <- 3 * (K - 1)
  r1 <- array(0, dim = c(M))
  c1 <- array(0, dim = c(M))
  x1 <- array(0, dim = c(M))

  l <- 0
  for (j in 1:K) {
    if (j == 1) {
        # let it stay zero; absorbing state
      } else if ((j %% 2) == 0) {    # even 
        k <- j / 2

        l <- l + 1
        r1[l] <- 1 + j
        c1[l] <- 1 + k
        x1[l] <- qeven

        l <- l + 1
        r1[l] <- 1 + j
        c1[l] <- 1 + 1
        x1[l] <- qabsorb

        l <- l + 1
        r1[l] <- 1 + j
        c1[l] <- 1 + j
        x1[l] <- -1.0 * (qabsorb + qeven)

      } else {           # odd but not 1
        k <- 3 * j + 1

        l <- l + 1
        r1[l] <- 1 + j
        c1[l] <- 1 + k
        x1[l] <- qodd

        l <- l + 1
        r1[l] <- 1 + j
        c1[l] <- 1 + 1
        x1[l] <- qabsorb

        l <- l + 1
        r1[l] <- 1 + j
        c1[l] <- 1 + j
        x1[l] <- -1.0 * (qabsorb + qodd)
      }
  }
  stopifnot(M == l)

  r <- append(r, r1)
  c <- append(c, c1)
  x <- append(x, x1)

  Q <- sparseMatrix(i = r, j = c, x = x)
  return(Q) 
}

textOutputFile <- 'ctmc-example-samples.txt'
N <- 250
qinit <- 10.0
qabsorb <- 0.001 # 0.01
qeven <- 1.0
qodd <- 1.0

max.steps <- 0
max.state <- 0
for (j in 1:N) {
  C <- getCollatzStepsAndMax(j)
  max.steps <- max(c(max.steps, C[1]))
  max.state <- max(c(max.state, C[2]))
}

print(sprintf('max steps = %i, and max state = %i; given N = %i', max.steps, max.state, N))

# 1. --- Draw the CTMC samples ---
cmdstr <- sprintf('ctmc %i %s collatz %i %f %f %f %f', numSamples, textOutputFile, N, qinit, qabsorb, qeven, qodd)

print(cmdstr)
system(cmdstr)  # get return code; make sure it is zero..

F <- read_csv(textOutputFile, col_names = c('T'))
summary(F)

tmax <- max(F['T'])

# 2. --- Forward equation method ---

# max.states determines the length of the probability vector we need to evolve..
pvec <- array(data = 0, dim = c(1 + max.state))
pvec[1] <- 1.0  # start in state 0; which then scatters onto states 1..N uniformly at first jump

# forward ODE: dp/dt = Q' * p, p(0) = initial probs (uniform on 1..N, zeros elsewhere)
# output: element of p correspondning to state 1 (absorbing)

Q <- getCollatzTransitionMatrix(N, max.state, qinit, qabsorb, qeven, qodd)
rowQ <- dim(Q)[1]
colQ <- dim(Q)[2]

tvec <- seq(from = 0, to = tmax, by = 0.05)
deltat <- tvec[2] - tvec[1]

print(sprintf('solving forward equation (dt=%f) for %i steps..', deltat, length(tvec)))

prob <- array(NA, dim = c(length(tvec)))
dprb <- array(NA, dim = c(length(tvec)))
prob[1] <- pvec[2]
for (i in 2:length(tvec)) {
  p0 <- pvec
  z1 <- as.vector(t(Q) %*% as.matrix(p0))
  f1 <- z1[1:(1 + max.state)]
  w1 <- z1[(2 + max.state):colQ]
  stopifnot(norm(w1, type = '2') == 0)
  p1 <- p0 + deltat * f1
  z2 <- as.vector(t(Q) %*% as.matrix(p1))
  f2 <- z2[1:(1 + max.state)]
  w2 <- z2[(2 + max.state):colQ]
  stopifnot(norm(w2, type = '2') == 0)
  pvec <- p0 + 0.5 * deltat * (f1 + f2)
  prob[i] <- pvec[2]
  dprb[i - 1] <- 0.5 * (f1[2] + f2[2])
}

# Produce a plot & then exit
G <- tibble(x = tvec[1:(length(tvec) - 1)], y = dprb[1:(length(tvec) - 1)])

dev.new()
gg <- ggplot(data = F, aes(x = T)) + 
        geom_histogram(bins = 400, 
                       aes(y = ..density..), 
                       fill = 'midnightblue', 
                       alpha = 0.75)
gg <- gg + geom_density(color = 'blue', lwd = 2.0)
gg <- gg + geom_line(data = G, aes(x, y), color = 'red', lwd = 1.0)
gg <- gg + labs(title = sprintf('blue = sample (samples = %i); red = forward eq.', numSamples), 
                x = 'time to absorption T', 
                y = 'count / density')
plot(gg)
