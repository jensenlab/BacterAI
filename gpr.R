
eps <- sqrt(.Machine$double.eps)

train_new_GP <- function(X, y, d=0.1, g=0.1*var(y), dK=TRUE, max_theta=100, verb=2) {
  gpi <- laGP::newGPsep(X, y, d=d, g=g, dK=dK)
  ndim <- dim(X)[[2]]
  tmin <- rep(eps, ndim+1)
  tmax <- c(rep(max_theta, ndim), var(y))
  mle <- laGP::mleGPsep(gpi, para="both", tmin=tmin, tmax=tmax, verb=verb)
  return(gpi)
}

update_GP <- function(gpi, X, y, verb=0) {
  laGP::updateGPsep(X, y, verb=verb)
}

assert_matrix <- function(X) {
  if (!is.matrix(X)) {
    matrix(X, nrow=1, dimnames=list(NULL, names(X)))
  } else {
    X
  }
}

predict_GP <- function(gpi, X) {
  laGP::predGPsep(gpi, assert_matrix(X), lite=TRUE)$mean
}

sample_GP <- function(gpi, X, n=1) {
  yp <- laGP::predGPsep(gpi, assert_matrix(X), lite=TRUE)
  if (length(yp$mean) < 2) {
    mvtnorm::rmvnorm(n, yp$mean, sigma=matrix(yp$s2))
  } else {
    mvtnorm::rmvnorm(n, yp$mean, sigma=diag(yp$s2))
  }
}

delete_GP <- function(gpi) {
  laGP::deleteGPsep(gpi)
}


