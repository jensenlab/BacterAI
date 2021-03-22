make_gpr = function(X, y) {
  eps <- sqrt(.Machine$double.eps)
  gpi <- laGP::newGPsep(X, y, d=0.1, g=0.1*stats::var(y), dK=TRUE)
  ndim <- dim(X)[[2]]
  tmin <- rep(eps, ndim+1)
  tmax <- c(rep(100, ndim), stats::var(y))
  mle <- laGP::mleGPsep(gpi, para="both", tmin=tmin, tmax=tmax, verb=2)
  return(gpi)
}

gpr_pred = function(model, Xtest) {
  yp <- laGP::predGPsep(model, Xtest)
  # print(yp)
  return(yp$mean)
}