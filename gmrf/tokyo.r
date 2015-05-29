
library(INLA)

## Load the data
data(Tokyo)
summary(Tokyo)

Tokyo$y[300:366] <- NA

## Define the model
formula = y ~ f(time, model="rw2", cyclic=TRUE, param=c(1,0.0001)) - 1

## The call to inla
result = inla(formula, family="binomial", Ntrials=n, data=Tokyo,
  control.predictor=list(compute=T))

## need to recompute the fitted values for those with data[i] = NA,
## as the identity link is used.
n = 366
fitted.values.mean = numeric(n)
for(i in 1:366) {
    if (is.na(Tokyo$y[i])) {
        if (FALSE) {
            ## either like this, which is slower...
            marg = inla.marginal.transform(
                            function(x) exp(x)/(1+exp(x)),
                            result$marginals.fitted.values[[i]] )
            fitted.values.mean[i] = inla.emarginal(
                            function(x) x, marg)
        } else {
            ## or like this,  which is much faster...
            fitted.values.mean[i] = inla.emarginal(
                            function(x) exp(x)/(1 +exp(x)),
                            result$marginals.fitted.values[[i]])
        }
    } else {
        fitted.values.mean[i] = result$summary.fitted.values[i,"mean"]
    }
}

plot(fitted.values.mean)
