
library(INLA)

train <- read.csv("../train.csv", header = T)

# aggregate(WnvPresent ~ Species, train, sum)

# formula <- time + location # + weather + spraying

test <- read.csv("../test.csv", header = T)
# train = head(train, 10)
# test = head(test, 10)

# n.seas = 12

train$Date <- as.numeric(as.POSIXct(as.Date(train$Date, "%m/%d/%Y")))	# capital Y is important
test$Date <- as.numeric(as.POSIXct(as.Date(test$Date, "%Y-%m-%d")))	# capital Y is important
test$WnvPresent <- NA
test$NumMosquitos <- NA
train$Id <- (1:nrow(train)) + nrow(test)

# seasonal <- rep(1:n.seas, ceiling(n/n.seas))[1:n]
data <- rbind(train, test)
# data$Season <- seasonal #
# data$seasonal <- (data$Date) %% n.seas + 1

# formula <- WnvPresent ~ f(Season, model = "seasonal", season.length = n.seas)# + f(Species) + f(Trap)
formula <- WnvPresent ~ f(Date) + f(Species) + f(Trap)
mod <- inla(formula, data = data, control.predictor = list(link = 1))

# print(mod$summary.fitted.values$mean)
# 31557600000


options(scipen = 500)
WnvPresent <- mod$summary.fitted.values$mean[(1:nrow(test)) + nrow(train)]
Id <- 1:nrow(test)
backup <- WnvPresent
WnvPresent <- pmax(0, WnvPresent)
write.csv(cbind(Id, WnvPresent), file = "result.csv", row.names  = FALSE)

# Date,Address,Species,Block,Street,Trap,AddressNumberAndStreet,Latitude,Longitude,AddressAccuracy,NumMosquitos,WnvPresent
