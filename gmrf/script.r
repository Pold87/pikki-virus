
library(INLA)

weather.station.1.lat <- 41.995
weather.station.1.lon <- -87.933
weather.station.1.elev <- 662

weather.station.2.lat <- 41.786
weather.station.2.lon <-  -87.752
weather.station.2.elev <- 612

print("Reading data ...")
weather <- read.csv("weather.csv", header = T)
weather.data <- weather[c(TRUE, FALSE), ]
weather.data <- data.frame(Date=weather.data$Date, Heat=weather.data$Heat, Cool=weather.data$Cool)

# Heat Cool

train <- read.csv("train.csv", header = T)
train.data <- data.frame(Date=train$Date, Species=train$Species, Trap=train$Trap, WnvPresent=train$WnvPresent)

# aggregate(WnvPresent ~ Species, train, sum)
# formula <- time + location # + weather + spraying

test <- read.csv("test.csv", header = T)
test.data <- data.frame(Date=test$Date, Species=test$Species, Trap=test$Trap)

# weather.data = head(weather.data, 100)
# train.data = head(train.data, 100)
# test.data = head(test.data, 100)

print("Merging data ...")
# n.seas = 12

weather.data$Date <- as.numeric(as.POSIXct(as.Date(weather.data$Date, "%Y-%m-%d")))	# capital Y is important
train.data$Date <- as.numeric(as.POSIXct(as.Date(train.data$Date, "%Y-%m-%d")))	# capital Y is important
test.data$Date <- as.numeric(as.POSIXct(as.Date(test.data$Date, "%Y-%m-%d")))	# capital Y is important
test.data$WnvPresent <- NA
# test.data$NumMosquitos <- NA
# train.data$Id <- (1:nrow(train.data)) + nrow(test.data)

joined.data <- rbind(train.data, test.data)
merged.data <- merge(joined.data, weather.data)

# seasonal <- rep(1:n.seas, ceiling(n/n.seas))[1:n]
# data$Season <- seasonal #
# data$seasonal <- (data$Date) %% n.seas + 1

print("Fitting model ...")
# formula <- WnvPresent ~ f(Season, model = "seasonal", season.length = n.seas)# + f(Species) + f(Trap)
formula <- WnvPresent ~ f(Date) + f(Species) + f(Trap) + f(Heat) + f(Cool)
mod <- inla(formula, data = merged.data, control.predictor = list(link = 1), verbose = TRUE)

# print(mod$summary.fitted.values$mean)
# 31557600000


options(scipen = 500) # make sure no scientific numbers are written
WnvPresent <- mod$summary.fitted.values$mean[(nrow(train.data) + 1):nrow(merged.data)]
Id <- 1:nrow(test)
backup <- WnvPresent
WnvPresent <- pmax(0, WnvPresent)
write.csv(cbind(Id, WnvPresent), file = "result.csv", row.names  = FALSE)

# Date,Address,Species,Block,Street,Trap,AddressNumberAndStreet,Latitude,Longitude,AddressAccuracy,NumMosquitos,WnvPresent
