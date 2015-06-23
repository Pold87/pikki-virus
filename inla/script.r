
library(INLA)

print("Reading data ...")
train <- read.csv("data/unique_train.csv", header = T)
train.data <- data.frame(Species=train$Species, Latitude=train$Latitude, Longitude=train$Longitude, Week=train$Calendar_Week, WnvPresent=train$WnvPresent_DateTrapSpecies, Tmin=train$Tmin, Year=train$Year)

test <- read.csv("data/new_test_priors.csv", header = T)
test.data <- data.frame(Species=test$Species, Latitude=test$Latitude, Longitude=test$Longitude, Week=test$Calendar_Week, Tmin=test$Tmin, Year=test$Year)

train.data = head(train.data, 100)
test.data = head(test.data, 100)

print("Merging data ...")
test.data$WnvPresent <- NA

merged.data <- rbind(train.data, test.data)

n = nrow(merged.data)

mapping <- read.csv("graphs/voronoi_mapping.csv", header = T)

merged.data$TrapID <- NA

for (i in 1:n)
{
	# 	merged.data[i,]$TrapID <- mapping$Block[which.min(abs(mapping$Latitude - merged.data[i,]$Latitude) + abs(mapping$Longitude - merged.data[i,]$Longitude))]
	merged.data[i,]$TrapID <- which.min(abs(mapping$Latitude - merged.data[i,]$Latitude) + abs(mapping$Longitude - merged.data[i,]$Longitude))
}

print("Fitting model ...")
formula <- WnvPresent ~ 1  + f(TrapID, model = "besag", graph = "graphs/voronoi.graph") + f(Week, model = "rw1", cyclic = TRUE) + Tmin + f(Species) + Year
mod <- inla(formula, data = merged.data, control.predictor = list(link = 1))

options(scipen = 500) # make sure no scientific numbers are written
WnvPresent <- mod$summary.fitted.values$mean[(nrow(train.data) + 1):nrow(merged.data)]
Id <- 1:nrow(test.data)
backup <- WnvPresent
WnvPresent <- pmax(0, WnvPresent)
write.csv(cbind(Id, WnvPresent), file = "result.csv", row.names  = FALSE)
