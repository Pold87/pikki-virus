
library(INLA)

print("Reading data ...")

train <- read.csv("unique_train.csv", header = T)
train.data <- data.frame(Species=train$Species, Latitude=train$Latitude, Longitude=train$Longitude, Calendar_Week=train$Calendar_Week, WnvPresent_DateTrapSpecies=train$WnvPresent_DateTrapSpecies)

# Species
# # Block
# Latitude
# Longitude
# # # AddressAccuracy
# Calendar_Week
# # # Station
# # # Tmax
# # # Tmin
# # # Tavg
# # # Depart
# # # DewPoint
# # # WetBulb
# # # Heat
# # # Cool
# # # Sunrise
# # # Sunset
# # # Depth
# # # SnowFall
# # # PrecipTotal
# # # StnPressure
# # # SeaLevel
# # # ResultSpeed
# # # ResultDir
# # # AvgSpeed
# # Year
# # Month
# WnvPresent_DateTrapSpecies

# species <- aggregate(WnvPresent ~ Species, train, sum)

test <- read.csv("new_test_priors.csv", header = T)
test.data <- data.frame(Species=test$Species, Latitude=test$Latitude, Longitude=test$Longitude, Calendar_Week=test$Calendar_Week)

train.data = head(train.data, 100)
test.data = head(test.data, 100)

print("Merging data ...")
test.data$WnvPresent_DateTrapSpecies <- NA

joined.data <- rbind(train.data, test.data)
merged.data <- joined.data

n = nrow(merged.data)

mapping <- read.csv("voronoi_mapping.csv", header = T)

merged.data$TrapID <- NA
for (i in 1:n)
{
	merged.data[i,]$TrapID <- which.min(abs(mapping$Latitude - merged.data[i,]$Latitude) + abs(mapping$Longitude - merged.data[i,]$Longitude))
}

formula <- WnvPresent_DateTrapSpecies ~ f(Species) + f(Calendar_Week, model = "rw1", cyclic = TRUE) + f(TrapID, model = "besag", graph = "voronoi.graph")
print("Fitting model ...")
mod <- inla(formula, data = merged.data, control.predictor = list(link = 1), verbose = TRUE)

options(scipen = 500) # make sure no scientific numbers are written
WnvPresent <- mod$summary.fitted.values$mean[(nrow(train.data) + 1):nrow(merged.data)]
Id <- 1:nrow(test.data)
backup <- WnvPresent
WnvPresent <- pmax(0, WnvPresent)
write.csv(cbind(Id, WnvPresent), file = "result3.csv", row.names  = FALSE)
