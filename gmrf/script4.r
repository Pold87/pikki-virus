
library(INLA)

print("Reading data ...")

train <- read.csv("unique_train.csv", header = T)
train.data <- data.frame(WnvPresent_conditional_Species=train$WnvPresent_conditional_Species,
		WnvPresent_conditional_AddressAccuracy=train$WnvPresent_conditional_AddressAccuracy,
		WnvPresent_conditional_Calendar_Week=train$WnvPresent_conditional_Calendar_Week,
		WnvPresent_conditional_Station=train$WnvPresent_conditional_Station,
		# WnvPresent_conditional_Tmin=train$WnvPresent_conditional_Tmin,
		# WnvPresent_conditional_DewPoint=train$WnvPresent_conditional_DewPoint,
		# WnvPresent_conditional_WetBulb=train$WnvPresent_conditional_WetBulb,
		# WnvPresent_conditional_Cool=train$WnvPresent_conditional_Cool,
		# WnvPresent_conditional_SnowFall=train$WnvPresent_conditional_SnowFall,
		WnvPresent_conditional_ResultDir=train$WnvPresent_conditional_ResultDir,
		WnvPresent_conditional_Month=train$WnvPresent_conditional_Month,
		WnvPresent_DateTrapSpecies=train$WnvPresent_DateTrapSpecies)

# species <- aggregate(WnvPresent ~ Species, train, sum)

test <- read.csv("new_test_priors.csv", header = T)
test.data <- data.frame(WnvPresent_conditional_Species=test$WnvPresent_conditional_Species,
		WnvPresent_conditional_AddressAccuracy=test$WnvPresent_conditional_AddressAccuracy,
		WnvPresent_conditional_Calendar_Week=test$WnvPresent_conditional_Calendar_Week,
		WnvPresent_conditional_Station=test$WnvPresent_conditional_Station,
		# WnvPresent_conditional_Tmin=test$WnvPresent_conditional_Tmin,
		# WnvPresent_conditional_DewPoint=test$WnvPresent_conditional_DewPoint,
		# WnvPresent_conditional_WetBulb=test$WnvPresent_conditional_WetBulb,
		# WnvPresent_conditional_Cool=test$WnvPresent_conditional_Cool,
		# WnvPresent_conditional_Depth=test$WnvPresent_conditional_Depth,
		# WnvPresent_conditional_SnowFall=test$WnvPresent_conditional_SnowFall,
		WnvPresent_conditional_ResultDir=test$WnvPresent_conditional_ResultDir,
		WnvPresent_conditional_Month=test$WnvPresent_conditional_Month)

train.data = head(train.data, 100)
test.data = head(test.data, 100)

print("Merging data ...")
test.data$WnvPresent_DateTrapSpecies <- NA

joined.data <- rbind(train.data, test.data)
merged.data <- joined.data

formula <- WnvPresent_DateTrapSpecies ~ f(WnvPresent_conditional_Species) +
	f(WnvPresent_conditional_AddressAccuracy) +
	f(WnvPresent_conditional_Calendar_Week) +
	f(WnvPresent_conditional_Station) +
	# f(WnvPresent_conditional_Tmin) +
	# f(WnvPresent_conditional_DewPoint) +
	# f(WnvPresent_conditional_WetBulb) +
	# f(WnvPresent_conditional_Cool) +
	# f(WnvPresent_conditional_Depth) +
	# f(WnvPresent_conditional_SnowFall) +
	f(WnvPresent_conditional_ResultDir) +
	f(WnvPresent_conditional_Month)

print("Fitting model ...")
mod <- inla(formula, data = merged.data, control.predictor = list(link = 1), verbose = TRUE)

options(scipen = 500) # make sure no scientific numbers are written
WnvPresent <- mod$summary.fitted.values$mean[(nrow(train.data) + 1):nrow(merged.data)]
Id <- 1:nrow(test.data)
backup <- WnvPresent
WnvPresent <- pmax(0, WnvPresent)
write.csv(cbind(Id, WnvPresent), file = "result4.csv", row.names  = FALSE)
