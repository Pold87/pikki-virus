
library(INLA)

weather.station.1.lat <- 41.995
weather.station.1.lon <- -87.933
weather.station.1.elev <- 662

weather.station.2.lat <- 41.786
weather.station.2.lon <-  -87.752
weather.station.2.elev <- 612

print("Reading data ...")
# weather <- read.csv("weather.csv", header = T)
# weather.data <- weather[c(TRUE, FALSE), ]
# weather.data <- data.frame(Date=weather.data$Date, Heat=weather.data$Heat, Cool=weather.data$Cool)

# Heat Cool

train <- read.csv("unique_train.csv", header = T)
# train <- read.csv("train.csv", header = T)
# train.data <- data.frame(Date=train$Date, Species=train$Species, Trap=train$Trap, WnvPresent=train$WnvPresent)
train.data <- data.frame(Species=train$Species, Latitude=train$Latitude,
		Longitude=train$Longitude, #Calendar_Week=train$Calendar_Week,
		Station=train$Station, Year=train$Year, Month=train$Month,
		WnvPresent_conditional_Species=train$WnvPresent_conditional_Species,
		WnvPresent_conditional_AddressAccuracy=train$WnvPresent_conditional_AddressAccuracy,
		WnvPresent_conditional_Calendar_Week=train$WnvPresent_conditional_Calendar_Week,
		WnvPresent_conditional_Station=train$WnvPresent_conditional_Station,
		# WnvPresent_conditional_Tmin=train$WnvPresent_conditional_Tmin,
		# WnvPresent_conditional_DewPoint=train$WnvPresent_conditional_DewPoint,
		# WnvPresent_conditional_WetBulb=train$WnvPresent_conditional_WetBulb,
		# WnvPresent_conditional_Cool=train$WnvPresent_conditional_Cool,
		# WnvPresent_conditional_Depth=train$WnvPresent_conditional_Depth,
		# WnvPresent_conditional_SnowFall=train$WnvPresent_conditional_SnowFall,
		WnvPresent_conditional_ResultDir=train$WnvPresent_conditional_ResultDir,
		WnvPresent_conditional_Month=train$WnvPresent_conditional_Month,
		WnvPresent_DateTrapSpecies=train$WnvPresent_DateTrapSpecies)

# species <- aggregate(WnvPresent ~ Species, train, sum)
# formula <- time + location # + weather + spraying

test <- read.csv("new_test_priors.csv", header = T)
# test <- read.csv("test.csv", header = T)
# test.data <- data.frame(Date=test$Date, Species=test$Species, Trap=test$Trap)
test.data <- data.frame(Species=test$Species, Latitude=test$Latitude,
		Longitude=test$Longitude, #Calendar_Week=test$Calendar_Week,
		Station=test$Station, Year=test$Year, Month=test$Month,
		WnvPresent_conditional_Species=test$WnvPresent_conditional_Species,
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
# Species Block Latitude Longitude AddressAccuracy Calendar_Week Station
# Tmax Tmin Tavg Depart DewPoint WetBulb Heat Cool Sunrise Sunset Depth
# SnowFall PrecipTotal StnPressure SeaLevel ResultSpeed ResultDir AvgSpeed
# Year Month Tavg_week precip_week  heat_dw cool_dw X1_week_avrgPrecipTotal
# X1_week_avrgTavg X2_week_avrgPrecipTotal X2_week_avrgTavg
# X3_week_avrgPrecipTotal X3_week_avrgTavg X4_week_avrgPrecipTotal
# X4_week_avrgTavg X5_week_avrgPrecipTotal X5_week_avrgTavg
# X6_week_avrgPrecipTotal X6_week_avrgTavg X7_week_avrgPrecipTotal
# X7_week_avrgTavg X8_week_avrgPrecipTotal X8_week_avrgTavg
# X9_week_avrgPrecipTotal X9_week_avrgTavg X10_week_avrgPrecipTotal
# X10_week_avrgTavg X11_week_avrgPrecipTotal X11_week_avrgTavg
# X12_week_avrgPrecipTotal X12_week_avrgTavg X13_week_avrgPrecipTotal
# X13_week_avrgTavg X14_week_avrgPrecipTotal X14_week_avrgTavg
# X15_week_avrgPrecipTotal X15_week_avrgTavg X16_week_avrgPrecipTotal
# X16_week_avrgTavg X17_week_avrgPrecipTotal X17_week_avrgTavg
# X18_week_avrgPrecipTotal X18_week_avrgTavg X19_week_avrgPrecipTotal
# X19_week_avrgTavg X20_week_avrgPrecipTotal X20_week_avrgTavg
# X21_week_avrgPrecipTotal X21_week_avrgTavg X22_week_avrgPrecipTotal
# X22_week_avrgTavg X23_week_avrgPrecipTotal X23_week_avrgTavg count_mosquitos
# WnvPresent_conditional_Species WnvPresent_conditional_AddressAccuracy
# WnvPresent_conditional_Calendar_Week WnvPresent_conditional_Station
# WnvPresent_conditional_Tmin WnvPresent_conditional_DewPoint
# WnvPresent_conditional_WetBulb WnvPresent_conditional_Cool
# WnvPresent_conditional_Depth WnvPresent_conditional_SnowFall
# WnvPresent_conditional_ResultDir WnvPresent_conditional_Month

# weather.data = head(weather.data, 100)
train.data = head(train.data, 100)
test.data = head(test.data, 100)

print("Merging data ...")

# weather.data$Date <- as.numeric(as.POSIXct(as.Date(weather.data$Date, "%Y-%m-%d")))	# capital Y is important
# train.data$Date <- as.numeric(as.POSIXct(as.Date(train.data$Date, "%Y-%m-%d")))	# capital Y is important
# test.data$Date <- as.numeric(as.POSIXct(as.Date(test.data$Date, "%Y-%m-%d")))	# capital Y is important
test.data$WnvPresent_DateTrapSpecies <- NA
# test.data$WnvPresent <- 0
# test.data$WnvPresent[which(is.element(test.data$Species, species$Species[which(species$WnvPresent != 0)]))] <- NA
# test.data$NumMosquitos <- NA
# train.data$Id <- (1:nrow(train.data)) + nrow(test.data)

joined.data <- rbind(train.data, test.data)
merged.data <- joined.data
# merged.data$Trap <- as.numeric(substr(merged.data$Trap, 2, 4))# + (substr(merged.data$Trap, 5, 5) != "") * 300
# traps.id <- as.numeric(substr(unique(graph.data$Trap), 2, 4))
# merged.data$Trap <- match(merged.data$Trap, traps.id)

# merged.data <- merge(joined.data, weather.data)

n = nrow(merged.data)
# seasonal <- rep(1:n.seas, ceiling(n / n.seas))[1:n]
# merged.data$Season <- seasonal #
# len.cycle = 31557600
# merged.data$DateInYear <- ((merged.data$Date) %% len.cycle + 1)
# merged.data$Season[1:(n.seas-1)] <- 1:(n.seas-1)

# formula <- WnvPresent ~ f(Date) + f(Species) + f(Trap)
# formula <- WnvPresent ~ f(Season, model = "seasonal", season.length = n.seas)# + f(Species) + f(Trap)
# formula <- WnvPresent ~ f(Date) + f(Species) + f(Trap, model = "besag", graph = "traps.graph")
# formula <- WnvPresent ~ f(Date) + f(Species) + f(Trap) + f(Tavg_week) + f(precip_week) + f(heat_dw) + f(cool_dw)
# formula <- WnvPresent ~ f(Species) + f(Date) + f(DateInYear, model = "rw2", cyclic = TRUE) + f(Trap, model = "besag", graph = "traps.graph")
formula <- WnvPresent_DateTrapSpecies ~ f(Species) + f(Latitude) + f(Longitude) + f(Station) + f(Month) + f(WnvPresent_conditional_Species) +
		f(WnvPresent_conditional_Calendar_Week) + f(WnvPresent_conditional_ResultDir) + f(WnvPresent_conditional_Month)


print("Fitting model ...")
# # formula <- WnvPresent ~ f(Season, model = "seasonal", season.length = n.seas)# + f(Species) + f(Trap)
mod <- inla(formula, data = merged.data, control.predictor = list(link = 1), verbose = TRUE)

# # print(mod$summary.fitted.values$mean)
# # 31557600

options(scipen = 500) # make sure no scientific numbers are written
WnvPresent <- mod$summary.fitted.values$mean[(nrow(train.data) + 1):nrow(merged.data)]
Id <- 1:nrow(test.data)
backup <- WnvPresent
WnvPresent <- pmax(0, WnvPresent)
write.csv(cbind(Id, WnvPresent), file = "result.csv", row.names  = FALSE)

# # Date,Address,Species,Block,Street,Trap,AddressNumberAndStreet,Latitude,Longitude,AddressAccuracy,NumMosquitos,WnvPresent



## Test
# Id	Species	Block	Latitude	Longitude	AddressAccuracy	Calendar_Week	Station
# Tmax	Tmin	Tavg	Depart	DewPoint	WetBulb	Heat	Cool	Sunrise	Sunset	Depth	SnowFall	PrecipTotal	StnPressure	SeaLevel	ResultSpeed	ResultDir	AvgSpeed	Year	Month	Tavg_week	precip_week	heat_dw	cool_dw
# 1_week_avrgPrecipTotal	1_week_avrgTavg	2_week_avrgPrecipTotal	2_week_avrgTavg	3_week_avrgPrecipTotal	3_week_avrgTavg	4_week_avrgPrecipTotal	4_week_avrgTavg	5_week_avrgPrecipTotal	5_week_avrgTavg	6_week_avrgPrecipTotal	6_week_avrgTavg	7_week_avrgPrecipTotal	7_week_avrgTavg	8_week_avrgPrecipTotal	8_week_avrgTavg	9_week_avrgPrecipTotal	9_week_avrgTavg	10_week_avrgPrecipTotal	10_week_avrgTavg	11_week_avrgPrecipTotal	11_week_avrgTavg	12_week_avrgPrecipTotal	12_week_avrgTavg	13_week_avrgPrecipTotal	13_week_avrgTavg	14_week_avrgPrecipTotal	14_week_avrgTavg	15_week_avrgPrecipTotal	15_week_avrgTavg	16_week_avrgPrecipTotal	16_week_avrgTavg	17_week_avrgPrecipTotal	17_week_avrgTavg	18_week_avrgPrecipTotal	18_week_avrgTavg	19_week_avrgPrecipTotal	19_week_avrgTavg	20_week_avrgPrecipTotal	20_week_avrgTavg	21_week_avrgPrecipTotal	21_week_avrgTavg	22_week_avrgPrecipTotal	22_week_avrgTavg	23_week_avrgPrecipTotal	23_week_avrgTavg
# count_mosquitos
# WnvPresent_conditional_Species	WnvPresent_conditional_AddressAccuracy	WnvPresent_conditional_Calendar_Week	WnvPresent_conditional_Station	WnvPresent_conditional_Tmin	WnvPresent_conditional_DewPoint	WnvPresent_conditional_WetBulb	WnvPresent_conditional_Cool	WnvPresent_conditional_Depth	WnvPresent_conditional_SnowFall	WnvPresent_conditional_ResultDir	WnvPresent_conditional_Month

## Train
# Id	Species	Block	Latitude	Longitude	AddressAccuracy	Calendar_Week	Station
# Tmax	Tmin	Tavg	Depart	DewPoint	WetBulb	Heat	Cool	Sunrise	Sunset	Depth	SnowFall	PrecipTotal	StnPressure	SeaLevel	ResultSpeed	ResultDir	AvgSpeed	Year	Month	Tavg_week	precip_week	heat_dw	cool_dw
# 1_week_avrgPrecipTotal	1_week_avrgTavg	2_week_avrgPrecipTotal	2_week_avrgTavg	3_week_avrgPrecipTotal	3_week_avrgTavg	4_week_avrgPrecipTotal	4_week_avrgTavg	5_week_avrgPrecipTotal	5_week_avrgTavg	6_week_avrgPrecipTotal	6_week_avrgTavg	7_week_avrgPrecipTotal	7_week_avrgTavg	8_week_avrgPrecipTotal	8_week_avrgTavg	9_week_avrgPrecipTotal	9_week_avrgTavg	10_week_avrgPrecipTotal	10_week_avrgTavg	11_week_avrgPrecipTotal	11_week_avrgTavg	12_week_avrgPrecipTotal	12_week_avrgTavg	13_week_avrgPrecipTotal	13_week_avrgTavg	14_week_avrgPrecipTotal	14_week_avrgTavg	15_week_avrgPrecipTotal	15_week_avrgTavg	16_week_avrgPrecipTotal	16_week_avrgTavg	17_week_avrgPrecipTotal	17_week_avrgTavg	18_week_avrgPrecipTotal	18_week_avrgTavg	19_week_avrgPrecipTotal	19_week_avrgTavg	20_week_avrgPrecipTotal	20_week_avrgTavg	21_week_avrgPrecipTotal	21_week_avrgTavg	22_week_avrgPrecipTotal	22_week_avrgTavg	23_week_avrgPrecipTotal	23_week_avrgTavg
# count_mosquitos
# WnvPresent_conditional_Species	WnvPresent_conditional_AddressAccuracy	WnvPresent_conditional_Calendar_Week	WnvPresent_conditional_Station	WnvPresent_conditional_Tmin	WnvPresent_conditional_DewPoint	WnvPresent_conditional_WetBulb	WnvPresent_conditional_Cool	WnvPresent_conditional_Depth	WnvPresent_conditional_SnowFall	WnvPresent_conditional_ResultDir	WnvPresent_conditional_Month
# WnvPresent_DateTrapSpecies
