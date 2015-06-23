
train.data <- read.csv("data/unique_train.csv", header = T)
spray.data <- read.csv("data/spray.csv", header = T)

# set useless species to 0
train.data[train.data$Species == 0 | train.data$Species == 4 | train.data$Species == 5 | train.data$Species == 6, ] <- NA

train.data$NumMosquitos <- train.data$count_mosquitos
train.data$WnvPresent <- train.data$WnvPresent_DateTrapSpecies
train.data$Week <- train.data$Calendar_Week
train.data$Precip <- train.data$PrecipTotal

train.data.perWeek <- aggregate(NumMosquitos ~ Week + Year, train.data, sum)
train.data.perWeek.tmp <- aggregate(WnvPresent ~ Week + Year, train.data, sum)
train.data.perWeek <- merge(train.data.perWeek, train.data.perWeek.tmp)
train.data.perWeek.tmp <- aggregate(Tavg ~ Week + Year, train.data, mean)
train.data.perWeek <- merge(train.data.perWeek, train.data.perWeek.tmp)
train.data.perWeek.tmp <- aggregate(Precip ~ Week + Year, train.data, mean)
train.data.perWeek <- merge(train.data.perWeek, train.data.perWeek.tmp)

# ranges
minWeek <- min(train.data.perWeek$Week)
maxWeek <- max(train.data.perWeek$Week)
maxNM <- max(train.data.perWeek$NumMosquitos)
maxWP <- max(train.data.perWeek$WnvPresent)
minT <- min(train.data.perWeek$Tavg)
maxT <- max(train.data.perWeek$Tavg)
minP <- min(train.data.perWeek$Precip)
maxP <- max(train.data.perWeek$Precip)
maxP <- maxP * 1.2


yearToPlot <- 2013
train.data.perWeek.year <- train.data.perWeek[train.data.perWeek$Year == yearToPlot, ]

spray.data.unique <- unique(spray.data$Date)
spray.data.years <- strftime(as.POSIXct(as.Date(spray.data.unique, "%Y-%m-%d")), format = "%Y")
spray.data.weeks <- strftime(as.POSIXct(as.Date(spray.data.unique, "%Y-%m-%d")), format = "%W")
spray.data.tmp <- data.frame(Year = spray.data.years, Week = spray.data.weeks)

spray.data.year <- spray.data.weeks[spray.data.years == yearToPlot]

par(mar = c(5, 7, 4, 7)+.1, c(minWeek, maxWeek), cex = 1.5)
barplot(train.data.perWeek.year$Precip, axes = FALSE, col = "cyan",
	xlim = c(+.5, maxWeek - minWeek + .5), xlab = "", ylab = "", ylim = c(minP, maxP), main = "", space = 0)
axis(2, ylim = c(minP, maxP), col = "black", lwd = 2)
mtext(2, text = "Precipitation", line = 2, cex = 1.5)

par(new = TRUE)
plot(train.data.perWeek.year$Week, train.data.perWeek.year$Tavg, axes = FALSE,
	'l', col = "red", lwd = 5, xlab = "", ylab = "", xlim = c(minWeek, maxWeek), ylim = c(minT, maxT), main = "")
axis(2, ylim = c(minT, maxT), lwd = 2, line = 3.5)
mtext(2, text = "T_avg", line = 5.5, cex = 1.5)

par(new = TRUE)
plot(train.data.perWeek.year$Week, train.data.perWeek.year$NumMosquitos, axes = FALSE,
	'l', col = "black", lwd = 5, xlab = "", ylab = "", xlim = c(minWeek, maxWeek), ylim = c(0, maxNM), main = "")
axis(4, ylim = c(0, maxNM), lwd = 2)
mtext(4, text = "NumMosquitos", line = 2, cex = 1.5)

par(new = TRUE)
plot(train.data.perWeek.year$Week, train.data.perWeek.year$WnvPresent, axes = FALSE,
	'l', col = "purple", lwd = 5, xlab = "", ylab = "", xlim = c(minWeek, maxWeek), ylim = c(0, maxWP))
axis(4, ylim = c(0, maxWP), lwd = 2, line = 3.5)
mtext(4, text = "WnvPresent", line = 5.5, cex = 1.5)

axis(1, unique(train.data.perWeek$Week))
mtext("Week", side = 1 , col = "black", line = 2, cex = 1.5)

# spray data
# abline(v = spray.data.year, col = "red", lwd = 2)
