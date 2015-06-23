

# train.data <- read.csv("train.csv", header = T)
# spray.data <- read.csv("spray.csv", header = T)

# train.data$Date <- as.numeric(as.POSIXct(as.Date(train.data$Date, "%Y-%m-%d")))	# capital Y is important
# spray.data$Date <- as.numeric(as.POSIXct(as.Date(spray.data$Date, "%Y-%m-%d")))	# capital Y is important

# train.data.perDate1 <- aggregate(NumMosquitos ~ Date, train.data, sum)
# train.data.perDate2 <- aggregate(WnvPresent ~ Date, train.data, sum)

# train.data.perDate <- merge(train.data.perDate1, train.data.perDate2)

# train.data.perDate.2007 <- train.data.perDate[train.data.perDate$Date < as.numeric(as.POSIXct(as.Date("2008-1-1", "%Y-%m-%d"))),]
# train.data.perDate.2009 <- train.data.perDate[train.data.perDate$Date > as.numeric(as.POSIXct(as.Date("2008-1-1", "%Y-%m-%d"))) &
# 					train.data.perDate$Date < as.numeric(as.POSIXct(as.Date("2010-1-1", "%Y-%m-%d"))),]
# train.data.perDate.2011 <- train.data.perDate[train.data.perDate$Date > as.numeric(as.POSIXct(as.Date("2010-1-1", "%Y-%m-%d"))) &
# 					train.data.perDate$Date < as.numeric(as.POSIXct(as.Date("2012-1-1", "%Y-%m-%d"))),]
# train.data.perDate.2013 <- train.data.perDate[train.data.perDate$Date > as.numeric(as.POSIXct(as.Date("2012-1-1", "%Y-%m-%d"))) &
# 					train.data.perDate$Date < as.numeric(as.POSIXct(as.Date("2014-1-1", "%Y-%m-%d"))),]

# spray.data.unique <- unique(spray.data$Date)
# spray.data.2011 <- spray.data.unique[spray.data.unique > as.numeric(as.POSIXct(as.Date("2010-1-1", "%Y-%m-%d"))) &
# 					spray.data.unique < as.numeric(as.POSIXct(as.Date("2012-1-1", "%Y-%m-%d")))]
# spray.data.2013 <- spray.data.unique[spray.data.unique > as.numeric(as.POSIXct(as.Date("2012-1-1", "%Y-%m-%d"))) &
# 					spray.data.unique < as.numeric(as.POSIXct(as.Date("2014-1-1", "%Y-%m-%d")))]

# plot(train.data.perDate$Date, train.data.perDate$NumMosquitos, 'l')
# plot(train.data.perDate.2013$Date, train.data.perDate.2013$NumMosquitos, 'l', col="black")
# par(new=TRUE)
# plot(train.data.perDate.2013$Date, train.data.perDate.2013$WnvPresent, 'l', col="green", xaxt="n", yaxt="n", xlab="", ylab="")
# axis(4)
# # mtext("NumMosquitos",side=2,line=3)
# mtext("WnvPresent",side=4,line=3)

# abline(v=spray.data.2013, col="red")

# order(sapply(train.data.perDate$Year_Week, `[`, 2))

# train.orig.data <- read.csv("train.csv", header = T)
train.data <- read.csv("unique_train.csv", header = T)
spray.data <- read.csv("spray.csv", header = T)

# species <- aggregate(WnvPresent ~ Species, train, sum)
# train.data[train.orig.data$Species == 'CULEX TARSALIS' | train.orig.data$Species == 'CULEX ERRATICUS' |
	# train.orig.data$Species == 'CULEX TERRITANS' | train.orig.data$Species == 'CULEX SALINARIUS', ] <- NA
train.data[train.data$Species == 0 | train.data$Species == 4 | train.data$Species == 5 | train.data$Species == 6, ] <- NA

train.data$NumMosquitos <- train.data$count_mosquitos
train.data$WnvPresent <- train.data$WnvPresent_DateTrapSpecies
train.data$Week <- train.data$Calendar_Week
train.data$Precip <- train.data$PrecipTotal

train.data.perWeek <- aggregate(NumMosquitos ~ Week + Species + Year, train.data, sum)
train.data.perWeek.tmp <- aggregate(WnvPresent ~ Week + Species + Year, train.data, sum)
train.data.perWeek <- merge(train.data.perWeek, train.data.perWeek.tmp)
train.data.perWeek.tmp <- aggregate(Tavg ~ Week + Species + Year, train.data, mean)
train.data.perWeek <- merge(train.data.perWeek, train.data.perWeek.tmp)
train.data.perWeek.tmp <- aggregate(Precip ~ Week + Species + Year, train.data, mean)
train.data.perWeek <- merge(train.data.perWeek, train.data.perWeek.tmp)

minWeek <- min(train.data.perWeek$Week)
maxWeek <- max(train.data.perWeek$Week)
maxNM <- max(train.data.perWeek$NumMosquitos)

train.data.total.tmp <- aggregate(WnvPresent ~ Week + Year, train.data, sum)
maxWP <- max(train.data.total.tmp$WnvPresent)
train.data.total.tmp <- aggregate(Tavg ~ Week + Year, train.data, mean)
minT <- min(train.data.total.tmp$Tavg)
maxT <- max(train.data.total.tmp$Tavg)
train.data.total.tmp <- aggregate(Precip ~ Week + Year, train.data, mean)
minP <- min(train.data.total.tmp$Precip)
maxP <- max(train.data.total.tmp$Precip)

train.data.perWeek.2007 <- train.data.perWeek[train.data.perWeek$Year == 2007, ]
train.data.perWeek.2009 <- train.data.perWeek[train.data.perWeek$Year == 2009, ]
train.data.perWeek.2011 <- train.data.perWeek[train.data.perWeek$Year == 2011, ]
train.data.perWeek.2013 <- train.data.perWeek[train.data.perWeek$Year == 2013, ]

train.data.perWeek.2007.pipiens <- train.data.perWeek.2007[train.data.perWeek.2007$Species == 1, ]
train.data.perWeek.2007.pipiensrestuans <- train.data.perWeek.2007[train.data.perWeek.2007$Species == 2, ]
train.data.perWeek.2007.restuans <- train.data.perWeek.2007[train.data.perWeek.2007$Species == 3, ]
train.data.perWeek.2007.tmp0 <- aggregate(WnvPresent ~ Week, train.data.perWeek.2007, sum)
train.data.perWeek.2007.tmp1 <- aggregate(Tavg ~ Week, train.data.perWeek.2007, mean)
train.data.perWeek.2007.tmp2 <- aggregate(Precip ~ Week, train.data.perWeek.2007, mean)
train.data.perWeek.2007 <- merge(train.data.perWeek.2007.tmp0, train.data.perWeek.2007.tmp1)
train.data.perWeek.2007 <- merge(train.data.perWeek.2007, train.data.perWeek.2007.tmp2)

train.data.perWeek.2009.pipiens <- train.data.perWeek.2009[train.data.perWeek.2009$Species == 1, ]
train.data.perWeek.2009.pipiensrestuans <- train.data.perWeek.2009[train.data.perWeek.2009$Species == 2, ]
train.data.perWeek.2009.restuans <- train.data.perWeek.2009[train.data.perWeek.2009$Species == 3, ]
train.data.perWeek.2009.tmp0 <- aggregate(WnvPresent ~ Week, train.data.perWeek.2009, sum)
train.data.perWeek.2009.tmp1 <- aggregate(Tavg ~ Week, train.data.perWeek.2009, mean)
train.data.perWeek.2009.tmp2 <- aggregate(Precip ~ Week, train.data.perWeek.2009, mean)
train.data.perWeek.2009 <- merge(train.data.perWeek.2009.tmp0, train.data.perWeek.2009.tmp1)
train.data.perWeek.2009 <- merge(train.data.perWeek.2009, train.data.perWeek.2009.tmp2)

train.data.perWeek.2011.pipiens <- train.data.perWeek.2011[train.data.perWeek.2011$Species == 1, ]
train.data.perWeek.2011.pipiensrestuans <- train.data.perWeek.2011[train.data.perWeek.2011$Species == 2, ]
train.data.perWeek.2011.restuans <- train.data.perWeek.2011[train.data.perWeek.2011$Species == 3, ]
train.data.perWeek.2011.tmp0 <- aggregate(WnvPresent ~ Week, train.data.perWeek.2011, sum)
train.data.perWeek.2011.tmp1 <- aggregate(Tavg ~ Week, train.data.perWeek.2011, mean)
train.data.perWeek.2011.tmp2 <- aggregate(Precip ~ Week, train.data.perWeek.2011, mean)
train.data.perWeek.2011 <- merge(train.data.perWeek.2011.tmp0, train.data.perWeek.2011.tmp1)
train.data.perWeek.2011 <- merge(train.data.perWeek.2011, train.data.perWeek.2011.tmp2)

train.data.perWeek.2013.pipiens <- train.data.perWeek.2013[train.data.perWeek.2013$Species == 1, ]
train.data.perWeek.2013.pipiensrestuans <- train.data.perWeek.2013[train.data.perWeek.2013$Species == 2, ]
train.data.perWeek.2013.restuans <- train.data.perWeek.2013[train.data.perWeek.2013$Species == 3, ]
train.data.perWeek.2013.tmp0 <- aggregate(WnvPresent ~ Week, train.data.perWeek.2013, sum)
train.data.perWeek.2013.tmp1 <- aggregate(Tavg ~ Week, train.data.perWeek.2013, mean)
train.data.perWeek.2013.tmp2 <- aggregate(Precip ~ Week, train.data.perWeek.2013, mean)
train.data.perWeek.2013 <- merge(train.data.perWeek.2013.tmp0, train.data.perWeek.2013.tmp1)
train.data.perWeek.2013 <- merge(train.data.perWeek.2013, train.data.perWeek.2013.tmp2)

spray.data.unique <- unique(spray.data$Date)
spray.data.years <- strftime(as.POSIXct(as.Date(spray.data.unique, "%Y-%m-%d")), format="%Y")
spray.data.weeks <- strftime(as.POSIXct(as.Date(spray.data.unique, "%Y-%m-%d")), format="%W")
spray.data.tmp <- data.frame(Year = spray.data.years, Week = spray.data.weeks)

spray.data.2007 <- NA
spray.data.2009 <- NA
spray.data.2011 <- spray.data.weeks[spray.data.years == 2011]
spray.data.2013 <- spray.data.weeks[spray.data.years == 2013]


par(mar=c(5, 4, 4, 5)+.1)
plot(train.data.perWeek.2013.pipiens$Week, train.data.perWeek.2013.pipiens$NumMosquitos,
	'l', col="orange", xlab="Week", ylab="NumMosquitoes", xlim=c(minWeek, maxWeek), ylim=c(0, maxNM))#, xaxt="n", yaxt="n", xlab="", ylab="")
lines(train.data.perWeek.2013.pipiensrestuans$Week, train.data.perWeek.2013.pipiensrestuans$NumMosquitos, 'l', col="purple")
lines(train.data.perWeek.2013.restuans$Week, train.data.perWeek.2013.restuans$NumMosquitos, 'l', col="green")#, xaxt="n", yaxt="n", xlab="", ylab="")
abline(v=spray.data.2013, col="red")
par(new=TRUE)
plot(train.data.perWeek.2013$Week, train.data.perWeek.2013$WnvPresent, 'b', col="black", xaxt="n", yaxt="n", xlab="", ylab="", xlim=c(minWeek, maxWeek), ylim=c(0, maxWP))
axis(4)
mtext("WnvPresent", side=4, line=3)
# legend("topleft",col=c("red","orange", "brown", "purple", "green", "blue", "black"),lty=1,legend=c("Sprayed", "Pipiens", "Pipiens/Restuans", "Restuans", "T_avg", "Precip", "WnvPresent"))
legend("topleft", col=c("red","orange", "purple", "green", "black"), lty=1, legend=c("Sprayed", "Pipiens", "Pipiens/Restuans", "Restuans", "WnvPresent"))


par(mar=c(5, 4, 4, 5)+.1, c(minWeek, maxWeek))
barplot(train.data.perWeek.2007$Precip, col="blue", xaxt="n", yaxt="n", xlab="Week", ylab="", ylim=c(minP, maxP))
axis(4)
mtext("Precipitation", side=4, line=3)
par(new=TRUE)
plot(train.data.perWeek.2007$Week, train.data.perWeek.2007$Tavg, 'l', col="red", xlab="", ylab="T_avg", xlim=c(minWeek, maxWeek), ylim=c(minT, maxT))
# plot(train.data.perWeek.2013$Week, train.data.perWeek.2013$Precip, 's', col="blue", xaxt="n", yaxt="n", xlab="", ylab="", xlim=c(minWeek, maxWeek), ylim=c(minP, maxP))
# axis(4)
legend("topleft",col=c("red","blue"),lty=1,legend=c("T_avg", "Precipitation"))
