# David O'Connor
# May, 2015
#
# Bubble plot of mosquito trap locations with weekly mosquito counts,
# WNV detection and spray locations by year.
# Circle - one circle for each week of year
# Circle size - Mosquito count
# Circle color - WNV Present

plotTraps = function(plotYear) {
  require(ggplot2)
  require(plyr)
  
  dataFolder = ".."
  dfSpray = read.csv(file.path(dataFolder,"spray.csv"))
  dfTrain = read.csv(file.path(dataFolder,"train.csv"))
  dfTrain$year = as.POSIXlt(dfTrain$Date)$year + 1900
  dfTrain$woy = as.POSIXlt(dfTrain$Date)$yday / 7
  dfSpray$year = as.POSIXlt(dfSpray$Date)$year + 1900
  dfTrain = subset(dfTrain,dfTrain$year %in% plotYear)
  dfSpray = subset(dfSpray,dfSpray$year %in% plotYear)
  # merge records for same trap, same week
  dfTrap = ddply(dfTrain,.(Trap,woy,year,Longitude,Latitude),summarize,
                 NumMosquitos = sum(NumMosquitos),
                 WnvPresent = ifelse(is.na(match(1,WnvPresent)),"no","yes"))
  trapPlot = ggplot(dfTrap) +
    coord_cartesian(xlim=c(-87.97,-87.5),ylim=c(41.62,42.06)) +
    geom_point(aes(x=Longitude,y=Latitude), data=dfSpray,size=4, colour="#ffcaca",alpha=1) +
    geom_point(aes(x=Longitude,y=Latitude, size=NumMosquitos, colour=WnvPresent),shape=1) +
    scale_size(range=c(4,50)) +
    guides(size=FALSE) +
    guides(colour=guide_legend(title="West Nile Virus Present", override.aes=list(size=4))) +
    scale_color_manual(name="",values=c("black","red")) +
    theme(panel.background=element_rect(fill="grey95"), panel.grid.minor=element_blank(),
          panel.grid.major=element_blank(), legend.position="bottom") +
    ggtitle("West Nile Virus Detection by Trap")
  if (length(plotYear > 1)) {
    trapPlot = trapPlot + facet_wrap(~year,ncol=2)
  }
  fname = paste("TrapPlot",paste(plotYear,collapse="_"),".png",sep="_")
  png(fname,width=7,height=8,units="in",res=72)
  print(trapPlot)
  dev.off()
}

plotTraps(2013)

