library(pROC)
library(INLA)
library(plyr)

basedir <- "data_per_year/"

use.inla <- FALSE

for (year in c('2007', '2009', '2011', '2013')){

    print(year)

    X_train <- read.csv(paste(basedir, year, "X_train.csv", sep=''))
    X_test <- read.csv(paste(basedir, year, "X_test.csv", sep=''))

    y_train <- read.csv(paste(basedir, year, "y_train.csv", sep=''))
    y_test <- read.csv(paste(basedir, year, "y_test.csv", sep=''))
    y_test <- y_test$WnvPresent

    df_train <- cbind(X_train, y_train)
    df_test <- cbind(X_test, y_test)

    df_test <- rename(df_test, c("y_test"="WnvPresent"))
    
    # X_test$Species <- gsub("UNSPECIFIED CULEX", "CULEX PIPIENS", X_test$Species)

    df <- rbind(df_train, df_test)

    results = c(0, 0, 0, 0)
    names(v) = c("2007", "2009", "2011", "2013")

    if (use.inla) {
    model <- inla(WnvPresent ~ Species
                 + Block
                 + Trap
                 + Latitude
                 + Longitude
                 + X9_week_avrgPrecipTotal
                 + X9_week_avrgTavg

                 + X7_week_avrgPrecipTotal
                 + X7_week_avrgTavg
                  
                 + Latitude * Longitude


                  + Month * Species
                  
                , data=df,
                  control.predictor = list(link = 1))
    #ypred <- predict(model, X_test)
    ypred <- model$summary.fitted.values$mean[(length(X_train$Trap) + 1) : (length(X_train$Trap) + length(X_test$Trap))]
}
    else {

        model <- glm(WnvPresent ~ 
                         + Block
                     + Species
#                      + Trap

                         + X3_week_avrgPrecipTotal * X3_week_avrgTavg
#                     + X4_week_avrgPrecipTotal * X4_week_avrgTavg
#                         + X5_week_avrgPrecipTotal * X5_week_avrgTavg
#                      + X6_week_avrgPrecipTotal * X6_week_avrgTavg
                         + X7_week_avrgPrecipTotal * X7_week_avrgTavg
#                      + X8_week_avrgPrecipTotal * X8_week_avrgTavg
                     + X9_week_avrgPrecipTotal * X9_week_avrgTavg
#                      + X15_week_avrgPrecipTotal * X15_week_avrgTavg
#                     + X12_week_avrgPrecipTotal * X12_week_avrgTavg
#                      + X10_week_avrgPrecipTotal * X10_week_avrgTavg
                     
#                     + X7_week_avrgPrecipTotal * X7_week_avrgTavg
#                       + WetBulb
#                      + ResultSpeed * ResultDir
#                       + Sunset * Sunrise
                     + Latitude * Longitude
#                     + Species * Month
                  
                , data=df_train, family=binomial)
    ypred <- predict(model, X_test)   
        }

    # Calculate AUC score
    g <- roc(y_test, ypred)
    v[year] = g$auc
    print(g)

    
}

print(mean(unlist(v)))
