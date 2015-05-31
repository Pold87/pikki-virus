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

            model <- glm(WnvPresent ~ Species
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
                  
                , data=df_train)
    ypred <- predict(model, X_test)        
        }

    # Calculate AUC score
    g <- roc(y_test, ypred)
    print(g)

    
}
