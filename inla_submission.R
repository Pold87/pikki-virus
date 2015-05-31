library(pROC)
library(INLA)
library(plyr)

X_train <- read.csv("train_filled_new.csv")
X_train$NumMosquitos <- NULL

X_test <- read.csv("test_filled_new.csv")
X_test$WnvPresent <- NA
X_test$Id <- NULL

use.inla <- TRUE

    
X_test$Species <- gsub("UNSPECIFIED CULEX", "CULEX PIPIENS", X_test$Species)
    
df <- rbind(X_train, X_test)

if (use.inla) {

    model <- inla(WnvPresent ~ Species
                  + Block
                                        #+ Trap
                  + Latitude
                  + Longitude

                  + X9_week_avrgPrecipTotal
                  + X9_week_avrgTavg
    
                  + X7_week_avrgPrecipTotal
                  + X7_week_avrgTavg
                  
                  + Latitude * Longitude, 
 
              
                                        #+ Month * Species
                  
                  data=df,
                  control.predictor = list(link = 1))

    ypred <- model$summary.fitted.values$mean[(length(X_train$Trap) + 1) : (length(X_train$Trap) + length(X_test$Trap))]
} else {
    
    model <- glm(WnvPresent ~ Species
                 + Block
                 #+ Trap # Seems to introduce new levels!
                 + Latitude
                 + Longitude
                 #+ X9_week_avrgPrecipTotal
                 #+ X9_week_avrgTavg

                 #+ X7_week_avrgPrecipTotal
                 #+ X7_week_avrgTavg
                  
                 #+ Latitude * Longitude

                 
                  #+ Month * Species
                 
               , data=X_train)
    ypred <- predict(model, X_test)        
}


write.csv(ypred, "output_inla.csv")
