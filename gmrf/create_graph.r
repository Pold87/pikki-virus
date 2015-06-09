

train <- read.csv("train.csv", header = T)
graph.data <- data.frame(Trap=train$Trap, Longitude=train$Longitude, Latitude=train$Latitude)

traps <- unique(graph.data)
n.nodes <- nrow(traps)

traps.id <- 1:n.nodes# as.numeric(substr(traps$Trap, 2, 4))# + (substr(traps$Trap, 5, 5) != "") * 300
# traps.data <- c(traps$Trap, rep(n.nodes-1, n.nodes))
traps.data <- matrix(rep(t(traps.id), n.nodes+1), byrow=TRUE, nrow=n.nodes)
traps.data[, 3:139] <- traps.data[, 2:138]
traps.data[, 2] <- as.character(n.nodes - 1)

graph.file.name <- "traps.graph"
if (file.exists(graph.file.name))
	file.remove(graph.file.name)

write(n.nodes, file = graph.file.name, ncolumns = 1, append = TRUE, sep = " ")
# write(t(traps.id), file = graph.file.name, ncolumns = n.nodes, append = TRUE, sep = " ")
write(t(traps.data), file = graph.file.name, ncolumns = n.nodes+1, append = TRUE, sep = " ")
