#Written by Le Yan, April 2017
#quick cheap fix to creating cluster profile radial plots
library(ggplot2)
library(plotrix)
source("http://onepager.togaware.com/CreateRadialPlot.R") #try using create radial plot function

rm(list=ls())
default.par <- par() 

setwd("C:/Users/Le Yan/Documents/KCL 2016-17/Applied Geocomputation/Assignment 2/outputs")
df <- read.csv("lsoa_model.csv", header = T)

df <- subset(df, select = -c(X, lsoa11nm, lsoa11nmw, objectid, st_areasha, st_lengths) )
kmedian <- data.frame(aggregate(df[, 1:12], list(df$KMeans), median)) #find median of each cluster group
CreateRadialPlot(kmedian, grid.min=-2, grid.max=2, plot.extent.x=1.5) #ok, but messy

#kind of inelegant fix to excluding Group.1
label <- data.frame(aggregate(df[, 1:12], list(df$KMeans), median))
label <- subset(label, select = -c(Group.1))
label.names <- names(label) #create labels

zero <- kmedian[1,]
one <- kmedian[2,]
two <- kmedian[3,]
three <- kmedian[4,]


zero <- subset(zero, select = -c(Group.1))
one <- subset(one, select = -c(Group.1))
two <- subset(two, select = -c(Group.1))
three <- subset(three, select = -c(Group.1))

#but i have no idea how to loop over label names so...
par(ps=10) #set font size
radial.plot(zero, labels=label.names,rp.type="p",main="Group 0", radial.lim=c(-2, 3),line.col="blue")
radial.plot(one, labels=label.names,rp.type="p",main="Group 1", radial.lim=c(-2, 3),line.col="blue")
radial.plot(two, labels=label.names,rp.type="p",main="Group 2", radial.lim=c(-2, 3),line.col="blue")
radial.plot(three, labels=label.names,rp.type="p",main="Group 3", radial.lim=c(-2, 3),line.col="blue")


"
#create loop ...doesn't seem to be working so find a fix later
dflist <- list(zero = zero, one = one, two = two, three = three)
dflist <- lapply(
dflist, 
function(x) 
{ 
x <- subset(x, select = -c(Group.1)) #remove group columns
radial.plot(x, labels=kmedian.names, rp.type="p", main="Group", radial.lim=c(-2, 3), line.col="blue")
}
)

for (x in dflist){
x <- subset(x, select = -c(Group.1))
}
"