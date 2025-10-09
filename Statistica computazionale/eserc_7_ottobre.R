rm(list=ls()) 
set.seed(349152)

#esercizio 1.1

#punto 1
mu<-1
sigma<-3
l<- -10
u<- 5

dtruncnorm = function(x, mu, sigma, l, u){
  
  pl = pnorm(l, mu, sigma)
  pu = pnorm(u, mu, sigma)
  
  return(dnorm(x, mu, sigma)/(pu - pl))
  
}

x <- seq(l, u, 0.02)
y <- dtruncnorm(x,mu, sigma, l, u)

plot(x,y, type="l")

#punto 2

#punto 3
n <- 1000
x_sim <- runif(n,l,u)
x_atteso <- sum(x_sim * dtruncnorm(x_sim, mu, sigma, l, u) / dunif(x_sim,l,u)) / n

#punto 4
a <- 1
b <- 1
y_beta <- rbeta(a,b)
dbetascal = function(x, l, u){
  return((u-l)*x + l)
}


#esercizio 1.2

#punto 1 e punto 2 e punto 3
rm(list=ls()) 
set.seed(349152)

mu<-1
sigma<-3
l<- -10
u<- 5

dtruncnorm = function(x, mu, sigma, l, u){
  
  pl = pnorm(l, mu, sigma)
  pu = pnorm(u, mu, sigma)
  
  return(dnorm(x, mu, sigma)/(pu - pl))
  
}

m <- dtruncnorm(mu, mu, sigma, l, u)
n <- 1000
y <- runif(n,l,u)
u_sim <- runif(n,0,m)

X <- y[u_sim < dtruncnorm(y, mu, sigma, l, u)]
U <- u_sim[u_sim < dtruncnorm(y, mu, sigma, l, u)]

# plot densita ’
xseq = seq(-10 ,5 , by =0.01)
plot (
  xseq ,dtruncnorm(xseq, mu, sigma, l, u) , ylim =c(0 , m) ,
  type ="l " , lwd =2 ,
)
points (y ,u_sim , pch =20 , cex = 0.1)
points (X ,U , pch =20 , cex = 0.1 , col =2)
lines ( density(X , from =l , to = u) , col =2 , lwd =2)

p_acc_teorica <- 1/((u-l)*m)
p_acc_pratica <- length(X)/length(y)
  
  
  
  
  
  
  
  
  
  
  
  