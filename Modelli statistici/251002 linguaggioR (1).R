
#   ---------------------------------------------  #
#       MODELLI STATISTICI - ESERCITAZIONE 1       #
#               Martina Amongero, 02/10/25.        #
#   ---------------------------------------------  #

#############################
# 1 - Concetti di base     #
#############################

# Cominciamo dall'elaborazione minima ovvero: usare R come calcolatrice
2+2

# oggetti di R                                    
a
a=2+2
a

# in R le assegnazioni possono avvenire col simbolo = oppure col simbolo <-
a1<-100

# quali sono gli oggetti memorizzati nella console?
ls()

# dove stiamo lavorando? Ossia in quale directory di lavoro (working directory) stiamo memorizzando i nostri dati?
getwd()

# cancellare la History
rm(list=ls()) 

# proviamo a cambiare la directory di lavoro
# tutto cio' che salveremo da adesso in poi, verra' memorizzato in questa directory di lavoro.
setwd("C:/Users/marti/OneDrive/Desktop/modelli_statistici/")
getwd()



#############################
# 2 - Allocare le variabili #
#############################
# Ci sono 5 tipi base:
#   numeric (numeri reali)
#   integer (numero interi)
#   logical (booleani)
#   character 
#   complex (numeri complessi)

a <- 1
type.a <- class(a)
print(paste("La variabile 'a' è di tipo ", type.a))

a.int <- 1L
type.a.int <- class(a.int)
print(paste("La variabile 'a.int' è di tipo ", type.a.int))

b <- 0.85
type.b <- class(b)
print(paste("La variabile 'b' è di tipo ", type.b))

c <- -.6
type.c <- class(c)
print(paste("La variabile 'c' è di tipo ", type.c))

char <- 'a'
type.char <- class(char)
print(paste("La variabile 'char' è di tipo ", type.char))

compl <- complex(real = 3, imaginary = 4)
type.compl<- class(compl)
print(paste("La variabile 'char' è di tipo ", type.compl))

str <- "modelli statistici"
type.str <- class(str)
print(paste("La variabile 'str' è di tipo ", type.str))

bool <- TRUE
type.bool <- class(bool)
print(paste("La variabile 'bool' è di tipo ", type.bool))

# operazioni logiche
bool_a <- T
bool_b <- F
bool_a | bool_b 
bool_a | !bool_b 
bool_a & bool_b
bool_a & !bool_b
bool_a==0
bool_a==1

# Valori speciali 
  # NA: “not available”, missing value
  # Inf: infinity
  # NaN: “not-a-number”, undefined value

na <- NA
type.na <- class(NA)
print(paste("La variabile 'na' è di tipo ", type.na))

nan <- NaN
type.nan <- class(NA)
print(paste("La variabile 'nan' è di tipo ", type.nan))

print(paste(
  "1/0 = ", 1 / 0,
  ", 0/0 = ", 0 / 0
))

# NOTE: Inf può apparire anche in caso dinumerical overflow
exp(1000)

# I tipi di variabili possono essere convertiti con as.<typename>() (come funzioni, purché la conversione abbia senso)
v <- TRUE
w <- "0"
x <- 3.2
y <- 2L
z <- "F"
cat(paste(
  paste(x, as.integer(x), sep = " => "),    #  numeric -> integer
  paste(y, as.numeric(y), sep = " => "),    #  integer -> numeric
  paste(y, as.character(y), sep = " => "),  #  integer -> character
  paste(w, as.numeric(w), sep = " => "),    #  number-char -> numeric
  paste(v, as.numeric(v), sep = " => "),    #  logical -> numeric
  sep = "\n"
))

as.numeric(z)


#############################
# 3 - Salvare i risultati   #
#############################
# salviamo l'intero workspace dando un nome al nostro file
save.image("Lez1.RData")

# salviamo solo alcuni elementi di interesse
save(a,b,file="Lez1a.RData")


#############################
# 4 - Vettori               #
#############################
# I vettori si costruiscono con la funzione c(). Un vettore contiene valori dello stesso tipo.
vec1 <- c(4, 3, 9, 5, 8)
vec1

# operazioni su vettori
vec2 <- vec1 - 1 # sottrarre 1 a tutti 1 valori
sum(vec1)        # somma
cumsum(vec1)
vec1*vec1
vec1%*%t(vec1)
t(vec1)%*%vec1
mean(vec2)
sort(vec1, decreasing = TRUE) # ordinare gli elementi in ordine decrescente

# Range vectors (a passo unitario) si costruiscono con la sintassi start:end.
# Nota: il tipo dei vettori intervallo è integer, non numeric.
x_range <- 1:10
class(x_range)

# Selezionare solo alcuni elementi
vec1[1:3] 
vec1[c(1, 3)]
vec1[-c(1, 3)]
vec1[seq(1, length(vec1), 2)] 

which(vec1 == 3) # per trovare un elemento in un vettore e ottenerne l’indice/gli indici, si può usare la funzione
which(vec1 < 5)

is.element(1,vec1)
is.element(3,vec1)

vec1[which(vec1 >= 5)] # per filtrare solo i valori che soddisfano una certa condizione
vec1[vec1 >= 5]

#############################
# 5 - matrici               #
#############################
mat1 <- matrix(1:24,
               nrow = 6, ncol = 4)
mat1      # riempe per colonne (default)
mat2 <- matrix(1:24,
               nrow = 6, ncol = 4, byrow = TRUE)
dim(mat2) # ottenere la dimensione
dim(mat2)[1]
dim(mat2)[2]
nrow(mat2)
ncol(mat2)

# accedere agli elementi (ome visto per i vettori)
mat2[3, ]
mat2[,3]
mat2[1:2, 1:2]
diag(mat2)
t(mat2) # matrice trasposta 

# identita'
diagonal_mat <- diag(nrow = 4)

#############################
# 6 - array                 #
#############################
arr1 <- array(1:24, dim = c(2, 4, 3))
arr1

dim(arr1)
arr1[,2,]
dim(arr1[,2,])


#############################
# 7 - Liste e Dataframe     #
#############################

# Le liste sono contenitori che possono contenere tipi di dati diversi. 

list1  <- list(1:3, TRUE, x = c("a", "b", "c"))
list1[[3]]
list1$x

# I data frame sono collezioni di colonne che hanno la stessa lunghezza. 
# A differenza delle matrici, le colonne di un data frame possono essere di tipi diversi.
# Sono il modo più comune di rappresentare dati strutturati e la maggior parte dei dataset viene memorizzata in data frame.

df1 <- data.frame(x = 1, y = 1:10,
                  char = sample(c("a", "b"), 10, replace = TRUE))
colnames(df1)=c('c1','c2','c3')
df1$c1   # accesso con nome
df1[, 3] # accesso stile matrice 
c1
attach(df1)
c1
detach(df1)
c1

####################################
# 8 - Pacchetti ed help in R       #
####################################
# da interfaccia di R studio :  Tools-> Install Packages-> <select package> -> install
install.packages("mvtnorm") # installazione del pacchetto
library(mvtnorm).           # upload (anche da interfaccia)

help.start() # Browser

# Per quanto riguarda la documentazione sulla sintassi di specifici comandi ed i relativi argomenti opzionali 
# sono utili i comandi in linea
  # help()
  # help.search()
help(quit)
#oppure
?quit

# altre operazioni numeriche e con quale sintassi?
?Arithmetic
help(Arithmetic)
help(Trig)
help(Special)

#############################
# 9 - Plot                  #
#############################

# Base plot
n_points <- 20
x <- 1:n_points
y <- 3 * x + 2 * rnorm(n_points)

plot(x, y)
abline(a = 0, b = 3, col = "red")

# GGplot
library(ggplot2)
library(tibble)
gg_df <- tibble(x = x, y = y)

ggplot(gg_df) +
  geom_point(mapping = aes(x, y)) +
  geom_abline(mapping = aes(intercept = 0, slope = 3), color = "red")


##############################
# 10 - Densità e V.A.        #
##############################

# campionamento i.i.d.: r, p, q, d
?Normal
n = 1000
x = rnorm(n,0,1)
dnorm(x,0,1)
y = runif(n,0,1)
plot(x,y)
par(mfrow=c(1,2))
plot(density(x))
lines(seq(-4,4,0.1),dnorm(seq(-4,4,0.1),0,1),col='red')
plot(density(y))
lines(seq(0,1,0.01),dunif(seq(0,1,0.01),0,1),col='red')
boxplot(x,y)

boxplot(x)
abline(h=qnorm(0.5,0,1))
abline(h=quantile(x,prob=0.5),col='red',lty=2)

x1 = rnorm(n,0,1)
x2 = rnorm(n,2,1)
x3 = rnorm(n,0.0001,1)
t.test(x1,x2)
t.test(x1,x3)

rnorm(4,0,1)
rnorm(4,0,1)

# come ottengo la riproducibilità?
set.seed(1)
rnorm(4,0,1)
set.seed(1)
rnorm(4,0,1)

##############################
# 11 - funzioni user-defined #
##############################
# scriviamo una funzione: nome_funzione <- function(param1,param2,..)

# def
ciao_nome <- function(nome) {
  messaggio <- paste("Ciao,", nome, "! Benvenuto in R ")
  return(messaggio)
}
# uso la funzione
ciao_nome('martina')

area_cerchio <- function(raggio) {
  area <- pi * raggio^2
  return(area)
}

area_cerchio(3)   
area_cerchio(5)


#######################
# 12 - IF-ELSE e Loop #
######################

# if-else 
rnd <- rbinom(1,1,0.5)
if(rnd==0){
  print(paste(rnd, " <- Hai perso"))
}else{
  print(paste(rnd, " <- Hai vinto"))
}


# for loop
rnd <- rbinom(100,1,0.1)
for(i in 1:length(rnd)){
  if(rnd[i]==0){
    print(paste(rnd[i], " <- Hai perso"))
  }else{
    print(paste(rnd[i], " <- Hai vinto"))
  }
}


# while loop
rnd <- 0
k <- 0
while (rnd==0) {
  rnd <- rbinom(1,1,0.1)
  if(rnd==0){
    print(paste(rnd, " <- Hai perso"))
  }else{
    print(paste(rnd, " <- Hai vinto"))
  }
  k=k+1
}
print(paste("Hai avuto il primo successo al tentativo ", k))



##############################
# 13 - esercizi              #
##############################

## ESERCIZIO 1
# Lanciamo i dadi: simuliamo il lancio di 2 dadi a 6 facce per 100 volte.
# Calcoliamo la somma dei due dadi a ogni lancio.
# Visualizziamo la distribuzione delle somme con un istogramma.
# Calcola la probabilità stimata di ottenere un 7.
# Calcola la probabilità stimata di ottenere un 6, da un solo dado.

# 1. Simuliamo i lanci
set.seed(123)  # per avere risultati ripetibili
dado1 <- sample(1:6, 100, replace = TRUE)
dado2 <- sample(1:6, 100, replace = TRUE)

# 2. Somma dei dadi
somma <- dado1 + dado2

# 3. Istogramma
hist(somma, breaks = 11, col = "skyblue", 
     main = "Distribuzione delle somme dei dadi",
     xlab = "Somma dei due dadi")

# 4. Probabilità stimata di ottenere 7
prob_7 <- mean(somma == 7)
print(paste("Probabilità stimata di 7:", round(prob_7, 3)))


# 4b. 
prob_6=c()
for(i in 1:10000)
{
  dado1 <- sample(1:6, i, replace = TRUE)
  prob_6[i] <- mean(dado1 == 6)
}
plot(1:10000,prob_6)

## ESERCIZIO 2
# Ripasso: p-value
# Simula due gruppi di dati (50 valori ciascuno) con la stessa media.
# Ripeti il test t molte volte (es. 1000) e conta quante volte ottieni p-value < 0.05.
# Confronta il risultato con il livello di significatività alpha = 0.05.

set.seed(123)
risultati <- replicate(10000, {
  x <- rnorm(20, 50, 5)
  y <- rnorm(20, 50, 5)
  t.test(x,y)$p.value < 0.05
})
mean(risultati)

## ESERCIZIO 3
# Intervalli di confidenza   
conf.int <- function(data, alpha, type, sd=NA, sd.known=TRUE){
  CI <- array(NA,dim=c(1,2))
  if(type=="normal"){
    mean <- mean(data)
    N <- length(data)
    if(sd.known==TRUE){
      z <- qnorm(c(alpha/2,1-alpha/2))
      CI <- mean + (z*sd)/sqrt(N)
    }else{
      sd <- sqrt(var(data))
      t <- qt(c(alpha/2,1-alpha/2), N-1)
      CI <- mean + (t*sd)/sqrt(N)
    }
  }
  if(type=="binomial"){
    p <- mean(data)
    sd <- sqrt(p*(1-p))
    N <- length(data)
    t <- qt(c(alpha/2,1-alpha/2), N-1)
    CI <- p + (t*sd)/sqrt(N)
  }
  return(CI)
}
perc.cover <- c()


# Normale con varianza nota
type <- "normal"
cover <- c()

for(k in 1:10000){
  dati <- rnorm(100)
  CI <- conf.int(data=dati, alpha=0.05, type=type, sd=1)
  cover[k] <- CI[1]<0 & CI[2]>0
}
perc.cover[1] <- mean(cover)

# Normale con varianza non nota
type <- "normal"
cover <- c()

for(k in 1:10000){
  dati <- rnorm(10)
  CI <- conf.int(data=dati, alpha=0.05, type=type, sd=1, sd.known = FALSE)
  cover[k] <- CI[1]<0 & CI[2]>0
}
perc.cover[2] <- mean(cover)


#############################
# 14 - Uscire da R          #
#############################

### Uscire: ci sono due diverse possibilita'
### 1) chiudiamo dil programma : Salva area di lavoro? SI NO
### 2) comando q()
### 1) e 2) salvano i seguenti oggetti:
## - File .RData : contiene ciò che abbiamo eseguito 
## - File .Rhistory: i comandi eseguiti (script)
