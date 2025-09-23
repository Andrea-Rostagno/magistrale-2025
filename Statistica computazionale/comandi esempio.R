set.seed (100)
runif (10 ,0 ,1)
set.seed (100)
runif (10 ,0 ,1)
runif (10 ,0 ,1)

# simuliamo dalla beta
x1 = rbeta (10 ,1 ,3)
x2 = rbeta (50 ,1 ,3)
x3 = rbeta (100 ,1 ,3)
# cacoliamo le funzioni di ripartizioni empiriche su xseq
xseq = seq (0 ,1 , by =0.001)
ec1 = ecdf ( x1 )( xseq )
ec2 = ecdf ( x2 )( xseq )
ec3 = ecdf ( x3 )( xseq )
plot ( xseq , ec1 , type ="s" , lwd =2 , col =2 , xlab ="X" , ylab = " ecdf ")
lines ( xseq , ec2 , type = "s" , col =3 , lwd =2)
lines ( xseq , ec3 , type = "s" , col =4 , lwd =2)
lines ( xseq , pbeta ( xseq ,1 ,3) , type = "s" , col =1 , lwd =2)
legend ( " bottomright " , c( " True " ,"n =10 " ,"n =50 " ,"n =100 ")
         , col =1:4 , lwd =3 , cex =2
)

