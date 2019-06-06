nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

abs(X,Y,Z) :- digit(X,X2), digit(Y,Y2), Z is max(X2-Y2,Y2-X2).