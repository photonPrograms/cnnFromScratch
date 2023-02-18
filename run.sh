javac ImageOps.java MatrixOps.java PrettyPrint.java
javac Conv3x3.java MaxPooling.java Dense.java CNN.java
javac Driver.java
java Driver
gnuplot -p plotEpochAcc.p
gnuplot -p plotEpochLoss.p
gnuplot -p plotStepAcc.p
gnuplot -p plotStepLoss.p